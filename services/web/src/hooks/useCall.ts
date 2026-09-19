import { useCallback, useEffect, useReducer, useRef, useState } from "react";
import { MicRecorder, type Capture } from "../lib/recorder";
import { ReplyPlayer } from "../lib/player";
import {
  callReducer,
  initialCallState,
  type AudioPreference,
  type ServerEvent,
} from "../lib/protocol";

export interface ExplainPayload {
  session_id: string;
  event_count: number;
  decisions: {
    at: number;
    node: string;
    chose: string;
    confidence: number | null;
    because: string | null;
    rejected: unknown[] | null;
  }[];
  tool_calls: { node: string; call: string; data: Record<string, unknown> }[];
  guardrail_checks: { node: string; verdict: string; data: Record<string, unknown> }[];
  escalated: boolean;
}

const PREF_KEY = "concierge.audioPreference";

function loadPreference(): AudioPreference {
  try {
    const v = localStorage.getItem(PREF_KEY);
    if (v === "native" || v === "english" || v === "both") return v;
  } catch {
    /* storage unavailable (private mode) -- fall through */
  }
  return "native";
}

function wsUrl(): string {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${location.host}/ws/call`;
}

export function useCall() {
  const [state, dispatch] = useReducer(callReducer, initialCallState);
  const [level, setLevel] = useState(0);
  const [audioPreference, setAudioPreferenceState] = useState<AudioPreference>(loadPreference);
  const [playing, setPlaying] = useState<"native" | "english" | null>(null);
  const [explain, setExplain] = useState<ExplainPayload | null>(null);
  const [canReplay, setCanReplay] = useState(false);

  const wsRef = useRef<WebSocket | null>(null);
  const recorderRef = useRef<MicRecorder | null>(null);
  const captureRef = useRef<Capture | null>(null);
  const sessionRef = useRef<string | null>(null);
  const playerRef = useRef<ReplyPlayer | null>(null);
  const prefRef = useRef(audioPreference);
  prefRef.current = audioPreference;

  if (!playerRef.current) {
    playerRef.current = new ReplyPlayer({
      onTrackStart: (variant) => {
        setCanReplay(true);
        setPlaying(variant);
        dispatch({ type: "playback_started" });
      },
      onIdle: () => {
        setPlaying(null);
        dispatch({ type: "playback_finished" });
      },
    });
  }
  const player = playerRef.current;

  const refreshExplain = useCallback(async () => {
    const id = sessionRef.current;
    if (!id) return;
    try {
      const res = await fetch(`/api/sessions/${encodeURIComponent(id)}/explain`);
      if (res.ok) setExplain((await res.json()) as ExplainPayload);
    } catch {
      /* the explain panel is a nicety; never let it break a call */
    }
  }, []);

  const handleServerEvent = useCallback(
    (event: ServerEvent) => {
      dispatch({ type: "server", event });
      switch (event.type) {
        case "ready":
          sessionRef.current = event.session_id;
          break;
        case "reply":
          setCanReplay(false); // the previous reply's audio is superseded
          player.beginTurn(event.audio);
          void refreshExplain();
          break;
        case "audio_track":
          player.markTrack(event.variant);
          break;
        case "audio_end":
          player.endTurn();
          break;
        case "barge_in":
          player.stop();
          setPlaying(null);
          dispatch({ type: "playback_finished" });
          break;
      }
    },
    [player, refreshExplain],
  );

  const stopCall = useCallback(() => {
    captureRef.current?.cancel();
    captureRef.current = null;
    player.stop();
    setPlaying(null);
    setLevel(0);
    const ws = wsRef.current;
    wsRef.current = null;
    if (ws && ws.readyState === WebSocket.OPEN) {
      try {
        ws.send(JSON.stringify({ type: "hangup" }));
      } catch {
        /* closing anyway */
      }
    }
    ws?.close();
    recorderRef.current?.close();
    recorderRef.current = null;
  }, [player]);

  const startCall = useCallback(
    async (callerPhone: string) => {
      dispatch({ type: "connecting" });
      setExplain(null);
      sessionRef.current = null;
      player.reset(); // nothing from a previous call may be "repeated" in this one
      setCanReplay(false);
      player.unlock(); // user gesture: satisfy the browser's autoplay policy

      try {
        const recorder = new MicRecorder();
        await recorder.open();
        recorderRef.current = recorder;
      } catch (err) {
        dispatch({
          type: "failed",
          message:
            err instanceof DOMException && err.name === "NotAllowedError"
              ? "Microphone permission was denied. Allow it in your browser and try again."
              : err instanceof Error
                ? err.message
                : "Could not open the microphone.",
        });
        return;
      }

      const ws = new WebSocket(wsUrl());
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () =>
        ws.send(
          JSON.stringify({
            type: "start",
            caller_phone: callerPhone,
            transport_kind: "browser",
            audio: prefRef.current,
          }),
        );
      ws.onmessage = (msg) => {
        if (typeof msg.data === "string") {
          try {
            handleServerEvent(JSON.parse(msg.data) as ServerEvent);
          } catch {
            /* ignore a malformed control frame */
          }
        } else {
          player.push(msg.data as ArrayBuffer);
        }
      };
      ws.onerror = () => dispatch({ type: "failed", message: "Could not reach the voice gateway." });
      ws.onclose = () => {
        if (wsRef.current === ws) {
          wsRef.current = null;
          recorderRef.current?.close();
          recorderRef.current = null;
          player.stop();
          setPlaying(null);
          dispatch({ type: "closed" });
        }
      };
    },
    [handleServerEvent, player],
  );

  /** Tap-to-talk. Tapping while the agent is speaking is a barge-in. */
  const startListening = useCallback(async () => {
    const recorder = recorderRef.current;
    const ws = wsRef.current;
    if (!recorder || !ws || ws.readyState !== WebSocket.OPEN || captureRef.current) return;

    player.stop(); // barge-in: silence the agent the moment the caller starts
    setPlaying(null);
    dispatch({ type: "listening" });

    const capture = recorder.capture({ onLevel: setLevel });
    captureRef.current = capture;
    const wav = await capture.result;
    if (captureRef.current === capture) captureRef.current = null;

    if (!wav || ws.readyState !== WebSocket.OPEN) {
      dispatch({ type: "listen_cancelled" });
      return;
    }
    ws.send(wav);
    dispatch({ type: "utterance_sent" });
  }, [player]);

  /** Say the last reply again. Works while it is still playing (restarts it). */
  const replay = useCallback(() => {
    if (captureRef.current) return; // never talk over a caller who is speaking
    player.unlock();
    player.replay();
  }, [player]);

  const finishListening = useCallback(() => captureRef.current?.finish(), []);
  const cancelListening = useCallback(() => captureRef.current?.cancel(), []);

  const setAudioPreference = useCallback((pref: AudioPreference) => {
    setAudioPreferenceState(pref);
    try {
      localStorage.setItem(PREF_KEY, pref);
    } catch {
      /* non-fatal */
    }
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "set_audio", audio: pref }));
    }
  }, []);

  useEffect(() => () => stopCall(), [stopCall]);

  return {
    state,
    level,
    playing,
    explain,
    audioPreference,
    canReplay,
    replay,
    startCall,
    stopCall,
    startListening,
    finishListening,
    cancelListening,
    setAudioPreference,
  };
}
