/**
 * The /ws/call wire protocol, mirrored from the backend
 * (services/gateway/app/audio/websocket_transport.py and call_loop.py).
 * Keep this file and those in step -- it is the only place the client knows
 * the server's event shapes.
 */

export type AudioPreference = "native" | "english" | "both";
export type AudioVariant = "native" | "english";

export interface AwaitingConfirmation {
  kind: string;
  prompt: string;
  [key: string]: unknown;
}

export type ServerEvent =
  | { type: "ready"; session_id: string }
  | { type: "transcript"; text: string; language: string | null }
  | {
      type: "reply";
      text: string; // caller's language (== English for an English caller)
      text_en: string;
      language: string;
      audio: AudioVariant[];
      intent?: string | null;
      awaiting_confirmation?: AwaitingConfirmation | null;
      escalated?: boolean;
    }
  | { type: "audio_track"; variant: AudioVariant; language: string }
  | { type: "audio_end" }
  | { type: "barge_in" }
  | { type: "error"; detail: string };

export interface CallerMessage {
  id: number;
  role: "caller";
  /** ASR runs in translate mode, so this is English whatever was spoken. */
  textEn: string;
  language: string | null;
}

export interface AgentMessage {
  id: number;
  role: "agent";
  text: string;
  textEn: string;
  language: string;
  intent: string | null;
  escalated: boolean;
  awaitingConfirmation: AwaitingConfirmation | null;
}

export type Message = CallerMessage | AgentMessage;

export type CallStatus =
  | "idle"
  | "connecting"
  | "ready"
  | "listening"
  | "processing"
  | "speaking";

export interface CallState {
  status: CallStatus;
  sessionId: string | null;
  messages: Message[];
  awaitingConfirmation: AwaitingConfirmation | null;
  error: string | null;
  nextId: number;
}

export const initialCallState: CallState = {
  status: "idle",
  sessionId: null,
  messages: [],
  awaitingConfirmation: null,
  error: null,
  nextId: 1,
};

export type CallAction =
  | { type: "connecting" }
  | { type: "server"; event: ServerEvent }
  | { type: "listening" }
  | { type: "utterance_sent" }
  | { type: "listen_cancelled" }
  | { type: "playback_started" }
  | { type: "playback_finished" }
  | { type: "closed" }
  | { type: "failed"; message: string };

export function callReducer(state: CallState, action: CallAction): CallState {
  switch (action.type) {
    case "connecting":
      return { ...initialCallState, status: "connecting" };

    case "listening":
      return { ...state, status: "listening", error: null };

    case "listen_cancelled":
      return { ...state, status: "ready" };

    case "utterance_sent":
      return { ...state, status: "processing" };

    case "playback_started":
      return { ...state, status: "speaking" };

    case "playback_finished":
      return state.status === "speaking" ? { ...state, status: "ready" } : state;

    case "closed":
      return { ...state, status: "idle" };

    case "failed":
      return { ...state, status: "idle", error: action.message };

    case "server":
      return reduceServerEvent(state, action.event);
  }
}

function reduceServerEvent(state: CallState, event: ServerEvent): CallState {
  switch (event.type) {
    case "ready":
      return { ...state, status: "ready", sessionId: event.session_id };

    case "transcript":
      return {
        ...state,
        messages: [
          ...state.messages,
          { id: state.nextId, role: "caller", textEn: event.text, language: event.language },
        ],
        nextId: state.nextId + 1,
      };

    case "reply": {
      const awaiting = event.awaiting_confirmation ?? null;
      return {
        ...state,
        // Audio (if any) follows; the player flips us to "speaking". With no
        // audio coming the turn is simply over.
        status: event.audio.length === 0 ? "ready" : state.status,
        awaitingConfirmation: awaiting,
        messages: [
          ...state.messages,
          {
            id: state.nextId,
            role: "agent",
            text: event.text,
            textEn: event.text_en,
            language: event.language,
            intent: event.intent ?? null,
            escalated: event.escalated ?? false,
            awaitingConfirmation: awaiting,
          },
        ],
        nextId: state.nextId + 1,
      };
    }

    case "error":
      return { ...state, error: describeServerError(event.detail) };

    // audio_track / audio_end / barge_in are handled by the audio player, not
    // by conversation state.
    default:
      return state;
  }
}

export function describeServerError(detail: string): string {
  if (detail === "tts_unavailable") {
    return "Voice reply unavailable right now -- the text reply is still shown.";
  }
  return detail;
}

/** Display names, matching services/gateway/app/audio/localize.py. */
export const LANGUAGE_NAMES: Record<string, string> = {
  "en-IN": "English",
  "hi-IN": "Hindi",
  "bn-IN": "Bengali",
  "gu-IN": "Gujarati",
  "kn-IN": "Kannada",
  "ml-IN": "Malayalam",
  "mr-IN": "Marathi",
  "od-IN": "Odia",
  "pa-IN": "Punjabi",
  "ta-IN": "Tamil",
  "te-IN": "Telugu",
};

export function languageName(code: string | null | undefined): string {
  if (!code) return "Unknown";
  return LANGUAGE_NAMES[code] ?? code;
}

export function isEnglish(code: string | null | undefined): boolean {
  return !code || code.toLowerCase().startsWith("en");
}
