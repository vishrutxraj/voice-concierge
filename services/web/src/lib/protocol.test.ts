import { describe, expect, it } from "vitest";
import {
  callReducer,
  describeServerError,
  initialCallState,
  isEnglish,
  languageName,
  type CallAction,
  type CallState,
  type ServerEvent,
} from "./protocol";

const run = (actions: CallAction[], from: CallState = initialCallState) =>
  actions.reduce(callReducer, from);
const server = (event: ServerEvent): CallAction => ({ type: "server", event });

describe("callReducer", () => {
  it("moves connecting -> ready when the server says ready, keeping the session id", () => {
    const s = run([{ type: "connecting" }, server({ type: "ready", session_id: "abc" })]);
    expect(s.status).toBe("ready");
    expect(s.sessionId).toBe("abc");
  });

  it("records the transcript as a caller message and the reply as an agent message", () => {
    const s = run([
      server({ type: "ready", session_id: "s" }),
      { type: "listening" },
      { type: "utterance_sent" },
      server({ type: "transcript", text: "where is my order", language: "hi-IN" }),
      server({
        type: "reply",
        text: "आपका ऑर्डर",
        text_en: "Your order",
        language: "hi-IN",
        audio: ["native"],
        intent: "order_lookup",
      }),
    ]);
    expect(s.messages.map((m) => m.role)).toEqual(["caller", "agent"]);
    const agent = s.messages[1];
    expect(agent.role === "agent" && agent.text).toBe("आपका ऑर्डर");
    expect(agent.role === "agent" && agent.textEn).toBe("Your order");
    expect(s.status).toBe("processing"); // audio is coming; the player flips to speaking
  });

  it("returns to ready immediately when a reply announces no audio", () => {
    const s = run([
      { type: "utterance_sent" },
      server({ type: "reply", text: "x", text_en: "x", language: "en-IN", audio: [] }),
    ]);
    expect(s.status).toBe("ready");
  });

  it("tracks a pending confirmation and clears it on the next reply", () => {
    const pending = server({
      type: "reply",
      text: "Confirm?",
      text_en: "Confirm?",
      language: "en-IN",
      audio: ["english"],
      awaiting_confirmation: { kind: "confirm_reschedule", prompt: "Confirm?" },
    });
    const done = server({
      type: "reply",
      text: "Done",
      text_en: "Done",
      language: "en-IN",
      audio: ["english"],
      awaiting_confirmation: null,
    });
    expect(run([pending]).awaitingConfirmation?.kind).toBe("confirm_reschedule");
    expect(run([pending, done]).awaitingConfirmation).toBeNull();
  });

  it("only leaves 'speaking' on playback_finished, not from other states", () => {
    expect(run([{ type: "playback_started" }, { type: "playback_finished" }]).status).toBe("ready");
    expect(run([{ type: "listening" }, { type: "playback_finished" }]).status).toBe("listening");
  });

  it("assigns unique increasing message ids", () => {
    const s = run([
      server({ type: "transcript", text: "a", language: null }),
      server({ type: "transcript", text: "b", language: null }),
    ]);
    expect(s.messages.map((m) => m.id)).toEqual([1, 2]);
  });

  it("turns a tts_unavailable server error into a readable message", () => {
    const s = run([server({ type: "error", detail: "tts_unavailable" })]);
    expect(s.error).toMatch(/text reply is still shown/i);
  });

  it("clears the previous conversation on a new call", () => {
    const s = run([
      server({ type: "transcript", text: "a", language: null }),
      { type: "connecting" },
    ]);
    expect(s.messages).toEqual([]);
    expect(s.status).toBe("connecting");
  });

  it("ignores audio-only events (the player handles them)", () => {
    const before = run([server({ type: "ready", session_id: "s" })]);
    expect(callReducer(before, server({ type: "audio_end" }))).toBe(before);
    expect(callReducer(before, server({ type: "barge_in" }))).toBe(before);
  });
});

describe("language helpers", () => {
  it("names known languages and falls back to the raw code", () => {
    expect(languageName("hi-IN")).toBe("Hindi");
    expect(languageName("xx-YY")).toBe("xx-YY");
    expect(languageName(null)).toBe("Unknown");
  });

  it("treats en-* and missing as English", () => {
    expect(isEnglish("en-IN")).toBe(true);
    expect(isEnglish("en")).toBe(true);
    expect(isEnglish(null)).toBe(true);
    expect(isEnglish("ta-IN")).toBe(false);
  });

  it("passes unknown server errors through unchanged", () => {
    expect(describeServerError("boom")).toBe("boom");
  });
});
