import { Loader2, Mic, PhoneOff, RotateCcw, Send } from "lucide-react";
import type { CallStatus } from "../lib/protocol";

interface Props {
  status: CallStatus;
  level: number;
  onTap: () => void;
  onEnd: () => void;
  /** There is a finished reply whose audio can be played again. */
  canReplay: boolean;
  onReplay: () => void;
}

const LABELS: Record<CallStatus, string> = {
  idle: "",
  connecting: "Connecting…",
  ready: "Tap to speak",
  listening: "Listening… tap to send",
  processing: "Thinking…",
  speaking: "Speaking… tap to interrupt",
};

export function MicDock({ status, level, onTap, onEnd, canReplay, onReplay }: Props) {
  const listening = status === "listening";
  // Repeating mid-question would talk over the caller, and mid-"thinking" the
  // reply it would repeat is about to be replaced.
  const replayEnabled = canReplay && (status === "ready" || status === "speaking");
  const busy = status === "connecting" || status === "processing";
  // Level ring: grows with mic volume while listening.
  const ringScale = listening ? 1 + Math.min(level * 9, 0.6) : 1;

  return (
    <div className="flex flex-col items-center gap-3 pb-2">
      <div className="flex items-center gap-3 sm:gap-6">
        <button
          type="button"
          onClick={onReplay}
          disabled={!replayEnabled}
          aria-label="Repeat the last reply"
          title="Repeat the last reply (R)"
          className="flex items-center gap-2 rounded-full border border-line px-4 py-2 text-sm text-muted transition-colors hover:text-ink focus-visible:outline-2 focus-visible:outline-accent disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:text-muted"
        >
          <RotateCcw size={16} aria-hidden /> <span className="hidden sm:inline">Repeat</span>
        </button>
        <div className="relative">
          {listening && (
            <span className="animate-ring absolute inset-0 rounded-full bg-accent/40" aria-hidden />
          )}
          <span
            className="absolute inset-0 rounded-full bg-accent/25 transition-transform duration-75"
            style={{ transform: `scale(${ringScale})` }}
            aria-hidden
          />
          <button
            type="button"
            onClick={onTap}
            disabled={status === "connecting"}
            aria-pressed={listening}
            aria-label={listening ? "Stop and send" : "Start speaking"}
            className={`relative grid size-20 place-items-center rounded-full border-2 transition-colors focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-accent disabled:opacity-60 ${
              listening
                ? "border-accent bg-accent text-bg"
                : "border-accent/60 bg-raised text-accent hover:bg-accent/15"
            }`}
          >
            {busy ? (
              <Loader2 className="animate-spin" size={30} />
            ) : listening ? (
              <Send size={28} />
            ) : (
              <Mic size={30} />
            )}
          </button>
        </div>
        <button
          type="button"
          onClick={onEnd}
          aria-label="End call"
          className="flex items-center gap-2 rounded-full border border-danger/50 px-4 py-2 text-sm text-danger transition-colors hover:bg-danger/15 focus-visible:outline-2 focus-visible:outline-danger"
        >
          <PhoneOff size={16} aria-hidden /> <span className="hidden sm:inline">End call</span>
        </button>
      </div>
      <p className="h-5 text-sm text-muted" aria-live="polite">
        {LABELS[status]}
      </p>
      <p className="hidden text-[11px] text-muted/70 sm:block">
        Tip: press <kbd className="rounded border border-line px-1">Space</kbd> to talk,{" "}
        <kbd className="rounded border border-line px-1">R</kbd> to repeat
      </p>
    </div>
  );
}
