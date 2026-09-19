import { useEffect, useState } from "react";
import { PanelRight, X } from "lucide-react";
import { useCall } from "./hooks/useCall";
import { languageName } from "./lib/protocol";
import { Conversation, ErrorNote } from "./components/Conversation";
import { ExplainPanel } from "./components/ExplainPanel";
import { HearToggle } from "./components/HearToggle";
import { MicDock } from "./components/MicDock";
import { StartScreen } from "./components/StartScreen";

export default function App() {
  const call = useCall();
  const { state } = call;
  const [showExplain, setShowExplain] = useState(false);

  const inCall = state.status !== "idle" && state.status !== "connecting";

  const onMicTap = () => {
    if (state.status === "listening") call.finishListening();
    else void call.startListening();
  };

  const canRepeatNow = call.canReplay && (state.status === "ready" || state.status === "speaking");

  // Space toggles talking, R repeats the last reply (both ignored while
  // typing in a form control).
  useEffect(() => {
    if (!inCall) return;
    const onKey = (e: KeyboardEvent) => {
      if ((e.code !== "Space" && e.code !== "KeyR") || e.repeat) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return; // leave browser shortcuts alone
      const tag = (e.target as HTMLElement | null)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "BUTTON") return;
      e.preventDefault();
      if (e.code === "KeyR") {
        if (canRepeatNow) call.replay();
      } else {
        onMicTap();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  if (!inCall) {
    return (
      <StartScreen
        connecting={state.status === "connecting"}
        error={state.error}
        audioPreference={call.audioPreference}
        onAudioPreference={call.setAudioPreference}
        onStart={(phone) => void call.startCall(phone)}
      />
    );
  }

  const lastLanguage = [...state.messages].reverse().find((m) => m.role === "caller")?.language;

  return (
    <div className="mx-auto flex h-full max-w-6xl gap-6 px-4 py-4">
      <main className="flex min-w-0 flex-1 flex-col">
        <header className="flex flex-wrap items-end justify-between gap-3 pb-3">
          <div>
            <h1 className="text-lg font-semibold">Delivery Voice Concierge</h1>
            <p className="flex items-center gap-2 text-xs text-muted">
              <span className="inline-block size-2 rounded-full bg-accent" aria-hidden />
              On a call
              {lastLanguage && <> · detected {languageName(lastLanguage)}</>}
            </p>
          </div>
          <div className="flex items-end gap-3">
            <HearToggle value={call.audioPreference} onChange={call.setAudioPreference} />
            <button
              type="button"
              onClick={() => setShowExplain((v) => !v)}
              aria-expanded={showExplain}
              className="flex items-center gap-1.5 rounded-full border border-line px-3 py-1.5 text-sm text-muted transition-colors hover:text-ink lg:hidden"
            >
              <PanelRight size={16} aria-hidden /> Why?
            </button>
          </div>
        </header>

        <div className="min-h-0 flex-1 overflow-y-auto rounded-2xl border border-line bg-bg/40 p-4">
          <Conversation
            messages={state.messages}
            playing={call.playing}
            awaitingPrompt={state.awaitingConfirmation?.prompt ?? null}
          />
        </div>

        {state.error && (
          <div className="pt-3">
            <ErrorNote message={state.error} />
          </div>
        )}

        <div className="pt-4">
          <MicDock
            status={state.status}
            level={call.level}
            onTap={onMicTap}
            onEnd={call.stopCall}
            canReplay={call.canReplay}
            onReplay={call.replay}
          />
        </div>
      </main>

      {/* Explain: side column on desktop, slide-over on small screens. */}
      <aside
        className={`${
          showExplain ? "fixed inset-y-0 right-0 z-10 flex w-[88%] max-w-sm" : "hidden"
        } flex-col gap-3 overflow-y-auto border-l border-line bg-surface p-4 lg:static lg:flex lg:w-80 lg:shrink-0 lg:rounded-2xl lg:border`}
        aria-label="How the agent decided"
      >
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold">Why the agent did that</h2>
          <button
            type="button"
            onClick={() => setShowExplain(false)}
            aria-label="Close panel"
            className="text-muted hover:text-ink lg:hidden"
          >
            <X size={18} />
          </button>
        </div>
        <ExplainPanel data={call.explain} />
      </aside>
    </div>
  );
}
