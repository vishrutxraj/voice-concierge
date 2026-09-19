import { useEffect, useRef } from "react";
import { Ear, Headset, TriangleAlert, Volume2 } from "lucide-react";
import { isEnglish, languageName, type AgentMessage, type CallerMessage, type Message } from "../lib/protocol";

interface Props {
  messages: Message[];
  /** Which rendering of the latest reply is being spoken right now. */
  playing: "native" | "english" | null;
  awaitingPrompt: string | null;
}

export function Conversation({ messages, playing, awaitingPrompt }: Props) {
  const endRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, awaitingPrompt]);

  const lastAgentId = [...messages].reverse().find((m) => m.role === "agent")?.id;

  return (
    <div className="flex flex-col gap-4 px-1 py-2" aria-live="polite">
      {messages.length === 0 && (
        <p className="mx-auto mt-16 max-w-sm text-center text-sm text-muted">
          Tap the microphone and speak in any supported Indian language or English. Try
          “where is my order?”
        </p>
      )}
      {messages.map((m) =>
        m.role === "caller" ? (
          <CallerBubble key={m.id} m={m} />
        ) : (
          <AgentBubble key={m.id} m={m} playing={m.id === lastAgentId ? playing : null} />
        ),
      )}
      {awaitingPrompt && (
        <div className="animate-rise mx-auto flex max-w-md items-center gap-2 rounded-full border border-warn/40 bg-warn/10 px-4 py-2 text-sm text-warn">
          <Ear size={16} aria-hidden />
          Waiting for your answer — say “yes” or “no”.
        </div>
      )}
      <div ref={endRef} />
    </div>
  );
}

function CallerBubble({ m }: { m: CallerMessage }) {
  return (
    <div className="animate-rise ml-auto max-w-[85%] sm:max-w-[70%]">
      <div className="rounded-2xl rounded-br-sm border border-line bg-raised px-4 py-3">
        <p className="text-[15px] leading-relaxed">{m.textEn}</p>
      </div>
      <p className="mt-1 text-right text-[11px] text-muted">
        You · heard as English
        {!isEnglish(m.language) && <> · spoke {languageName(m.language)}</>}
      </p>
    </div>
  );
}

function AgentBubble({ m, playing }: { m: AgentMessage; playing: "native" | "english" | null }) {
  const bilingual = !isEnglish(m.language) && m.text !== m.textEn;
  return (
    <div className="animate-rise mr-auto max-w-[92%] sm:max-w-[78%]">
      <div className="rounded-2xl rounded-bl-sm border border-line bg-surface px-4 py-3">
        {bilingual ? (
          <>
            <Line
              label={languageName(m.language)}
              text={m.text}
              speaking={playing === "native"}
              primary
            />
            <div className="my-2 border-t border-line/70" />
            <Line label="English" text={m.textEn} speaking={playing === "english"} />
          </>
        ) : (
          <Line label={null} text={m.textEn} speaking={playing !== null} primary />
        )}
      </div>
      {(m.escalated || m.intent) && (
        <div className="mt-1 flex flex-wrap items-center gap-2 text-[11px] text-muted">
          {m.intent && <span className="rounded bg-raised px-1.5 py-0.5">{m.intent.replace("_", " ")}</span>}
          {m.escalated && (
            <span className="flex items-center gap-1 text-warn">
              <Headset size={12} aria-hidden /> handing over to a human colleague
            </span>
          )}
        </div>
      )}
    </div>
  );
}

function Line({
  label,
  text,
  speaking,
  primary,
}: {
  label: string | null;
  text: string;
  speaking: boolean;
  primary?: boolean;
}) {
  return (
    <div className="flex items-start gap-3">
      <div className="min-w-0 flex-1">
        {label && (
          <span className="mb-0.5 block text-[10px] font-semibold uppercase tracking-wider text-accent">
            {label}
          </span>
        )}
        <p className={`leading-relaxed ${primary ? "text-[16px]" : "text-[14px] text-muted"}`}>{text}</p>
      </div>
      {speaking && (
        <Volume2 size={16} className="mt-1 shrink-0 text-accent" aria-label="Speaking now" />
      )}
    </div>
  );
}

export function ErrorNote({ message }: { message: string }) {
  return (
    <div role="alert" className="mx-auto flex max-w-md items-start gap-2 rounded-xl border border-danger/40 bg-danger/10 px-4 py-2 text-sm text-danger">
      <TriangleAlert size={16} className="mt-0.5 shrink-0" aria-hidden />
      {message}
    </div>
  );
}
