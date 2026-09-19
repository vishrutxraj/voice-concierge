import { useState } from "react";
import { AudioLines, Languages, ShieldCheck } from "lucide-react";
import { DEMO_CALLERS } from "../lib/demoCallers";
import { HearToggle } from "./HearToggle";
import type { AudioPreference } from "../lib/protocol";

interface Props {
  connecting: boolean;
  error: string | null;
  audioPreference: AudioPreference;
  onAudioPreference: (p: AudioPreference) => void;
  onStart: (phone: string) => void;
}

export function StartScreen({ connecting, error, audioPreference, onAudioPreference, onStart }: Props) {
  const [phone, setPhone] = useState(DEMO_CALLERS[0].phone);

  return (
    <div className="mx-auto flex min-h-full max-w-xl flex-col justify-center gap-8 px-5 py-10">
      <header className="text-center">
        <div className="mx-auto mb-4 grid size-14 place-items-center rounded-2xl border border-accent/40 bg-accent/10 text-accent">
          <AudioLines size={28} aria-hidden />
        </div>
        <h1 className="text-3xl font-semibold tracking-tight">Delivery Voice Concierge</h1>
        <p className="mt-2 text-muted">
          Ask about an order, reschedule a delivery, or fix an address — by voice, in your own language.
        </p>
      </header>

      <section className="rounded-2xl border border-line bg-surface p-5">
        <h2 className="mb-3 text-sm font-medium">Call as</h2>
        <div
          className="flex max-h-[22rem] flex-col gap-2 overflow-y-auto pr-1"
          role="radiogroup"
          aria-label="Demo caller"
        >
          {DEMO_CALLERS.map((c) => (
            <label
              key={c.phone}
              className={`flex cursor-pointer items-center gap-3 rounded-xl border px-4 py-3 transition-colors ${
                phone === c.phone ? "border-accent bg-accent/10" : "border-line hover:border-muted/50"
              }`}
            >
              <input
                type="radio"
                name="caller"
                className="accent-[var(--color-accent)]"
                checked={phone === c.phone}
                onChange={() => setPhone(c.phone)}
              />
              <span className="min-w-0 flex-1">
                <span className="block text-sm font-medium">{c.label}</span>
                <span className="block text-xs text-muted">{c.note}</span>
                {phone === c.phone && (
                  <span className="mt-1 block text-xs text-accent">Try: {c.try}</span>
                )}
              </span>
              <span className="shrink-0 font-mono text-xs text-muted">{c.phone}</span>
            </label>
          ))}
        </div>

        <div className="mt-5">
          <HearToggle value={audioPreference} onChange={onAudioPreference} />
        </div>

        <button
          type="button"
          onClick={() => onStart(phone)}
          disabled={connecting}
          className="mt-6 w-full rounded-xl bg-accent py-3 font-semibold text-bg transition-colors hover:bg-accent-strong focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-accent disabled:opacity-60"
        >
          {connecting ? "Connecting…" : "Start call"}
        </button>
        {error && (
          <p role="alert" className="mt-3 text-sm text-danger">
            {error}
          </p>
        )}
      </section>

      <ul className="grid gap-3 text-xs text-muted sm:grid-cols-2">
        <li className="flex items-start gap-2">
          <Languages size={16} className="mt-0.5 shrink-0 text-accent" aria-hidden />
          Understands 10+ Indian languages and English. Replies appear in both your language and English.
        </li>
        <li className="flex items-start gap-2">
          <ShieldCheck size={16} className="mt-0.5 shrink-0 text-accent" aria-hidden />
          Changes to your delivery are always read back and confirmed before anything is saved.
        </li>
      </ul>
    </div>
  );
}
