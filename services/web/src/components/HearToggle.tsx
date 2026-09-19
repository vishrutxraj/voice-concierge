import type { AudioPreference } from "../lib/protocol";

const OPTIONS: { value: AudioPreference; label: string; hint: string }[] = [
  { value: "native", label: "My language", hint: "Hear replies in the language you spoke" },
  { value: "english", label: "English", hint: "Hear replies in English" },
  { value: "both", label: "Both", hint: "Your language first, then English" },
];

interface Props {
  value: AudioPreference;
  onChange: (value: AudioPreference) => void;
}

/** Text always shows both languages; this only picks what is SPOKEN. */
export function HearToggle({ value, onChange }: Props) {
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[11px] font-medium uppercase tracking-wider text-muted">Hear replies in</span>
      <div role="radiogroup" aria-label="Reply audio language" className="inline-flex rounded-full border border-line bg-surface p-1">
        {OPTIONS.map((o) => {
          const active = o.value === value;
          return (
            <button
              key={o.value}
              type="button"
              role="radio"
              aria-checked={active}
              title={o.hint}
              onClick={() => onChange(o.value)}
              className={`rounded-full px-3 py-1 text-sm transition-colors focus-visible:outline-2 focus-visible:outline-accent ${
                active ? "bg-accent text-bg font-semibold" : "text-muted hover:text-ink"
              }`}
            >
              {o.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
