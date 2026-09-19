import { ShieldCheck, Wrench, Brain } from "lucide-react";
import type { ExplainPayload } from "../hooks/useCall";

/**
 * "Why did the agent do that" -- the backend's explainability endpoint,
 * rendered for people rather than as JSON. This is a Responsible-AI
 * requirement of the project, so it is a first-class panel, not a debug dump.
 */
export function ExplainPanel({ data }: { data: ExplainPayload | null }) {
  if (!data || data.event_count === 0) {
    return (
      <p className="text-sm text-muted">
        Decisions the agent makes will appear here as you talk — what it chose, how sure it was, and why.
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-5 text-sm">
      <Section icon={<Brain size={14} />} title="Decisions">
        {data.decisions.length === 0 && <Empty />}
        {data.decisions.map((d, i) => (
          <li key={i} className="rounded-lg border border-line bg-surface p-3">
            <div className="flex items-baseline justify-between gap-2">
              <span className="font-medium">{d.chose}</span>
              {d.confidence !== null && (
                <span className="shrink-0 text-xs text-accent">{Math.round(d.confidence * 100)}%</span>
              )}
            </div>
            <div className="text-[11px] text-muted">{d.node}</div>
            {d.because && <p className="mt-1 text-[13px] text-muted">{d.because}</p>}
          </li>
        ))}
      </Section>

      <Section icon={<ShieldCheck size={14} />} title="Guardrails">
        {data.guardrail_checks.length === 0 && <Empty />}
        {data.guardrail_checks.map((g, i) => (
          <li key={i} className="flex items-center justify-between rounded-lg border border-line bg-surface px-3 py-2">
            <span>{g.node}</span>
            <span className="text-xs text-muted">{g.verdict}</span>
          </li>
        ))}
      </Section>

      <Section icon={<Wrench size={14} />} title="Tools used">
        {data.tool_calls.length === 0 && <Empty />}
        {data.tool_calls.map((t, i) => (
          <li key={i} className="rounded-lg border border-line bg-surface px-3 py-2">
            <span className="font-medium">{t.call}</span>
            <span className="ml-2 text-[11px] text-muted">{t.node}</span>
          </li>
        ))}
      </Section>
    </div>
  );
}

function Section({ icon, title, children }: { icon: React.ReactNode; title: string; children: React.ReactNode }) {
  return (
    <section>
      <h3 className="mb-2 flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-muted">
        {icon}
        {title}
      </h3>
      <ul className="flex flex-col gap-2">{children}</ul>
    </section>
  );
}

function Empty() {
  return <li className="text-[13px] text-muted">None yet.</li>;
}
