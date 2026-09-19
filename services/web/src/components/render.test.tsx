/**
 * Render smoke tests (server-side render, no DOM needed). They can't judge
 * how anything looks, but they catch the failures that would show up as a
 * blank white page: a component that throws on real message shapes, a missing
 * language, an empty explain payload.
 */
import { renderToString as ssr } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { Conversation } from "./Conversation";
import { ExplainPanel } from "./ExplainPanel";
import { HearToggle } from "./HearToggle";
import { MicDock } from "./MicDock";
import { StartScreen } from "./StartScreen";
import { DEMO_CALLERS } from "../lib/demoCallers";
import type { Message } from "../lib/protocol";

// React's SSR output splits adjacent text nodes with "<!-- -->" markers
// ("spoke <!-- -->Hindi"); strip them so assertions read like the page does.
const renderToString = (node: Parameters<typeof ssr>[0]) => ssr(node).replace(/<!-- -->/g, "");

const hindiTurn: Message[] = [
  { id: 1, role: "caller", textEn: "where is my order", language: "hi-IN" },
  {
    id: 2,
    role: "agent",
    text: "आपका ऑर्डर डिलीवरी के रास्ते में है",
    textEn: "Your order is on its way",
    language: "hi-IN",
    intent: "order_lookup",
    escalated: false,
    awaitingConfirmation: null,
  },
];

describe("components render", () => {
  it("StartScreen shows the demo callers and a start button", () => {
    const html = renderToString(
      <StartScreen
        connecting={false}
        error={null}
        audioPreference="native"
        onAudioPreference={() => {}}
        onStart={() => {}}
      />,
    );
    expect(html).toContain("Start call");
    expect(html).toContain("9990000002");
    expect(html).toContain("Hear replies in");
  });

  it("StartScreen surfaces a mic/permission error", () => {
    const html = renderToString(
      <StartScreen
        connecting={false}
        error="Microphone permission was denied."
        audioPreference="both"
        onAudioPreference={() => {}}
        onStart={() => {}}
      />,
    );
    expect(html).toContain("Microphone permission was denied.");
  });

  it("Conversation shows BOTH languages for a translated reply", () => {
    const html = renderToString(<Conversation messages={hindiTurn} playing="native" awaitingPrompt={null} />);
    expect(html).toContain("आपका ऑर्डर डिलीवरी के रास्ते में है");
    expect(html).toContain("Your order is on its way");
    expect(html).toContain("Hindi");
    expect(html).toContain("spoke Hindi");
  });

  it("Conversation shows an English reply once, without a language split", () => {
    const messages: Message[] = [
      {
        id: 1,
        role: "agent",
        text: "Hello",
        textEn: "Hello",
        language: "en-IN",
        intent: null,
        escalated: true,
        awaitingConfirmation: null,
      },
    ];
    const html = renderToString(<Conversation messages={messages} playing={null} awaitingPrompt={null} />);
    expect(html.match(/Hello/g)).toHaveLength(1);
    expect(html).toContain("human colleague");
  });

  it("Conversation shows the empty-state hint and the confirmation prompt", () => {
    expect(renderToString(<Conversation messages={[]} playing={null} awaitingPrompt={null} />)).toContain(
      "Tap the microphone",
    );
    expect(renderToString(<Conversation messages={hindiTurn} playing={null} awaitingPrompt="Confirm?" />)).toContain(
      "say “yes” or “no”",
    );
  });

  it("MicDock labels every call status", () => {
    for (const [status, text] of [
      ["ready", "Tap to speak"],
      ["listening", "tap to send"],
      ["processing", "Thinking"],
      ["speaking", "tap to interrupt"],
    ] as const) {
      const html = renderToString(
        <MicDock status={status} level={0.05} onTap={() => {}} onEnd={() => {}} canReplay={false} onReplay={() => {}} />,
      );
      expect(html).toContain(text);
    }
  });

  it("MicDock's Repeat button is only enabled when there is a reply to repeat and nobody is mid-turn", () => {
    const repeatTag = (status: "ready" | "speaking" | "listening" | "processing", canReplay: boolean) => {
      const html = renderToString(
        <MicDock status={status} level={0} onTap={() => {}} onEnd={() => {}} canReplay={canReplay} onReplay={() => {}} />,
      );
      const tag = html.match(/<button[^>]*aria-label="Repeat the last reply"[^>]*>/);
      expect(tag).not.toBeNull();
      return tag![0];
    };
    // Match the attribute itself: the class list always contains "disabled:" variants.
    const isDisabled = (tag: string) => /\sdisabled=""/.test(tag);
    expect(isDisabled(repeatTag("ready", true))).toBe(false);
    expect(isDisabled(repeatTag("speaking", true))).toBe(false); // restarts the reply
    expect(isDisabled(repeatTag("ready", false))).toBe(true); // nothing to repeat yet
    expect(isDisabled(repeatTag("listening", true))).toBe(true); // don't talk over the caller
    expect(isDisabled(repeatTag("processing", true))).toBe(true); // that reply is about to change
  });

  it("StartScreen offers every demo caller, and shows how to try the selected one", () => {
    const html = renderToString(
      <StartScreen
        connecting={false}
        error={null}
        audioPreference="native"
        onAudioPreference={() => {}}
        onStart={() => {}}
      />,
    );
    for (const c of DEMO_CALLERS) {
      expect(html).toContain(c.phone);
      expect(html).toContain(c.label);
    }
    expect(html).toContain("Try: ");
    expect(html.match(/Try: /g)).toHaveLength(1); // only the selected caller's hint
  });

  it("HearToggle marks the active option", () => {
    const html = renderToString(<HearToggle value="both" onChange={() => {}} />);
    expect(html).toMatch(/aria-checked="true"[^>]*>Both/);
  });

  it("ExplainPanel handles empty and populated payloads", () => {
    expect(renderToString(<ExplainPanel data={null} />)).toContain("Decisions the agent makes");
    const html = renderToString(
      <ExplainPanel
        data={{
          session_id: "s",
          event_count: 3,
          escalated: false,
          decisions: [{ at: 1, node: "intent_router", chose: "order_lookup", confidence: 0.95, because: "asked about delivery", rejected: null }],
          tool_calls: [{ node: "order_lookup", call: "get_order", data: {} }],
          guardrail_checks: [{ node: "input_guardrail", verdict: "allowed", data: {} }],
        }}
      />,
    );
    expect(html).toContain("order_lookup");
    expect(html).toContain("95%");
    expect(html).toContain("input_guardrail");
  });
});
