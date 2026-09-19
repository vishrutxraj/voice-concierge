import { describe, expect, it } from "vitest";
import { DEMO_CALLERS } from "./demoCallers";

describe("DEMO_CALLERS", () => {
  it("has unique phone numbers, each a seeded 999xxxxxxx number", () => {
    const phones = DEMO_CALLERS.map((c) => c.phone);
    expect(new Set(phones).size).toBe(phones.length);
    for (const p of phones) expect(p).toMatch(/^99900000\d{2}$/);
  });

  it("gives every caller a label, a note and something to try", () => {
    for (const c of DEMO_CALLERS) {
      expect(c.label.length).toBeGreaterThan(2);
      expect(c.note.length).toBeGreaterThan(2);
      expect(c.try.length).toBeGreaterThan(2);
    }
  });

  it("offers a good range of scenarios (more than the original three)", () => {
    expect(DEMO_CALLERS.length).toBeGreaterThanOrEqual(10);
  });
});
