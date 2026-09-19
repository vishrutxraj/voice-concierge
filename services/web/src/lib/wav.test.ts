import { describe, expect, it } from "vitest";
import { concatFloat32, encodeWav, resample, rms } from "./wav";

function ascii(view: DataView, offset: number, len: number): string {
  let s = "";
  for (let i = 0; i < len; i++) s += String.fromCharCode(view.getUint8(offset + i));
  return s;
}

describe("encodeWav", () => {
  it("writes a valid 16 kHz mono PCM16 header", () => {
    const view = new DataView(encodeWav(new Float32Array(160), 16000));
    expect(ascii(view, 0, 4)).toBe("RIFF");
    expect(ascii(view, 8, 4)).toBe("WAVE");
    expect(ascii(view, 12, 4)).toBe("fmt ");
    expect(view.getUint16(20, true)).toBe(1); // PCM
    expect(view.getUint16(22, true)).toBe(1); // mono
    expect(view.getUint32(24, true)).toBe(16000);
    expect(view.getUint32(28, true)).toBe(32000); // byte rate
    expect(view.getUint16(34, true)).toBe(16);
    expect(ascii(view, 36, 4)).toBe("data");
    expect(view.getUint32(40, true)).toBe(320); // 160 samples * 2 bytes
    expect(view.getUint32(4, true)).toBe(36 + 320); // RIFF size
    expect(view.byteLength).toBe(44 + 320);
  });

  it("maps full-scale floats to the int16 extremes and clamps overshoot", () => {
    const view = new DataView(encodeWav(new Float32Array([1, -1, 0, 2, -2])));
    expect(view.getInt16(44, true)).toBe(32767);
    expect(view.getInt16(46, true)).toBe(-32768);
    expect(view.getInt16(48, true)).toBe(0);
    expect(view.getInt16(50, true)).toBe(32767); // clamped, not wrapped
    expect(view.getInt16(52, true)).toBe(-32768);
  });
});

describe("resample", () => {
  it("returns the input untouched at the same rate", () => {
    const a = new Float32Array([0.1, 0.2]);
    expect(resample(a, 16000, 16000)).toBe(a);
  });

  it("downsamples 48 kHz to 16 kHz by a factor of three", () => {
    const out = resample(new Float32Array(4800), 48000, 16000);
    expect(out.length).toBe(1600);
  });

  it("preserves a constant signal", () => {
    const out = resample(new Float32Array(480).fill(0.5), 48000, 16000);
    expect(Array.from(out).every((v) => Math.abs(v - 0.5) < 1e-6)).toBe(true);
  });
});

describe("helpers", () => {
  it("concatenates chunks in order", () => {
    const out = concatFloat32([new Float32Array([1, 2]), new Float32Array([3])]);
    expect(Array.from(out)).toEqual([1, 2, 3]);
  });

  it("computes RMS", () => {
    expect(rms(new Float32Array([]))).toBe(0);
    expect(rms(new Float32Array([0.5, -0.5, 0.5, -0.5]))).toBeCloseTo(0.5);
  });
});
