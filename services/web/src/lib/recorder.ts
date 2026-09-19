/**
 * Microphone capture with end-of-speech detection.
 *
 * The gateway has no server-side VAD: one binary frame == one whole
 * utterance. So the client decides when an utterance is over. Strategy:
 * tap to start listening, then auto-finish after a stretch of silence that
 * follows detected speech (or finish immediately when the user taps again).
 *
 * The mic stream stays open for the whole call (one permission prompt, no
 * per-turn startup latency) but samples are only kept while a capture is
 * active.
 */

import { TARGET_SAMPLE_RATE, concatFloat32, encodeWav, resample, rms } from "./wav";

// Runs on the audio thread; batches 128-sample render quanta into ~2k-sample
// messages so the main thread isn't flooded.
const WORKLET_SOURCE = `
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() { super(); this.buf = []; this.len = 0; }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) {
      this.buf.push(new Float32Array(ch));
      this.len += ch.length;
      if (this.len >= 2048) {
        const out = new Float32Array(this.len);
        let o = 0;
        for (const b of this.buf) { out.set(b, o); o += b.length; }
        this.port.postMessage(out, [out.buffer]);
        this.buf = []; this.len = 0;
      }
    }
    return true;
  }
}
registerProcessor("capture-processor", CaptureProcessor);
`;

export interface CaptureOptions {
  /** Called ~20x/second with the current mic level (0..1-ish RMS). */
  onLevel?: (level: number) => void;
  /** Silence after speech that ends the utterance. */
  silenceMs?: number;
  /** Give up if nobody speaks for this long. */
  noSpeechTimeoutMs?: number;
  /** Hard cap on one utterance. */
  maxMs?: number;
}

export interface Capture {
  /** Resolves with a 16 kHz mono WAV, or null if cancelled / nothing was said. */
  result: Promise<ArrayBuffer | null>;
  /** Stop now and send what was heard so far. */
  finish(): void;
  /** Stop now and discard. */
  cancel(): void;
}

const MIN_SPEECH_MS = 300;
const CALIBRATION_MS = 300;

export class MicRecorder {
  private ctx: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private node: AudioWorkletNode | null = null;
  private sink: ((samples: Float32Array) => void) | null = null;

  get isOpen(): boolean {
    return this.ctx !== null;
  }

  async open(): Promise<void> {
    if (this.ctx) return;
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("This browser cannot access the microphone (is the page served over HTTPS?).");
    }
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    const ctx = new AudioContext();
    const blobUrl = URL.createObjectURL(new Blob([WORKLET_SOURCE], { type: "application/javascript" }));
    try {
      await ctx.audioWorklet.addModule(blobUrl);
    } finally {
      URL.revokeObjectURL(blobUrl);
    }
    const source = ctx.createMediaStreamSource(this.stream);
    const node = new AudioWorkletNode(ctx, "capture-processor");
    node.port.onmessage = (e: MessageEvent<Float32Array>) => this.sink?.(e.data);
    // Some browsers only pull the graph if it reaches the destination; route
    // through a muted gain so nothing is echoed to the speakers.
    const mute = ctx.createGain();
    mute.gain.value = 0;
    source.connect(node).connect(mute).connect(ctx.destination);
    this.ctx = ctx;
    this.node = node;
  }

  capture(opts: CaptureOptions = {}): Capture {
    const ctx = this.ctx;
    if (!ctx) throw new Error("Microphone not open");
    if (ctx.state === "suspended") void ctx.resume();

    const silenceMs = opts.silenceMs ?? 1100;
    const noSpeechTimeoutMs = opts.noSpeechTimeoutMs ?? 8000;
    const maxMs = opts.maxMs ?? 30000;
    const sampleRate = ctx.sampleRate;

    const chunks: Float32Array[] = [];
    let elapsedMs = 0;
    let calibrationSum = 0;
    let calibrationN = 0;
    let threshold = 0.02;
    let speechMs = 0;
    let silenceRunMs = 0;
    let done = false;

    let resolve!: (v: ArrayBuffer | null) => void;
    const result = new Promise<ArrayBuffer | null>((r) => (resolve = r));

    const end = (send: boolean) => {
      if (done) return;
      done = true;
      this.sink = null;
      opts.onLevel?.(0);
      if (!send || speechMs < MIN_SPEECH_MS) {
        resolve(null);
        return;
      }
      const all = concatFloat32(chunks);
      resolve(encodeWav(resample(all, sampleRate, TARGET_SAMPLE_RATE), TARGET_SAMPLE_RATE));
    };

    this.sink = (samples) => {
      if (done) return;
      chunks.push(samples);
      const ms = (samples.length / sampleRate) * 1000;
      elapsedMs += ms;
      const level = rms(samples);
      opts.onLevel?.(level);

      // Learn the room's noise floor from the first moments, so a fan or a
      // noisy street doesn't read as speech.
      if (elapsedMs <= CALIBRATION_MS) {
        calibrationSum += level;
        calibrationN += 1;
        threshold = Math.max(0.02, (calibrationSum / calibrationN) * 3.5);
        return;
      }

      if (level > threshold) {
        speechMs += ms;
        silenceRunMs = 0;
      } else if (speechMs > 0) {
        silenceRunMs += ms;
        if (silenceRunMs >= silenceMs) end(true);
      } else if (elapsedMs >= noSpeechTimeoutMs) {
        end(false);
      }
      if (elapsedMs >= maxMs) end(true);
    };

    return { result, finish: () => end(true), cancel: () => end(false) };
  }

  close(): void {
    this.sink = null;
    this.node?.disconnect();
    this.stream?.getTracks().forEach((t) => t.stop());
    void this.ctx?.close();
    this.ctx = null;
    this.stream = null;
    this.node = null;
  }
}
