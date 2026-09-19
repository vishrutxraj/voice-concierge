/**
 * Reply audio playback.
 *
 * The gateway streams a reply as raw binary frames that are slices of ONE
 * WAV file (only the first slice carries the header), so nothing can be
 * decoded until a whole track has arrived. We therefore buffer per track,
 * decode when the track ends (next `audio_track` marker or `audio_end`), and
 * play tracks strictly in order -- native then English in "both" mode.
 * That trades a short wait for correctness; true streaming playback would need
 * the server to send self-contained chunks.
 *
 * Barge-in: stop() drops everything buffered, decoding, queued and playing.
 * A generation counter makes in-flight async decodes from a stopped turn
 * discard themselves instead of playing late.
 *
 * Repeat: the decoded tracks of the latest reply are kept (lastTurn) until the
 * next reply begins, so "repeat" is instant and costs no network or TTS call.
 * stop() deliberately leaves them alone -- interrupting a reply must not make
 * it un-repeatable.
 */

import type { AudioVariant } from "./protocol";

export interface PlayerCallbacks {
  /** A track began playing. */
  onTrackStart?: (variant: AudioVariant) => void;
  /** Everything for the turn has finished (or was stopped). */
  onIdle?: () => void;
}

interface Decoded {
  variant: AudioVariant;
  buffer: AudioBuffer;
}

export class ReplyPlayer {
  private ctx: AudioContext | null = null;
  private generation = 0;
  private variant: AudioVariant = "native";
  private chunks: ArrayBuffer[] = [];
  private queue: Decoded[] = [];
  private source: AudioBufferSourceNode | null = null;
  private pendingDecodes = 0;
  private streamEnded = true;
  private lastTurn: Decoded[] = [];

  constructor(private readonly cb: PlayerCallbacks = {}) {}

  /** Must be called from a user gesture (autoplay policy). Safe to repeat. */
  unlock(): void {
    if (!this.ctx) this.ctx = new AudioContext();
    if (this.ctx.state === "suspended") void this.ctx.resume();
  }

  /** A reply is about to stream: `variants` are the tracks announced by the server. */
  beginTurn(variants: AudioVariant[]): void {
    this.stop();
    this.lastTurn = []; // a new reply supersedes the one that could be repeated
    this.streamEnded = variants.length === 0;
    this.variant = variants[0] ?? "native";
    if (variants.length === 0) this.cb.onIdle?.();
  }

  /** `audio_track` marker: the previous track is complete, a new one starts. */
  markTrack(next: AudioVariant): void {
    this.flushTrack();
    this.variant = next;
  }

  push(chunk: ArrayBuffer): void {
    this.chunks.push(chunk);
  }

  /** `audio_end`: the last track is complete. */
  endTurn(): void {
    this.flushTrack();
    this.streamEnded = true;
    this.maybeIdle();
  }

  /** True once at least one track of the latest reply has been decoded. */
  get canReplay(): boolean {
    return this.lastTurn.length > 0;
  }

  /** Play the latest reply's audio again, from the start. Returns false if there is none. */
  replay(): boolean {
    if (!this.ctx || this.lastTurn.length === 0) return false;
    const tracks = this.lastTurn; // stop() below leaves lastTurn intact
    this.stop();
    this.queue = [...tracks];
    this.streamEnded = true;
    this.playNext();
    return true;
  }

  /** Barge-in / hang-up: silence everything immediately. */
  stop(): void {
    this.generation += 1;
    this.chunks = [];
    this.queue = [];
    this.pendingDecodes = 0;
    if (this.source) {
      this.source.onended = null;
      try {
        this.source.stop();
      } catch {
        /* already stopped */
      }
      this.source = null;
    }
    this.streamEnded = true;
  }

  /** Forget the repeatable reply too (start of a new call). */
  reset(): void {
    this.stop();
    this.lastTurn = [];
  }

  dispose(): void {
    this.stop();
    void this.ctx?.close();
    this.ctx = null;
  }

  private flushTrack(): void {
    if (this.chunks.length === 0 || !this.ctx) {
      this.chunks = [];
      return;
    }
    const bytes = joinBuffers(this.chunks);
    this.chunks = [];
    const gen = this.generation;
    const variant = this.variant;
    this.pendingDecodes += 1;
    // decodeAudioData is async and may complete out of order across tracks if
    // called concurrently; chain so playback order always matches stream order.
    this.decodeChain = this.decodeChain.then(async () => {
      try {
        const buffer = await this.ctx!.decodeAudioData(bytes);
        if (gen !== this.generation) return; // superseded while decoding
        this.queue.push({ variant, buffer });
        this.lastTurn.push({ variant, buffer });
        this.playNext();
      } catch (err) {
        console.warn("Could not decode reply audio", err);
      } finally {
        if (gen === this.generation) {
          this.pendingDecodes -= 1;
          this.maybeIdle();
        }
      }
    });
  }

  private decodeChain: Promise<void> = Promise.resolve();

  private playNext(): void {
    if (this.source || !this.ctx) return;
    const next = this.queue.shift();
    if (!next) return;
    const src = this.ctx.createBufferSource();
    src.buffer = next.buffer;
    src.connect(this.ctx.destination);
    this.source = src;
    src.onended = () => {
      if (this.source !== src) return;
      this.source = null;
      this.playNext();
      this.maybeIdle();
    };
    this.cb.onTrackStart?.(next.variant);
    src.start();
  }

  private maybeIdle(): void {
    if (this.streamEnded && this.pendingDecodes === 0 && !this.source && this.queue.length === 0) {
      this.cb.onIdle?.();
    }
  }
}

function joinBuffers(parts: ArrayBuffer[]): ArrayBuffer {
  const total = parts.reduce((n, p) => n + p.byteLength, 0);
  const out = new Uint8Array(total);
  let offset = 0;
  for (const p of parts) {
    out.set(new Uint8Array(p), offset);
    offset += p.byteLength;
  }
  return out.buffer;
}
