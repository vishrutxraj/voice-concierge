/**
 * ReplyPlayer with a fake Web Audio implementation: decoding "succeeds" with a
 * buffer that just carries the first byte of the input (so a test can tell
 * tracks apart), and sources only finish when the test says so.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ReplyPlayer } from "./player";

interface FakeSource {
  buffer: { id: number };
  onended: (() => void) | null;
  started: boolean;
  stopped: boolean;
  connect: () => void;
  start: () => void;
  stop: () => void;
}

let sources: FakeSource[] = [];
let played: number[] = [];

class FakeAudioContext {
  state = "running";
  destination = {};
  decodeAudioData(bytes: ArrayBuffer) {
    return Promise.resolve({ id: new Uint8Array(bytes)[0] });
  }
  createBufferSource(): FakeSource {
    const s: FakeSource = {
      buffer: { id: -1 },
      onended: null,
      started: false,
      stopped: false,
      connect() {},
      start() {
        s.started = true;
        played.push(s.buffer.id);
      },
      stop() {
        s.stopped = true;
      },
    };
    sources.push(s);
    return s;
  }
  resume() {}
  close() {
    return Promise.resolve();
  }
}

const bytes = (id: number) => new Uint8Array([id, 0, 0, 0]).buffer;
const settle = () => new Promise((r) => setTimeout(r, 0));
/** Finish whichever source is currently playing, like the audio hardware would. */
const finishCurrent = async () => {
  const current = [...sources].reverse().find((s) => s.started && !s.stopped && s.onended);
  current?.onended?.();
  await settle();
};

beforeEach(() => {
  sources = [];
  played = [];
  vi.stubGlobal("AudioContext", FakeAudioContext);
});
afterEach(() => vi.unstubAllGlobals());

function newPlayer() {
  const events = { starts: [] as string[], idles: 0 };
  const player = new ReplyPlayer({
    onTrackStart: (v) => events.starts.push(v),
    onIdle: () => (events.idles += 1),
  });
  player.unlock();
  return { player, events };
}

/** Stream a two-track ("both") reply: native (id 1) then English (id 2). */
async function streamBothTracks(player: ReplyPlayer) {
  player.beginTurn(["native", "english"]);
  player.push(bytes(1));
  player.markTrack("english");
  player.push(bytes(2));
  player.endTurn();
  await settle();
}

describe("ReplyPlayer playback", () => {
  it("plays the tracks of a reply in stream order", async () => {
    const { player, events } = newPlayer();
    await streamBothTracks(player);
    expect(played).toEqual([1]); // second track waits for the first to finish
    await finishCurrent();
    expect(played).toEqual([1, 2]);
    await finishCurrent();
    expect(events.starts).toEqual(["native", "english"]);
    expect(events.idles).toBeGreaterThan(0);
  });
});

describe("ReplyPlayer repeat", () => {
  it("cannot repeat before any audio has been decoded", () => {
    const { player } = newPlayer();
    expect(player.canReplay).toBe(false);
    expect(player.replay()).toBe(false);
    expect(played).toEqual([]);
  });

  it("replays the whole reply, in order, after it finished", async () => {
    const { player } = newPlayer();
    await streamBothTracks(player);
    await finishCurrent();
    await finishCurrent();
    expect(player.canReplay).toBe(true);

    expect(player.replay()).toBe(true);
    await settle();
    expect(played).toEqual([1, 2, 1]);
    await finishCurrent();
    expect(played).toEqual([1, 2, 1, 2]);
  });

  it("can repeat more than once", async () => {
    const { player } = newPlayer();
    await streamBothTracks(player);
    await finishCurrent();
    await finishCurrent();
    for (let i = 0; i < 2; i++) {
      player.replay();
      await finishCurrent();
      await finishCurrent();
    }
    expect(played).toEqual([1, 2, 1, 2, 1, 2]);
  });

  it("restarts from the top if repeated while still playing, without overlap", async () => {
    const { player } = newPlayer();
    await streamBothTracks(player);
    expect(played).toEqual([1]);

    player.replay(); // mid-track
    await settle();
    expect(sources[0].stopped).toBe(true); // the first playback was cut, not layered
    expect(played).toEqual([1, 1]);
  });

  it("is still repeatable after the caller interrupted the reply (barge-in)", async () => {
    const { player } = newPlayer();
    await streamBothTracks(player);
    expect(played).toEqual([1]); // native is playing, english is queued behind it
    player.stop(); // the caller taps the mic mid-reply
    expect(player.canReplay).toBe(true);

    expect(player.replay()).toBe(true);
    await settle();
    expect(played).toEqual([1, 1]); // the full reply again, from its first track
    await finishCurrent();
    expect(played).toEqual([1, 1, 2]); // ...including the track that was cut off
  });

  it("is replaced by the next reply, never a mix of the two", async () => {
    const { player } = newPlayer();
    await streamBothTracks(player);
    await finishCurrent();
    await finishCurrent();

    player.beginTurn(["native"]); // the next reply starts
    expect(player.canReplay).toBe(false);
    player.push(bytes(7));
    player.endTurn();
    await settle();
    await finishCurrent();

    played = [];
    player.replay();
    await settle();
    expect(played).toEqual([7]);
  });

  it("forgets the repeatable reply when a new call starts", async () => {
    const { player } = newPlayer();
    await streamBothTracks(player);
    await finishCurrent();
    await finishCurrent();

    player.reset();
    expect(player.canReplay).toBe(false);
    expect(player.replay()).toBe(false);
  });

  it("reports idle again after a repeat finishes", async () => {
    const { player, events } = newPlayer();
    await streamBothTracks(player);
    await finishCurrent();
    await finishCurrent();
    const idlesBefore = events.idles;

    player.replay();
    await finishCurrent();
    await finishCurrent();
    expect(events.idles).toBeGreaterThan(idlesBefore);
  });
});
