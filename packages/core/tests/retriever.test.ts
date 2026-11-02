import { describe, expect, it } from "vitest";
import {
  AI,
  DEFAULT_CHUNK_OVERLAP,
  DEFAULT_CHUNK_SIZE,
  DEFAULT_EMBED_BATCH_SIZE,
  ExecutionCancelledError,
  InMemoryVectorStore,
  Retriever,
  ValidationError,
  chunkText,
} from "@murphychain/core";
import type { Chunk, EmbedOptions } from "@murphychain/core";
import { FakeProvider } from "./helpers";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

/** Deterministic keyword embedding: "cat*" → [1,0], "dog*" → [0,1], else [0.1,0.1]. */
function keywordEmbed(text: string): number[] {
  const lower = text.toLowerCase();
  const cat = lower.includes("cat") ? 1 : 0;
  const dog = lower.includes("dog") ? 1 : 0;
  if (cat === 0 && dog === 0) return [0.1, 0.1];
  return [cat, dog];
}

function makeAI(provider: FakeProvider): AI {
  return new AI({ provider, model: "m", retries: 0 });
}

/** FakeProvider that records each embedMany batch (for batch-bound tests). */
class CountingProvider extends FakeProvider {
  readonly embedBatches: string[][] = [];

  override async embedMany(texts: string[], options?: EmbedOptions): Promise<number[][]> {
    this.embedBatches.push([...texts]);
    return super.embedMany(texts, options);
  }
}

// ---------------------------------------------------------------------------
// chunkText
// ---------------------------------------------------------------------------

describe("chunkText", () => {
  it("returns one chunk for text within the chunk size", () => {
    expect(chunkText("short text", { chunkSize: 100 })).toEqual(["short text"]);
    expect(chunkText("exact-size", { chunkSize: 10 })).toEqual(["exact-size"]);
  });

  it("windows with the requested overlap", () => {
    const chunks = chunkText("0123456789ABCDEF", { chunkSize: 10, chunkOverlap: 2 });
    expect(chunks).toEqual(["0123456789", "89ABCDEF"]);
    // Consecutive chunks share the overlap region.
    expect(chunks[1]!.startsWith(chunks[0]!.slice(-2))).toBe(true);
  });

  it("drops a final remainder fully contained in the previous chunk", () => {
    // size 5, overlap 4 → step 1; the tail never grows past the overlap.
    const chunks = chunkText("abcdef", { chunkSize: 5, chunkOverlap: 4 });
    expect(chunks).toEqual(["abcde", "bcdef"]);
  });

  it("uses 1000/200 defaults", () => {
    expect(DEFAULT_CHUNK_SIZE).toBe(1_000);
    expect(DEFAULT_CHUNK_OVERLAP).toBe(200);
    const text = "x".repeat(1_500);
    const chunks = chunkText(text);
    expect(chunks).toHaveLength(2);
    expect(chunks[0]).toHaveLength(1_000);
    expect(chunks[1]).toHaveLength(700);
    // The 200-char overlap is carried over exactly.
    expect(chunks[1]!.slice(0, 200)).toBe(chunks[0]!.slice(-200));
  });

  it("validates its inputs", () => {
    expect(() => chunkText("")).toThrow(ValidationError);
    expect(() => chunkText("x", { chunkSize: 0 })).toThrow(/integer >= 1/);
    expect(() => chunkText("x", { chunkSize: 2.5 })).toThrow(/integer >= 1/);
    expect(() => chunkText("x", { chunkSize: 10, chunkOverlap: 10 })).toThrow(
      /smaller than "chunkSize"/,
    );
    expect(() => chunkText("x", { chunkSize: 10, chunkOverlap: -1 })).toThrow(/integer >= 0/);
    expect(() => chunkText("x", { chunkSize: 10, chunkOverlap: 1.5 })).toThrow(/integer >= 0/);
    expect(() => chunkText("x", null as unknown as object)).toThrow(/options object/);
  });
});

// ---------------------------------------------------------------------------
// Retriever construction
// ---------------------------------------------------------------------------

describe("Retriever construction", () => {
  it("requires an AI instance", () => {
    expect(() => new Retriever({ ai: {} as unknown as AI })).toThrow(ValidationError);
  });

  it("requires a VectorStore instance", () => {
    const ai = makeAI(new FakeProvider());
    expect(
      () => new Retriever({ ai, store: {} as unknown as InMemoryVectorStore<Chunk> }),
    ).toThrow(/VectorStore instance/);
  });

  it("applies documented defaults", () => {
    const retriever = new Retriever({ ai: makeAI(new FakeProvider()) });
    expect(retriever.chunkSize).toBe(1_000);
    expect(retriever.chunkOverlap).toBe(200);
    expect(retriever.embedBatchSize).toBe(100);
    expect(DEFAULT_EMBED_BATCH_SIZE).toBe(100);
    expect(retriever.store).toBeInstanceOf(InMemoryVectorStore);
  });

  it("validates chunking and batching options at construction", () => {
    const ai = makeAI(new FakeProvider());
    expect(() => new Retriever({ ai, chunkSize: 0 })).toThrow(ValidationError);
    expect(() => new Retriever({ ai, chunkSize: 10, chunkOverlap: 10 })).toThrow(
      /smaller than "chunkSize"/,
    );
    expect(() => new Retriever({ ai, chunkOverlap: -1 })).toThrow(/integer >= 0/);
    expect(() => new Retriever({ ai, embedBatchSize: 0 })).toThrow(/integer >= 1/);
    expect(() => new Retriever({ ai, embedBatchSize: 1_001 })).toThrow(/<= 1000/);
    expect(() => new Retriever(null as unknown as { ai: AI })).toThrow(/configuration object/);
  });
});

// ---------------------------------------------------------------------------
// index()
// ---------------------------------------------------------------------------

describe("Retriever.index", () => {
  it("chunks, embeds, and stores documents with stable ids and metadata", async () => {
    const provider = new FakeProvider({ embedImpl: async (text) => keywordEmbed(text) });
    const retriever = new Retriever({ ai: makeAI(provider) });

    const result = await retriever.index([
      { id: "feline", text: "Cats are feline animals.", metadata: { source: "faq" } },
      { id: "canine", text: "Dogs bark loudly." },
    ]);

    expect(result).toEqual({ documents: 2, chunks: 2 });
    expect(retriever.store.size).toBe(2);
    const matches = await retriever.store.query([1, 0], { topK: 2 });
    expect(matches[0]!.id).toBe("feline#0");
    expect(matches[0]!.payload.documentId).toBe("feline");
    expect(matches[0]!.payload.index).toBe(0);
    expect(matches[0]!.payload.metadata).toEqual({ source: "faq" });
    expect(matches[0]!.payload.text).toBe("Cats are feline animals.");
  });

  it("trims chunks and skips whitespace-only pieces, keeping positions", async () => {
    const provider = new FakeProvider({ embedImpl: async (text) => keywordEmbed(text) });
    const retriever = new Retriever({
      ai: makeAI(provider),
      chunkSize: 8,
      chunkOverlap: 0,
    });

    await retriever.index([{ id: "d", text: "hello    world" }]);

    expect(retriever.store.size).toBe(2);
    const matches = await retriever.store.query([1, 0], { topK: 5 });
    const byId = new Map(matches.map((m) => [m.id, m.payload]));
    expect(byId.get("d#0")!.text).toBe("hello");
    expect(byId.get("d#1")!.text).toBe("world");
    expect(byId.get("d#1")!.index).toBe(1);
  });

  it("treats an empty document list as a no-op", async () => {
    const provider = new FakeProvider();
    const retriever = new Retriever({ ai: makeAI(provider) });
    await expect(retriever.index([])).resolves.toEqual({ documents: 0, chunks: 0 });
    expect(provider.embedCalls).toHaveLength(0);
    expect(retriever.store.size).toBe(0);
  });

  it("validates documents before any embedding happens", async () => {
    const provider = new FakeProvider();
    const retriever = new Retriever({ ai: makeAI(provider) });
    const bad: unknown[][] = [
      [null],
      [{}], // missing id
      [{ id: "", text: "x" }], // empty id
      [{ id: "a" }], // missing text
      [{ id: "a", text: "" }], // empty text
      [{ id: "a", text: "   \n " }], // whitespace-only text
      [{ id: "a", text: "x", metadata: [] }], // metadata must be an object
    ];
    for (const documents of bad) {
      await expect(retriever.index(documents as never)).rejects.toThrow(ValidationError);
    }
    // Duplicate ids within one call are ambiguous.
    await expect(
      retriever.index([
        { id: "a", text: "one" },
        { id: "a", text: "two" },
      ]),
    ).rejects.toThrow(/Duplicate document id/);
    await expect(retriever.index("nope" as never)).rejects.toThrow(/array of documents/);

    // Nothing was embedded and nothing was stored.
    expect(provider.embedCalls).toHaveLength(0);
    expect(retriever.store.size).toBe(0);
  });

  it("replaces stale chunks when a document is re-indexed with fewer chunks", async () => {
    const provider = new FakeProvider({ embedImpl: async (text) => keywordEmbed(text) });
    const retriever = new Retriever({
      ai: makeAI(provider),
      chunkSize: 10,
      chunkOverlap: 0,
    });

    await retriever.index([{ id: "doc", text: "x".repeat(25) }]); // 3 chunks
    expect(retriever.store.size).toBe(3);

    await retriever.index([{ id: "doc", text: "short" }]); // 1 chunk
    expect(retriever.store.size).toBe(1);
    const matches = await retriever.store.query([0.1, 0.1], { topK: 10 });
    expect(matches.map((m) => m.id)).toEqual(["doc#0"]);
  });

  it("leaves existing data untouched when re-indexing fails to embed", async () => {
    let fail = false;
    const provider = new FakeProvider({
      embedImpl: async (text) => {
        if (fail) throw new Error("embed down");
        return keywordEmbed(text);
      },
    });
    const retriever = new Retriever({ ai: makeAI(provider) });

    await retriever.index([{ id: "doc", text: "cats are feline" }]);
    expect(retriever.store.size).toBe(1);

    fail = true;
    await expect(retriever.index([{ id: "doc", text: "dogs bark" }])).rejects.toThrow(
      "embed down",
    );

    // Old chunks are still there — nothing was deleted before embedding.
    expect(retriever.store.size).toBe(1);
    const matches = await retriever.store.query([1, 0], { topK: 1 });
    expect(matches[0]!.id).toBe("doc#0");
    expect(matches[0]!.payload.text).toBe("cats are feline");
  });

  it("embeds in bounded batches", async () => {
    const provider = new CountingProvider({ embedImpl: async (text) => keywordEmbed(text) });
    const retriever = new Retriever({
      ai: makeAI(provider),
      chunkSize: 4,
      chunkOverlap: 0,
      embedBatchSize: 10,
    });

    const result = await retriever.index([{ id: "doc", text: "x".repeat(100) }]);

    expect(result.chunks).toBe(25);
    expect(provider.embedBatches.map((batch) => batch.length)).toEqual([10, 10, 5]);
    expect(retriever.store.size).toBe(25);
  });

  it("forwards the caller's signal to the embedding calls", async () => {
    let seen: AbortSignal | undefined;
    const controller = new AbortController();
    const provider = new FakeProvider({
      embedImpl: async (text, options) => {
        seen = options?.signal;
        // Abort while the call is in flight: the linked signal must follow.
        controller.abort();
        return keywordEmbed(text);
      },
    });
    const retriever = new Retriever({ ai: makeAI(provider) });

    // The engine may map the in-flight abort to a cancellation; either way
    // the caller's signal reached the provider.
    await retriever.index([{ id: "doc", text: "cats" }], { signal: controller.signal }).catch(
      (error: unknown) => {
        expect(error).toBeInstanceOf(ExecutionCancelledError);
      },
    );

    expect(seen).toBeDefined();
    expect(seen!.aborted).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// retrieve()
// ---------------------------------------------------------------------------

describe("Retriever.retrieve", () => {
  it("round-trips: index then retrieve returns the closest chunks first", async () => {
    const provider = new FakeProvider({ embedImpl: async (text) => keywordEmbed(text) });
    const retriever = new Retriever({ ai: makeAI(provider) });

    await retriever.index([
      { id: "feline", text: "Cats are feline animals." },
      { id: "canine", text: "Dogs bark loudly." },
    ]);
    const matches = await retriever.retrieve("cats", { topK: 2 });

    expect(matches).toHaveLength(2);
    expect(matches[0]!.id).toBe("feline#0");
    expect(matches[0]!.score).toBeCloseTo(1, 10);
    expect(matches[1]!.id).toBe("canine#0");
    expect(matches[1]!.score).toBeCloseTo(0, 10);
    expect(matches[0]!.payload.text).toContain("Cats");
  });

  it("applies a payload filter before ranking", async () => {
    const provider = new FakeProvider({ embedImpl: async (text) => keywordEmbed(text) });
    const retriever = new Retriever({ ai: makeAI(provider) });
    await retriever.index([
      { id: "feline", text: "Cats are feline animals." },
      { id: "canine", text: "Dogs bark loudly." },
    ]);

    const matches = await retriever.retrieve("cats", {
      topK: 5,
      filter: (payload) => payload.documentId === "canine",
    });

    expect(matches.map((m) => m.id)).toEqual(["canine#0"]);
  });

  it("fails fast on bad input before spending an embedding call", async () => {
    const provider = new FakeProvider({ embedImpl: async (text) => keywordEmbed(text) });
    const retriever = new Retriever({ ai: makeAI(provider) });

    await expect(retriever.retrieve("", { topK: 1 })).rejects.toThrow(ValidationError);
    await expect(retriever.retrieve("   ", { topK: 1 })).rejects.toThrow(ValidationError);
    await expect(retriever.retrieve("q", { topK: 0 })).rejects.toThrow(/integer >= 1/);
    await expect(
      retriever.retrieve("q", { topK: 1, filter: "x" as unknown as () => boolean }),
    ).rejects.toThrow(/must be a function/);
    await expect(
      retriever.retrieve("q", null as unknown as { topK: number }),
    ).rejects.toThrow(/options object/);

    expect(provider.embedCalls).toHaveLength(0);
  });

  it("returns an empty list when the store is empty", async () => {
    const provider = new FakeProvider({ embedImpl: async (text) => keywordEmbed(text) });
    const retriever = new Retriever({ ai: makeAI(provider) });
    await expect(retriever.retrieve("cats", { topK: 3 })).resolves.toEqual([]);
    // The query was embedded even though there was nothing to search.
    expect(provider.embedCalls).toHaveLength(1);
  });

  it("forwards the caller's signal to the query embedding", async () => {
    let seen: AbortSignal | undefined;
    const controller = new AbortController();
    const provider = new FakeProvider({
      embedImpl: async (text, options) => {
        seen = options?.signal;
        // Abort while the call is in flight: the linked signal must follow.
        controller.abort();
        return keywordEmbed(text);
      },
    });
    const retriever = new Retriever({ ai: makeAI(provider) });

    await retriever.retrieve("cats", { topK: 1, signal: controller.signal }).catch(
      (error: unknown) => {
        expect(error).toBeInstanceOf(ExecutionCancelledError);
      },
    );

    expect(seen).toBeDefined();
    expect(seen!.aborted).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// remove()
// ---------------------------------------------------------------------------

describe("Retriever.remove", () => {
  it("removes a document's chunks and reports the count", async () => {
    const provider = new FakeProvider({ embedImpl: async (text) => keywordEmbed(text) });
    const retriever = new Retriever({
      ai: makeAI(provider),
      chunkSize: 10,
      chunkOverlap: 0,
    });
    await retriever.index([
      { id: "a", text: "x".repeat(25) }, // 3 chunks
      { id: "b", text: "cats" }, // 1 chunk
    ]);
    expect(retriever.store.size).toBe(4);

    await expect(retriever.remove(["a"])).resolves.toBe(3);
    expect(retriever.store.size).toBe(1);
    await expect(retriever.remove(["a", "missing"])).resolves.toBe(0);
  });

  it("validates every id before removing anything", async () => {
    const provider = new FakeProvider({ embedImpl: async (text) => keywordEmbed(text) });
    const retriever = new Retriever({ ai: makeAI(provider) });
    await retriever.index([{ id: "keep", text: "cats" }]);

    await expect(retriever.remove(["keep", ""])).rejects.toThrow(/non-empty string/);
    await expect(retriever.remove("x" as unknown as string[])).rejects.toThrow(
      /array of document ids/,
    );
    // Nothing was removed by the failed calls.
    expect(retriever.store.size).toBe(1);
  });

  it("allows re-indexing a removed document", async () => {
    const provider = new FakeProvider({ embedImpl: async (text) => keywordEmbed(text) });
    const retriever = new Retriever({ ai: makeAI(provider) });
    await retriever.index([{ id: "doc", text: "cats" }]);
    await retriever.remove(["doc"]);
    expect(retriever.store.size).toBe(0);

    const result = await retriever.index([{ id: "doc", text: "dogs" }]);
    expect(result).toEqual({ documents: 1, chunks: 1 });
    expect(retriever.store.size).toBe(1);
  });
});
