import { describe, expect, it } from "vitest";
import { DEFAULT_MAX_VECTORS, InMemoryVectorStore, ValidationError, VectorStore } from "@murphychain/core";
import type { UpsertItem } from "@murphychain/core";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

function item(id: string, vector: number[], payload: Record<string, unknown> = { id }): UpsertItem<Record<string, unknown>> {
  return { id, vector, payload };
}

/** A tiny store with four vectors in 2-D. */
async function sampleStore(): Promise<InMemoryVectorStore<Record<string, unknown>>> {
  const store = new InMemoryVectorStore<Record<string, unknown>>();
  await store.upsert([
    item("east", [1, 0]),
    item("north", [0, 1]),
    item("diag", [1, 1]),
    item("west", [-1, 0]),
  ]);
  return store;
}

// ---------------------------------------------------------------------------
// VectorStore contract
// ---------------------------------------------------------------------------

describe("InMemoryVectorStore", () => {
  it("starts empty and answers queries with no matches", async () => {
    const store = new InMemoryVectorStore();
    expect(store.size).toBe(0);
    await expect(store.query([1, 0], { topK: 3 })).resolves.toEqual([]);
  });

  it("extends the abstract VectorStore contract", () => {
    expect(new InMemoryVectorStore()).toBeInstanceOf(VectorStore);
  });

  it("ranks matches by cosine similarity, best first", async () => {
    const store = await sampleStore();
    const matches = await store.query([1, 0], { topK: 4 });
    expect(matches.map((m) => m.id)).toEqual(["east", "diag", "north", "west"]);
    expect(matches[0]!.score).toBeCloseTo(1, 10);
    expect(matches[1]!.score).toBeCloseTo(Math.SQRT1_2, 10);
    expect(matches[2]!.score).toBeCloseTo(0, 10);
    expect(matches[3]!.score).toBeCloseTo(-1, 10);
    expect(matches[0]!.payload).toEqual({ id: "east" });
  });

  it("returns at most topK matches", async () => {
    const store = await sampleStore();
    const matches = await store.query([1, 0], { topK: 2 });
    expect(matches).toHaveLength(2);
    expect(matches[0]!.id).toBe("east");
    expect(matches[1]!.id).toBe("diag");
  });

  it("returns everything when topK exceeds the store size", async () => {
    const store = await sampleStore();
    await expect(store.query([1, 0], { topK: 99 })).resolves.toHaveLength(4);
  });

  it("breaks score ties deterministically by id", async () => {
    const store = new InMemoryVectorStore<Record<string, unknown>>();
    await store.upsert([
      item("b", [1, 0]),
      item("a", [1, 0]),
      item("c", [1, 0]),
    ]);
    const matches = await store.query([1, 0], { topK: 3 });
    expect(matches.map((m) => m.id)).toEqual(["a", "b", "c"]);
  });

  it("replaces an existing id on upsert", async () => {
    const store = new InMemoryVectorStore<Record<string, unknown>>();
    await store.upsert([item("x", [1, 0], { v: 1 })]);
    await store.upsert([item("x", [0, 1], { v: 2 })]);
    expect(store.size).toBe(1);
    const matches = await store.query([0, 1], { topK: 1 });
    expect(matches[0]!.payload).toEqual({ v: 2 });
  });

  it("lets the last duplicate within one batch win", async () => {
    const store = new InMemoryVectorStore<Record<string, unknown>>();
    await store.upsert([item("x", [1, 0], { v: 1 }), item("x", [0, 1], { v: 2 })]);
    expect(store.size).toBe(1);
    const matches = await store.query([0, 1], { topK: 1 });
    expect(matches[0]!.payload).toEqual({ v: 2 });
  });

  it("validates the whole batch before writing anything", async () => {
    const store = new InMemoryVectorStore<Record<string, unknown>>();
    await expect(
      store.upsert([item("good", [1, 0]), item("bad", [1, "x" as unknown as number])]),
    ).rejects.toThrow(ValidationError);
    expect(store.size).toBe(0);
  });

  it("rejects dimension mismatches against the stored dimension", async () => {
    const store = new InMemoryVectorStore<Record<string, unknown>>();
    await store.upsert([item("a", [1, 0, 0])]);
    await expect(store.upsert([item("b", [1, 0])])).rejects.toThrow(/dimension 2/);
    // Same guard applies to queries.
    await expect(store.query([1, 0], { topK: 1 })).rejects.toThrow(/dimension 2/);
    expect(store.size).toBe(1);
  });

  it("rejects mixed dimensions within a single first batch", async () => {
    const store = new InMemoryVectorStore<Record<string, unknown>>();
    await expect(
      store.upsert([item("a", [1, 0]), item("b", [1, 0, 0])]),
    ).rejects.toThrow(/dimension/);
    expect(store.size).toBe(0);
  });

  it("rejects malformed vectors", async () => {
    const store = new InMemoryVectorStore<Record<string, unknown>>();
    await expect(store.upsert([item("a", [])])).rejects.toThrow(ValidationError);
    await expect(store.upsert([item("a", [1, NaN])])).rejects.toThrow(/finite number/);
    await expect(store.upsert([item("a", [1, Infinity])])).rejects.toThrow(/finite number/);
    await expect(store.upsert([item("a", [0, 0])])).rejects.toThrow(/non-zero magnitude/);
    await expect(store.query([], { topK: 1 })).rejects.toThrow(ValidationError);
    await expect(store.query([0, 0], { topK: 1 })).rejects.toThrow(/non-zero magnitude/);
  });

  it("rejects invalid ids and items", async () => {
    const store = new InMemoryVectorStore<Record<string, unknown>>();
    await expect(store.upsert([item("", [1, 0])])).rejects.toThrow(/non-empty string/);
    await expect(store.upsert([null as unknown as UpsertItem<never>])).rejects.toThrow(
      /must be an object/,
    );
    await expect(store.upsert("nope" as unknown as UpsertItem<never>[])).rejects.toThrow(
      /array of items/,
    );
  });

  it("validates query options", async () => {
    const store = await sampleStore();
    await expect(store.query([1, 0], { topK: 0 })).rejects.toThrow(/integer >= 1/);
    await expect(store.query([1, 0], { topK: -1 })).rejects.toThrow(/integer >= 1/);
    await expect(store.query([1, 0], { topK: 1.5 })).rejects.toThrow(/integer >= 1/);
    await expect(
      store.query([1, 0], { topK: 1, filter: "yes" as unknown as () => boolean }),
    ).rejects.toThrow(/must be a function/);
    await expect(
      store.query([1, 0], null as unknown as { topK: number }),
    ).rejects.toThrow(/options object/);
  });

  it("filters payloads before ranking", async () => {
    const store = await sampleStore();
    const matches = await store.query([1, 0], {
      topK: 4,
      filter: (payload) => payload["id"] !== "east",
    });
    expect(matches.map((m) => m.id)).toEqual(["diag", "north", "west"]);
  });

  it("bounds capacity with maxVectors", async () => {
    const store = new InMemoryVectorStore<Record<string, unknown>>({ maxVectors: 2 });
    await store.upsert([item("a", [1, 0]), item("b", [0, 1])]);
    await expect(store.upsert([item("c", [1, 1])])).rejects.toThrow(/maxVectors limit of 2/);
    // Replacing an existing id does not grow the store.
    await expect(store.upsert([item("a", [1, 0], { v: 2 })])).resolves.toBeUndefined();
    expect(store.size).toBe(2);
    expect(DEFAULT_MAX_VECTORS).toBe(10_000);
  });

  it("validates constructor options", () => {
    expect(() => new InMemoryVectorStore({ maxVectors: 0 })).toThrow(ValidationError);
    expect(() => new InMemoryVectorStore({ maxVectors: 2.5 })).toThrow(ValidationError);
    expect(() => new InMemoryVectorStore({ maxVectors: -1 })).toThrow(ValidationError);
  });

  it("deletes ids and reports how many were stored", async () => {
    const store = await sampleStore();
    await expect(store.delete(["east", "missing"])).resolves.toBe(1);
    expect(store.size).toBe(3);
    await expect(store.delete([])).resolves.toBe(0);
    await expect(store.delete([""])).rejects.toThrow(/non-empty string/);
    await expect(store.delete("x" as unknown as string[])).rejects.toThrow(/array of ids/);
  });

  it("forgets the dimension once emptied (reusable with another model)", async () => {
    const store = new InMemoryVectorStore<Record<string, unknown>>();
    await store.upsert([item("a", [1, 0, 0])]);
    await store.delete(["a"]);
    // Different dimension is fine now that nothing is stored.
    await expect(store.upsert([item("b", [1, 0])])).resolves.toBeUndefined();
    expect(store.size).toBe(1);
  });

  it("stores a defensive copy of the vector", async () => {
    const store = new InMemoryVectorStore<Record<string, unknown>>();
    const vector = [1, 0];
    await store.upsert([item("a", vector)]);
    vector[0] = -1; // caller mutates after upsert
    const matches = await store.query([1, 0], { topK: 1 });
    expect(matches[0]!.score).toBeCloseTo(1, 10); // store unaffected
  });
});
