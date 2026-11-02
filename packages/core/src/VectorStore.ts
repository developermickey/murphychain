import { ValidationError } from "./Errors";
import { assertPositiveInteger } from "./validation";

/**
 * Vector stores: the persistence layer for embedding-based retrieval.
 *
 * A vector store maps opaque ids to fixed-length numeric vectors plus an
 * application payload, and answers nearest-neighbour queries by cosine
 * similarity. MurphyChain ships one implementation —
 * {@link InMemoryVectorStore} — behind an abstract {@link VectorStore} class
 * so application code can swap in a remote store (pgvector, a service, …)
 * without touching retrieval logic.
 *
 * The store validates every vector at the boundary: non-empty, finite
 * components, non-zero magnitude (cosine is undefined for the zero vector),
 * and one consistent dimension across all stored vectors. Malformed input
 * fails before anything is written.
 */
export interface VectorMatch<T> {
  id: string;
  /** Cosine similarity in [-1, 1]; higher means closer. */
  score: number;
  /** The payload stored with the vector (returned by reference — treat as read-only). */
  payload: T;
}

/** One item to write to a store. */
export interface UpsertItem<T> {
  id: string;
  vector: number[];
  payload: T;
}

/** Options for {@link VectorStore.query}. */
export interface QueryOptions<T> {
  /** Maximum number of matches to return (the store may return fewer). */
  topK: number;
  /**
   * Optional predicate applied before ranking. In-memory stores evaluate it
   * directly; remote adapters must reject filters they cannot push down
   * rather than scanning unbounded data.
   */
  filter?: ((payload: T) => boolean) | undefined;
}

/**
 * Abstract vector store contract. Methods are async so remote stores fit
 * the same interface as the in-memory implementation.
 */
export abstract class VectorStore<T = Record<string, unknown>> {
  /** Number of vectors currently stored. */
  abstract get size(): number;
  /** Inserts or replaces items by id. Validates everything before writing. */
  abstract upsert(items: readonly UpsertItem<T>[]): Promise<void>;
  /** Returns the closest matches by cosine similarity, best first. */
  abstract query(vector: readonly number[], options: QueryOptions<T>): Promise<VectorMatch<T>[]>;
  /** Removes the given ids; returns how many were actually stored. */
  abstract delete(ids: readonly string[]): Promise<number>;
}

/** Default hard cap on vectors held by {@link InMemoryVectorStore}. */
export const DEFAULT_MAX_VECTORS = 10_000;

export interface InMemoryVectorStoreOptions {
  /**
   * Hard cap on stored vectors — bounds memory even if an application feeds
   * an unbounded document stream. Default: 10 000.
   */
  maxVectors?: number | undefined;
}

interface StoredVector<T> {
  vector: number[];
  norm: number;
  payload: T;
}

/**
 * Validates and copies a vector: non-empty, finite components, non-zero
 * magnitude. Returns the copy plus its Euclidean norm (cached per upsert so
 * queries never recompute stored norms).
 */
function assertVector(value: unknown, field: string): { vector: number[]; norm: number } {
  if (!Array.isArray(value) || value.length === 0) {
    throw new ValidationError(`"${field}" must be a non-empty array of finite numbers.`, { field });
  }
  const vector: number[] = new Array<number>(value.length);
  let sum = 0;
  for (let i = 0; i < value.length; i++) {
    const component: unknown = value[i];
    if (typeof component !== "number" || !Number.isFinite(component)) {
      throw new ValidationError(`"${field}"[${i}] must be a finite number.`, { field });
    }
    vector[i] = component;
    sum += component * component;
  }
  const norm = Math.sqrt(sum);
  if (norm === 0) {
    throw new ValidationError(
      `"${field}" must have a non-zero magnitude (cosine similarity is undefined for the zero vector).`,
      { field },
    );
  }
  return { vector, norm };
}

/**
 * Bounded in-memory vector store: linear scan with cosine similarity.
 *
 * Suitable for prototypes, tests, and small corpora. For large or shared
 * datasets implement {@link VectorStore} over a dedicated index — the scan
 * here is O(n) per query and makes no performance claims.
 */
export class InMemoryVectorStore<T = Record<string, unknown>> extends VectorStore<T> {
  private readonly items = new Map<string, StoredVector<T>>();
  private readonly maxVectors: number;
  private dimension: number | undefined;

  constructor(options: InMemoryVectorStoreOptions = {}) {
    super();
    const maxVectors = options.maxVectors ?? DEFAULT_MAX_VECTORS;
    if (!Number.isInteger(maxVectors) || maxVectors < 1) {
      throw new ValidationError('"maxVectors" must be an integer >= 1.', { field: "maxVectors" });
    }
    this.maxVectors = maxVectors;
  }

  get size(): number {
    return this.items.size;
  }

  async upsert(items: readonly UpsertItem<T>[]): Promise<void> {
    if (!Array.isArray(items)) {
      throw new ValidationError("upsert() expects an array of items.", { field: "items" });
    }

    // Validate the whole batch first: an invalid item must not leave a
    // partially written store behind.
    const prepared: Array<{ id: string; stored: StoredVector<T> }> = [];
    const positions = new Map<string, number>();
    let newIds = 0;
    let dimension = this.dimension;
    for (let i = 0; i < items.length; i++) {
      const item: unknown = items[i];
      if (typeof item !== "object" || item === null) {
        throw new ValidationError(`items[${i}] must be an object.`, { field: "items" });
      }
      const { id, vector, payload } = item as UpsertItem<T>;
      if (typeof id !== "string" || id.length === 0) {
        throw new ValidationError(`items[${i}].id must be a non-empty string.`, { field: "items" });
      }
      const { vector: copied, norm } = assertVector(vector, `items[${i}].vector`);
      if (dimension !== undefined && copied.length !== dimension) {
        throw new ValidationError(
          `items[${i}].vector has dimension ${copied.length}, but this store holds dimension ${dimension} vectors.`,
          { field: "items", index: i, expected: dimension, received: copied.length },
        );
      }
      dimension ??= copied.length;
      const stored: StoredVector<T> = { vector: copied, norm, payload };
      const previous = positions.get(id);
      if (previous !== undefined) {
        prepared[previous] = { id, stored }; // last write wins within the batch
        continue;
      }
      positions.set(id, prepared.length);
      if (!this.items.has(id)) newIds += 1;
      prepared.push({ id, stored });
    }

    if (this.items.size + newIds > this.maxVectors) {
      throw new ValidationError(
        `upsert would exceed the store's maxVectors limit of ${this.maxVectors}.`,
        { field: "maxVectors", limit: this.maxVectors, size: this.items.size, adding: newIds },
      );
    }

    for (const entry of prepared) {
      this.items.set(entry.id, entry.stored);
      this.dimension ??= entry.stored.vector.length;
    }
  }

  async query(vector: readonly number[], options: QueryOptions<T>): Promise<VectorMatch<T>[]> {
    if (typeof options !== "object" || options === null) {
      throw new ValidationError("query() expects an options object.", { field: "options" });
    }
    const topK = assertPositiveInteger(options.topK, "topK");
    if (options.filter !== undefined && typeof options.filter !== "function") {
      throw new ValidationError('"filter" must be a function.', { field: "filter" });
    }
    const { vector: query, norm: queryNorm } = assertVector(vector, "query vector");
    if (this.dimension !== undefined && query.length !== this.dimension) {
      throw new ValidationError(
        `query vector has dimension ${query.length}, but this store holds dimension ${this.dimension} vectors.`,
        { field: "vector", expected: this.dimension, received: query.length },
      );
    }
    if (this.items.size === 0) return [];

    const matches: VectorMatch<T>[] = [];
    for (const [id, item] of this.items) {
      if (options.filter !== undefined && !options.filter(item.payload)) continue;
      let dot = 0;
      for (let i = 0; i < query.length; i++) {
        dot += query[i]! * item.vector[i]!;
      }
      matches.push({ id, score: dot / (queryNorm * item.norm), payload: item.payload });
    }
    // Best first; equal scores break by id so results are deterministic.
    matches.sort(
      (a, b) => b.score - a.score || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0),
    );
    return matches.slice(0, topK);
  }

  async delete(ids: readonly string[]): Promise<number> {
    if (!Array.isArray(ids)) {
      throw new ValidationError("delete() expects an array of ids.", { field: "ids" });
    }
    for (let i = 0; i < ids.length; i++) {
      if (typeof ids[i] !== "string" || ids[i]!.length === 0) {
        throw new ValidationError(`ids[${i}] must be a non-empty string.`, { field: "ids" });
      }
    }
    let removed = 0;
    for (const id of ids) {
      if (this.items.delete(id)) removed += 1;
    }
    // An empty store carries no dimension constraint anymore: it may be
    // reused with vectors from a different embedding model.
    if (this.items.size === 0) this.dimension = undefined;
    return removed;
  }
}
