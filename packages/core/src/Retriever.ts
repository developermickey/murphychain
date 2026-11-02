import { AI } from "./AI";
import { ValidationError } from "./Errors";
import { InMemoryVectorStore, VectorStore } from "./VectorStore";
import type { VectorMatch } from "./VectorStore";
import { assertPositiveInteger } from "./validation";

/**
 * Retrieval-augmented generation (RAG) primitives: turn documents into
 * chunks, embed them into a {@link VectorStore}, and answer queries with the
 * most similar chunks.
 *
 * The {@link Retriever} composes three small, honest pieces:
 *
 * 1. {@link chunkText} — a naive fixed-size window splitter with overlap
 *    (bring your own splitter if you need semantic boundaries: pass
 *    pre-chunked documents as separate {@link Document}s).
 * 2. the engine's `embed`/`embedMany` — provider-portable, retried,
 *    validated (shape, count, and dimension consistency).
 * 3. a {@link VectorStore} — in-memory by default, swappable for any
 *    remote implementation of the abstract class.
 *
 * Re-indexing a document id replaces its previous chunks (stale chunk ids
 * are deleted only after the new vectors were embedded successfully, so a
 * failed embed never destroys existing data).
 */
export interface Document {
  /** Stable application-defined id. Re-indexing the same id replaces it. */
  id: string;
  text: string;
  /** Free-form metadata stored with every chunk of this document. */
  metadata?: Record<string, unknown> | undefined;
}

/** One retrievable piece of a document — the vector store's payload. */
export interface Chunk {
  /** `${documentId}#${index}` — stable across re-indexing. */
  id: string;
  text: string;
  documentId: string;
  /** Position of this chunk within the document's text. */
  index: number;
  metadata?: Record<string, unknown> | undefined;
}

export interface ChunkOptions {
  /** Maximum characters per chunk. Default: 1 000. */
  chunkSize?: number | undefined;
  /** Characters carried over between consecutive chunks. Must be < chunkSize. Default: 200. */
  chunkOverlap?: number | undefined;
}

export const DEFAULT_CHUNK_SIZE = 1_000;
export const DEFAULT_CHUNK_OVERLAP = 200;
/** Upper bound matching the engine's per-call embedMany limit. */
export const DEFAULT_EMBED_BATCH_SIZE = 100;

/**
 * Splits `text` into windows of at most `chunkSize` characters advancing by
 * `chunkSize - chunkOverlap`, so consecutive chunks share context. A final
 * remainder fully contained in the previous chunk is dropped (it would be a
 * duplicate). Pure function; whitespace is left untouched — the retriever
 * trims chunks when indexing.
 */
export function chunkText(text: string, options: ChunkOptions = {}): string[] {
  if (typeof text !== "string" || text.length === 0) {
    throw new ValidationError("chunkText() expects a non-empty string.", { field: "text" });
  }
  if (typeof options !== "object" || options === null) {
    throw new ValidationError("chunkText() expects an options object.", { field: "options" });
  }
  const chunkSize = options.chunkSize ?? DEFAULT_CHUNK_SIZE;
  assertPositiveInteger(chunkSize, "chunkSize");
  const explicitOverlap = options.chunkOverlap !== undefined;
  const overlap = options.chunkOverlap ?? DEFAULT_CHUNK_OVERLAP;
  if (typeof overlap !== "number" || !Number.isInteger(overlap) || overlap < 0) {
    throw new ValidationError('"chunkOverlap" must be an integer >= 0.', { field: "chunkOverlap" });
  }
  if (explicitOverlap && overlap >= chunkSize) {
    // An explicit overlap >= chunkSize would make step <= 0 (no progress).
    throw new ValidationError('"chunkOverlap" must be smaller than "chunkSize".', {
      field: "chunkOverlap",
    });
  }
  // The default overlap is clamped for small chunk sizes, so
  // chunkText(text, { chunkSize: 10 }) behaves as a plain window instead of
  // failing on a default the caller never chose.
  const effectiveOverlap = Math.min(overlap, chunkSize - 1);
  if (text.length <= chunkSize) return [text];

  const step = chunkSize - effectiveOverlap;
  const chunks: string[] = [];
  for (let start = 0; start < text.length; start += step) {
    const end = Math.min(start + chunkSize, text.length);
    // A remainder no longer than the overlap is fully inside the previous
    // chunk — pushing it would only duplicate text.
    if (start > 0 && end - start <= effectiveOverlap) break;
    chunks.push(text.slice(start, end));
    if (end === text.length) break;
  }
  return chunks;
}

export interface RetrieverConfig {
  /** Engine used for embedding (same provider configuration as the app). */
  ai: AI;
  /** Destination store. Default: a new InMemoryVectorStore<Chunk>. */
  store?: VectorStore<Chunk> | undefined;
  /** Chunk size in characters. Default: 1 000. */
  chunkSize?: number | undefined;
  /** Chunk overlap in characters (< chunkSize). Default: 200. */
  chunkOverlap?: number | undefined;
  /** Texts per embedMany call (<= 1 000, the engine limit). Default: 100. */
  embedBatchSize?: number | undefined;
}

/** Result of {@link Retriever.index}. */
export interface IndexResult {
  /** Documents processed. */
  documents: number;
  /** Chunks embedded and stored. */
  chunks: number;
}

export interface RetrieveOptions {
  /** Maximum matches to return (fewer when the store is smaller). */
  topK: number;
  /** Optional payload predicate applied before ranking. */
  filter?: ((payload: Chunk) => boolean) | undefined;
  /** Cancellation signal forwarded to the query embedding call. */
  signal?: AbortSignal | undefined;
}

/** Per-call controls for {@link Retriever.index}. */
export interface IndexOptions {
  /** Cancellation signal forwarded to every embedding batch. */
  signal?: AbortSignal | undefined;
}

/**
 * RAG retriever: index documents, retrieve similar chunks.
 *
 * ```ts
 * const retriever = new Retriever({ ai });
 * await retriever.index([{ id: "faq-1", text: "Reset your password …" }]);
 * const matches = await retriever.retrieve("how do I log in?", { topK: 3 });
 * matches[0].payload.text; // most similar chunk
 * ```
 */
export class Retriever {
  readonly chunkSize: number;
  readonly chunkOverlap: number;
  readonly embedBatchSize: number;
  readonly store: VectorStore<Chunk>;

  private readonly ai: AI;
  /** documentId → chunk ids currently stored (for stale cleanup on re-index). */
  private readonly indexed = new Map<string, string[]>();

  constructor(config: RetrieverConfig) {
    if (typeof config !== "object" || config === null) {
      throw new ValidationError("Retriever requires a configuration object.");
    }
    if (!(config.ai instanceof AI)) {
      throw new ValidationError('"ai" must be an AI instance.', { field: "ai" });
    }
    this.ai = config.ai;

    this.chunkSize = config.chunkSize ?? DEFAULT_CHUNK_SIZE;
    assertPositiveInteger(this.chunkSize, "chunkSize");
    const explicitOverlap = config.chunkOverlap !== undefined;
    const overlap = config.chunkOverlap ?? DEFAULT_CHUNK_OVERLAP;
    if (!Number.isInteger(overlap) || overlap < 0) {
      throw new ValidationError('"chunkOverlap" must be an integer >= 0.', {
        field: "chunkOverlap",
      });
    }
    if (explicitOverlap && overlap >= this.chunkSize) {
      throw new ValidationError('"chunkOverlap" must be smaller than "chunkSize".', {
        field: "chunkOverlap",
      });
    }
    // A defaulted overlap is clamped for small chunk sizes (same rule as
    // chunkText) so `chunkSize` alone never fails on a default.
    this.chunkOverlap = Math.min(overlap, this.chunkSize - 1);

    this.embedBatchSize = config.embedBatchSize ?? DEFAULT_EMBED_BATCH_SIZE;
    assertPositiveInteger(this.embedBatchSize, "embedBatchSize");
    if (this.embedBatchSize > 1_000) {
      throw new ValidationError('"embedBatchSize" must be <= 1000 (the engine limit).', {
        field: "embedBatchSize",
      });
    }

    const store = config.store ?? new InMemoryVectorStore<Chunk>();
    if (!(store instanceof VectorStore)) {
      throw new ValidationError(
        '"store" must be a VectorStore instance (e.g. InMemoryVectorStore).',
        { field: "store" },
      );
    }
    this.store = store;
  }

  /**
   * Chunks, embeds, and stores every document. Re-indexing a known id first
   * embeds the new chunks (a failure leaves the previous data untouched),
   * then replaces the old chunks. An empty array is a no-op.
   */
  async index(documents: readonly Document[], options: IndexOptions = {}): Promise<IndexResult> {
    if (!Array.isArray(documents)) {
      throw new ValidationError("index() expects an array of documents.", { field: "documents" });
    }
    if (documents.length === 0) return { documents: 0, chunks: 0 };
    if (typeof options !== "object" || options === null) {
      throw new ValidationError("index() expects an options object.", { field: "options" });
    }

    // 1. Validate every document and build its chunks before any embedding.
    const prepared: Array<{ documentId: string; chunkIds: string[]; chunks: Chunk[] }> = [];
    const seenIds = new Set<string>();
    for (let i = 0; i < documents.length; i++) {
      const doc: unknown = documents[i];
      if (typeof doc !== "object" || doc === null) {
        throw new ValidationError(`documents[${i}] must be an object.`, { field: "documents" });
      }
      const { id, text, metadata } = doc as Document;
      if (typeof id !== "string" || id.length === 0) {
        throw new ValidationError(`documents[${i}].id must be a non-empty string.`, {
          field: "documents",
        });
      }
      if (seenIds.has(id)) {
        throw new ValidationError(`Duplicate document id "${id}" in this index() call.`, {
          field: "documents",
        });
      }
      seenIds.add(id);
      if (typeof text !== "string" || text.trim().length === 0) {
        throw new ValidationError(`documents[${i}] ("${id}") must have non-empty text.`, {
          field: "documents",
        });
      }
      if (
        metadata !== undefined &&
        (typeof metadata !== "object" || metadata === null || Array.isArray(metadata))
      ) {
        throw new ValidationError(`documents[${i}].metadata must be a plain object.`, {
          field: "documents",
        });
      }

      const pieces = chunkText(text, {
        chunkSize: this.chunkSize,
        chunkOverlap: this.chunkOverlap,
      });
      const chunks: Chunk[] = [];
      const chunkIds: string[] = [];
      for (let c = 0; c < pieces.length; c++) {
        const chunkTextContent = pieces[c]!.trim();
        if (chunkTextContent.length === 0) continue; // whitespace carries nothing to embed
        const chunkId = `${id}#${c}`;
        chunks.push({
          id: chunkId,
          text: chunkTextContent,
          documentId: id,
          index: c,
          ...(metadata !== undefined ? { metadata } : {}),
        });
        chunkIds.push(chunkId);
      }
      prepared.push({ documentId: id, chunkIds, chunks });
    }

    // 2. Embed in bounded batches (sequential: one request in flight at a time).
    const allChunks = prepared.flatMap((entry) => entry.chunks);
    const vectors: number[][] = [];
    for (let start = 0; start < allChunks.length; start += this.embedBatchSize) {
      const batch = allChunks.slice(start, start + this.embedBatchSize);
      const embedded = await this.ai.embedMany(
        batch.map((chunk) => chunk.text),
        options.signal !== undefined ? { signal: options.signal } : {},
      );
      vectors.push(...embedded);
    }

    // 3. Commit: drop stale chunks of re-indexed documents, then upsert.
    //    (Embedding succeeded first — a failure above changed nothing.)
    const staleIds: string[] = [];
    for (const entry of prepared) {
      const previous = this.indexed.get(entry.documentId);
      if (previous === undefined) continue;
      const current = new Set(entry.chunkIds);
      for (const chunkId of previous) {
        if (!current.has(chunkId)) staleIds.push(chunkId);
      }
    }
    if (staleIds.length > 0) await this.store.delete(staleIds);
    await this.store.upsert(
      allChunks.map((chunk, i) => ({ id: chunk.id, vector: vectors[i]!, payload: chunk })),
    );
    for (const entry of prepared) {
      this.indexed.set(entry.documentId, entry.chunkIds);
    }
    return { documents: prepared.length, chunks: allChunks.length };
  }

  /**
   * Embeds the query and returns the closest chunks, best first. Fails fast
   * on bad input (before spending an embedding call).
   */
  async retrieve(query: string, options: RetrieveOptions): Promise<VectorMatch<Chunk>[]> {
    if (typeof options !== "object" || options === null) {
      throw new ValidationError("retrieve() expects an options object.", { field: "options" });
    }
    assertPositiveInteger(options.topK, "topK");
    if (options.filter !== undefined && typeof options.filter !== "function") {
      throw new ValidationError('"filter" must be a function.', { field: "filter" });
    }
    if (typeof query !== "string" || query.trim().length === 0) {
      throw new ValidationError("retrieve() expects a non-empty query.", { field: "query" });
    }
    const vector = await this.ai.embed(
      query,
      options.signal !== undefined ? { signal: options.signal } : {},
    );
    return this.store.query(vector, {
      topK: options.topK,
      ...(options.filter !== undefined ? { filter: options.filter } : {}),
    });
  }

  /** Removes every chunk of the given documents; returns chunks removed. */
  async remove(documentIds: readonly string[]): Promise<number> {
    if (!Array.isArray(documentIds)) {
      throw new ValidationError("remove() expects an array of document ids.", {
        field: "documentIds",
      });
    }
    const chunkIds: string[] = [];
    const seen = new Set<string>();
    for (let i = 0; i < documentIds.length; i++) {
      const id: unknown = documentIds[i];
      if (typeof id !== "string" || id.length === 0) {
        throw new ValidationError(`documentIds[${i}] must be a non-empty string.`, {
          field: "documentIds",
        });
      }
      const existing = this.indexed.get(id);
      if (existing !== undefined) {
        for (const chunkId of existing) {
          if (!seen.has(chunkId)) {
            seen.add(chunkId);
            chunkIds.push(chunkId);
          }
        }
      }
    }
    // Validate first, mutate second — an invalid id never half-removes.
    for (const id of documentIds as string[]) this.indexed.delete(id);
    if (chunkIds.length === 0) return 0;
    return this.store.delete(chunkIds);
  }
}
