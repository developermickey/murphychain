/**
 * Retrieval-augmented generation (RAG) — runs fully offline: no API key, no
 * network.
 *
 *   pnpm example:rag
 *
 * Demonstrates: `chunkText` windowing with overlap, `Retriever` indexing
 * documents into an `InMemoryVectorStore` through bounded embedding batches,
 * cosine-similarity query with payload filters, re-indexing that replaces
 * stale chunks, removal, the store's `maxVectors` memory bound, and a full
 * retrieve → prompt → generate → answer loop whose answer can only come from
 * the retrieved context.
 */
import {
  AI,
  InMemoryVectorStore,
  Retriever,
  ValidationError,
  chunkText,
  type AIResponse,
  type GenerateOptions,
  type ModelProvider,
  type ProviderCapabilities,
} from "murphychain";

// ---------------------------------------------------------------------------
// 1. Offline embeddings: a bag-of-words hashed into fixed buckets.
// ---------------------------------------------------------------------------

const DIMENSIONS = 64;

/** FNV-1a over UTF-16 code units — stable across runs and platforms. */
function hash(text: string): number {
  let value = 0x811c9dc5;
  for (let i = 0; i < text.length; i++) {
    value ^= text.charCodeAt(i);
    value = Math.imul(value, 0x01000193);
  }
  return value >>> 0;
}

function tokens(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((token) => token.length > 0);
}

/** Deterministic embedding: word counts hashed into buckets, L2-normalized. */
function embed(text: string): number[] {
  const vector = new Array<number>(DIMENSIONS).fill(0);
  for (const token of tokens(text)) {
    const index = hash(token) % DIMENSIONS;
    vector[index] = (vector[index] ?? 0) + 1;
  }
  let norm = 0;
  for (const value of vector) norm += value * value;
  norm = Math.sqrt(norm);
  if (norm === 0) {
    vector[0] = 1; // token-less text still yields a valid, non-zero vector
    return vector;
  }
  return vector.map((value) => value / norm);
}

const CAPABILITIES: ProviderCapabilities = Object.freeze({
  generate: true,
  streaming: false,
  toolCalling: false,
  embeddings: true,
  systemMessage: true,
  structuredOutput: false,
});

// ---------------------------------------------------------------------------
// 2. An extractive offline "model": it can only answer from the context the
//    application retrieved — if the context does not contain the answer, it
//    says so. This is what makes the RAG contract checkable end to end.
// ---------------------------------------------------------------------------

const CONTEXT_PATTERN = /Context:\n([\s\S]*?)\n\nQuestion:/;

/** Picks the context sentence sharing the most words with the question. */
function answerFromContext(context: string, question: string): string {
  const questionTokens = new Set(tokens(question));
  let best = "";
  let bestScore = 0;
  for (const sentence of context.split(/(?<=[.!?])\s+/)) {
    const trimmed = sentence.trim();
    if (trimmed.length === 0) continue;
    const score = tokens(trimmed).reduce(
      (sum, token) => sum + (questionTokens.has(token) ? 1 : 0),
      0,
    );
    if (score > bestScore) {
      bestScore = score;
      best = trimmed;
    }
  }
  if (bestScore === 0) {
    return "I don't know — the context does not answer the question.";
  }
  return best;
}

/** Counts embedMany batches so the example can verify bounded batching. */
const embedBatches: number[] = [];

const provider: ModelProvider = {
  name: "offline-rag",
  capabilities: CAPABILITIES,
  async generate(options: GenerateOptions): Promise<AIResponse> {
    const prompt = options.messages[options.messages.length - 1]?.content ?? "";
    const match = CONTEXT_PATTERN.exec(prompt);
    const questionMarker = prompt.lastIndexOf("Question:");
    const question = questionMarker >= 0 ? prompt.slice(questionMarker + 9).trim() : "";
    const text =
      match !== null && match[1] !== undefined && match[1].length > 0
        ? answerFromContext(match[1], question)
        : "I don't know — no context was provided.";
    const promptTokens = Math.ceil(prompt.length / 4);
    const completionTokens = Math.ceil(text.length / 4);
    return {
      text,
      provider: "offline-rag",
      model: "offline-rag-1",
      message: { role: "assistant", content: text },
      finishReason: "stop",
      usage: { promptTokens, completionTokens, totalTokens: promptTokens + completionTokens },
      latencyMs: 1,
    };
  },
  stream(): AsyncIterable<never> {
    throw new Error("streaming is not used by this example");
  },
  async embed(text: string): Promise<number[]> {
    return embed(text);
  },
  async embedMany(texts: string[]): Promise<number[][]> {
    embedBatches.push(texts.length);
    return texts.map(embed);
  },
};

// ---------------------------------------------------------------------------
// 3. Documents and chunking.
// ---------------------------------------------------------------------------

const documents = [
  {
    id: "cat-facts",
    text: "Cats purr at a frequency between 25 and 150 hertz. A domestic cat sleeps for twelve to sixteen hours each day.",
  },
  {
    id: "dog-facts",
    text: "Dogs wag their tails to communicate excitement. A trained dog can learn hundreds of words.",
  },
  {
    id: "space-facts",
    text: "Light from the sun takes eight minutes to reach earth. Jupiter is the largest planet in the solar system.",
  },
];

const longSpaceText = `
Light from the sun takes eight minutes to reach earth, which means we always
see the sun as it was eight minutes ago. Jupiter is the largest planet in the
solar system and is so massive that it could contain all other planets combined.
The four largest moons of Jupiter — Io, Europa, Ganymede and Callisto — were
discovered by Galileo in 1610 and are still studied by spacecraft today.
`.trim();

console.log("── Chunking ────────────────────────────────────────────");
const CHUNK_SIZE = 120;
const CHUNK_OVERLAP = 30;
const longChunks = chunkText(longSpaceText, { chunkSize: CHUNK_SIZE, chunkOverlap: CHUNK_OVERLAP });
console.log(`long document: ${longSpaceText.length} chars → ${longChunks.length} chunks`);
for (const [i, chunk] of longChunks.entries()) {
  console.log(`  chunk ${i}: ${chunk.length} chars`);
}

// Consecutive windows share exactly the overlap region.
if (longChunks.length < 2) {
  throw new Error(`chunking contract violated: expected >1 chunk, got ${longChunks.length}`);
}
for (let i = 1; i < longChunks.length; i++) {
  const shared = longChunks[i]!.slice(0, CHUNK_OVERLAP) === longChunks[i - 1]!.slice(-CHUNK_OVERLAP);
  if (!shared) {
    throw new Error(`chunking contract violated: chunk ${i} does not carry the overlap`);
  }
}

// ---------------------------------------------------------------------------
// 4. Indexing: documents → chunks → bounded embedding batches → store.
// ---------------------------------------------------------------------------

console.log("\n── Indexing ────────────────────────────────────────────");
const ai = new AI({ provider, model: "offline-rag-1" });
const retriever = new Retriever({
  ai,
  chunkSize: CHUNK_SIZE,
  chunkOverlap: CHUNK_OVERLAP,
  embedBatchSize: 2,
});

const indexed = await retriever.index(documents);
console.log(
  `indexed ${indexed.documents} documents → ${indexed.chunks} chunks ` +
    `(batches so far: [${embedBatches.join(", ")}])`,
);

const totalChunks = documents.reduce(
  (sum, doc) => sum + chunkText(doc.text, { chunkSize: CHUNK_SIZE, chunkOverlap: CHUNK_OVERLAP }).length,
  0,
);
if (indexed.chunks !== totalChunks || retriever.store.size !== totalChunks) {
  throw new Error(
    `index contract violated: expected ${totalChunks} chunks, got ${indexed.chunks} indexed / ${retriever.store.size} stored`,
  );
}
if (embedBatches.length !== Math.ceil(totalChunks / 2) || embedBatches.some((n) => n > 2)) {
  throw new Error(
    `batching contract violated: expected batches of <=2, got [${embedBatches.join(", ")}]`,
  );
}

// ---------------------------------------------------------------------------
// 5. Query: cosine similarity, ranking, and payload filters.
// ---------------------------------------------------------------------------

console.log("\n── Query ──────────────────────────────────────────────");
for (const [query, expected] of [
  ["cat sleeps hours", "cat-facts"],
  ["jupiter planet", "space-facts"],
] as const) {
  const matches = await retriever.retrieve(query, { topK: 2 });
  const top = matches[0];
  if (top === undefined || top.payload.documentId !== expected) {
    throw new Error(
      `retrieval contract violated: "${query}" should rank ${expected} first, got ${top?.payload.documentId ?? "nothing"}`,
    );
  }
  if (top.score < -1 || top.score > 1) {
    throw new Error(`retrieval contract violated: score ${top.score} outside [-1, 1]`);
  }
  for (let i = 1; i < matches.length; i++) {
    if (matches[i - 1]!.score < matches[i]!.score) {
      throw new Error("retrieval contract violated: matches are not sorted best-first");
    }
  }
  console.log(`"${query}" → ${top.payload.documentId} (score ${top.score.toFixed(3)})`);
}

const filtered = await retriever.retrieve("cat", {
  topK: 5,
  filter: (payload) => payload.documentId === "dog-facts",
});
if (filtered.length === 0 || filtered.some((m) => m.payload.documentId !== "dog-facts")) {
  throw new Error("filter contract violated: filter must apply before ranking");
}
console.log(`filtered to dog-facts only: ${filtered.length} match(es)`);

// ---------------------------------------------------------------------------
// 6. RAG loop: retrieve → stuff into the prompt → generate → answer.
// ---------------------------------------------------------------------------

console.log("\n── RAG loop ───────────────────────────────────────────");
const question = "cat sleeps hours";
const context = await retriever.retrieve(question, { topK: 1 });
if (context.length === 0) {
  throw new Error("rag contract violated: nothing retrieved for the demo question");
}
const prompt = `Context:\n${context[0]!.payload.text}\n\nQuestion: ${question}`;
const answer = await ai.generate(prompt);
console.log(`answer: ${answer.text}`);

// The answer must come from the retrieved chunk — without the context the
// offline model refuses, so a correct answer proves the whole pipeline.
if (!answer.text.includes("twelve to sixteen hours")) {
  throw new Error(
    `rag contract violated: expected the retrieved fact in the answer, got "${answer.text}"`,
  );
}

// ---------------------------------------------------------------------------
// 7. Re-indexing replaces stale chunks; removal drops whole documents.
// ---------------------------------------------------------------------------

console.log("\n── Re-index & remove ───────────────────────────────────");
const before = retriever.store.size;
const longDocChunks = chunkText(longSpaceText, { chunkSize: CHUNK_SIZE, chunkOverlap: CHUNK_OVERLAP }).length;

await retriever.index([{ id: "space-facts", text: longSpaceText }]);
const grown = retriever.store.size;
if (grown !== before - 1 + longDocChunks) {
  throw new Error(
    `re-index contract violated: expected ${before - 1 + longDocChunks} chunks after expansion, got ${grown}`,
  );
}

await retriever.index([documents[2]!]); // back to the short version
const shrunk = retriever.store.size;
if (shrunk !== before) {
  throw new Error(
    `stale-chunk contract violated: expected ${before} chunks after shrinking, got ${shrunk}`,
  );
}
console.log(`space-facts: 1 chunk → ${longDocChunks} chunks → 1 chunk (stale chunks cleaned up)`);

const dogChunks = chunkText(documents[1]!.text, { chunkSize: CHUNK_SIZE, chunkOverlap: CHUNK_OVERLAP }).length;
const removed = await retriever.remove(["dog-facts"]);
if (removed !== dogChunks || retriever.store.size !== shrunk - dogChunks) {
  throw new Error(
    `remove contract violated: expected ${dogChunks} chunks removed, got ${removed}`,
  );
}
console.log(`removed dog-facts (${removed} chunks); store now holds ${retriever.store.size}`);

// ---------------------------------------------------------------------------
// 8. The store's memory bound: an unbounded document stream cannot grow the
//    in-memory store past its configured limit.
// ---------------------------------------------------------------------------

console.log("\n── Store bound ────────────────────────────────────────");
const bounded = new InMemoryVectorStore<{ name: string }>({ maxVectors: 2 });
await bounded.upsert([
  { id: "a", vector: embed("alpha"), payload: { name: "a" } },
  { id: "b", vector: embed("beta"), payload: { name: "b" } },
]);
let rejected = false;
try {
  await bounded.upsert([{ id: "c", vector: embed("gamma"), payload: { name: "c" } }]);
} catch (error) {
  rejected = error instanceof ValidationError && /maxVectors/.test(error.message);
}
if (!rejected) {
  throw new Error("bound contract violated: the third vector should exceed maxVectors");
}
console.log("maxVectors=2 enforced: the third upsert was rejected");

console.log("\nDone — chunking, indexing, retrieval, RAG, and bounds, no network needed.");
