# MurphyChain Architecture

**Version:** 0.4.0
**Status:** Approved for implementation
**Organization:** MurphTech

> "Build AI. Connect Intelligence."

This document defines the architecture of MurphyChain: the layer model, package
graph, execution engine, provider contract, retrieval layer, error model,
security posture, and the decisions taken from version 0.1.0 through 0.4.0. It
is the reference against which implementation changes are reviewed.

---

## 1. Goals and Non-Goals

### 1.1 Goals (0.1.0, extended in 0.2.0 through 0.4.0)

1. A tiny, provider-agnostic core with a two-line developer experience:
   `new AI(...)` → `await ai.generate(...)`.
2. A standardized provider contract with capability detection, streaming,
   tool-calling pass-through, usage tracking, and error normalization.
3. Production-grade execution controls in the core: timeouts, cancellation,
   bounded retries with backoff, input limits, and typed events.
4. Strict TypeScript, minimal dependencies, lazy loading of optional
   providers, and security enforced by runtime code rather than documentation.
5. (0.2.0) Workflow primitives — `Tool`, `Chain`, structured output — built
   on the same engine, portable across every provider, with no
   provider-specific code paths.
6. (0.3.0) An agent runtime — a bounded tool-calling loop with memory —
   built on the same engine, with hard step budgets, single-flight runs, and
   content-free observability.
7. (0.4.0) A retrieval layer — embeddings validated at the engine boundary,
   a swappable `VectorStore` contract, and a `Retriever` that chunks, embeds,
   and ranks documents — so RAG works with any provider and any store.

### 1.2 Non-Goals (explicitly deferred)

| Deferred feature | Target release |
| --- | --- |
| ~~Chains, Tool class, structured output~~ | shipped in 0.2.0 |
| ~~Agent runtime, memory~~ | shipped in 0.3.0 |
| ~~Embeddings infrastructure, RAG, vector stores~~ | shipped in 0.4.0 |
| MCP, advanced workflows, OpenTelemetry | 0.5.0 |
| Tool approval gates, permission boundaries | 1.0.0 |
| Benchmarks vs. other frameworks | before 1.0.0 (no performance claims until reproducible) |

Provider-level `embed()` ships in 0.1.0 because the provider contract requires
it; the retrieval infrastructure that consumes it ships in 0.4.0 (section 5).

---

## 2. Layer Model

```
┌───────────────────────────────────────────────────────────────┐
│ A. Developer API layer                                        │
│    AI · Model · Prompt · Tool · Chain · (0.3+: Agent, Memory) │
├───────────────────────────────────────────────────────────────┤
│ B. Core execution engine                                     │
│    lifecycle · context · events · retry · timeout ·           │
│    cancellation · validation · error normalization            │
├───────────────────────────────────────────────────────────────┤
│ C. Provider adapter layer                                     │
│    ModelProvider contract · registry · capability detection   │
│    OpenAI · Gemini · (0.2+: Anthropic, Ollama, custom)        │
├───────────────────────────────────────────────────────────────┤
│ D. Agent runtime + memory   (0.3.0 — shipped in core)         │
│ E. Data / retrieval layer    (0.4.0 — shipped in core)         │
│ F. MCP integration           (0.5.0)                           │
└───────────────────────────────────────────────────────────────┘
```

Dependency rule: arrows point downward only. `core` never imports a provider.
Providers import `core`. The `murphychain` facade composes both.

---

## 3. Package Graph

```
murphychain (facade)         — what users `pnpm add`
├── @murphychain/core        — engine + public API re-exports (dep)
├── @murphychain/openai      — optional peer, lazy-loaded (dynamic import)
├── @murphychain/gemini      — optional peer, lazy-loaded (dynamic import)
└── zod                      — re-exported as `z` for schema authoring

@murphychain/core
└── zod                      — runtime validation only

@murphychain/openai          @murphychain/gemini
├── @murphychain/core        ├── @murphychain/core
├── openai (official SDK)    ├── @google/genai (official SDK)
└── zod                      └── zod
```

Decisions:

- **Official SDKs, not hand-rolled HTTP.** OpenAI and Gemini adapters wrap the
  official SDKs; MurphyChain owns mapping, validation, and error translation —
  not transport protocols.
- **Providers are optional peers of the facade.** Installing `murphychain`
  alone costs one small dependency; provider SDKs (tens of MB) are only pulled
  in when the user installs the provider package. The facade registers
  *lazy loaders* (`() => import(...)`) so provider code is not evaluated until
  first use. A missing provider produces an actionable
  `ConfigurationError` ("run: pnpm add @murphychain/openai").
- **No Turborepo yet.** `pnpm -r` already builds topologically. Add Turborepo
  when the build graph grows (caching across CI/locals), not before.
- **Co-located tests** (`packages/<pkg>/tests/*.test.ts`) instead of a root
  `tests/` directory: each package owns its suite, and `vitest` runs everything
  from one root config.

---

## 4. Core Execution Engine

### 4.1 Request lifecycle

```
generate/stream(input, options)
  │
  ├─ 1. resolve request      (string | Message[] | GenerateRequest)
  │                          merge defaults: call > Model > AI config
  ├─ 2. validate             (zod: shapes; limits: maxMessages, maxInputChars)
  ├─ 3. capability check     (requested features vs provider capabilities)
  ├─ 4. create context       (requestId, deadline, abort linking, metadata)
  ├─ 5. emit "request:start" (content-free event)
  ├─ 6. resolve provider     (registry → lazy loader → cached instance)
  ├─ 7. execute with retry   (bounded attempts, abortable backoff)
  │        └─ provider.generate / provider.stream  ← abort signal threaded
  ├─ 8. emit "retry" | "request:success" | "request:error" | "abort"
  └─ 9. return AIResponse / AsyncIterable<AIChunk>
```

### 4.2 Timeouts, cancellation, retry

- **One `AbortController` per call.** The caller's `signal` is linked to it;
  a deadline timer aborts it. The abort *reason* is recorded so the engine can
  distinguish `ExecutionTimeoutError` from `ExecutionCancelledError`.
- **`timeoutMs` semantics:** for `generate` it bounds the whole call (default
  120 s); for `stream` it bounds *time-to-first-chunk* (long generations must
  not be killed mid-output). `0` disables the deadline.
- **Retries live in the core, not in providers.** Official SDKs retry
  internally by default; adapters configure `maxRetries: 0` so retry policy is
  single-sourced. Default policy: 2 retries, 250 ms → ×2 backoff, 4 s cap,
  full jitter. Only errors classified `retryable` are retried (429, 5xx,
  connection); auth/bad-request/content-filter errors are never retried.
- **Streams:** retry applies only until the first chunk is emitted (nothing
  observable has happened yet). After the first chunk, errors propagate
  immediately — replaying half a stream is never safe.
- **Backoff sleeps are abortable** so cancellation is instant, even mid-retry.

### 4.3 Events and observability

A tiny typed emitter (no `EventEmitter` dependency, browser-safe):

| Event | Payload (all content-free) |
| --- | --- |
| `request:start` | requestId, provider, model, messageCount, totalChars |
| `request:success` | requestId, latencyMs, usage, finishReason, chunkCount |
| `request:error` | requestId, latencyMs, error |
| `retry` | requestId, attempt, delayMs, error |
| `chunk` | requestId, index, textLength |
| `abort` | requestId, reason (`"timeout"` \| `"user"`) |

**Events never carry prompt or completion text.** A listener that throws is
isolated (caught) and cannot break request execution. `ai.stats()` exposes
cumulative counters (requests, errors, retries, tokens, latency) for
debugging. OpenTelemetry tracing arrives in 0.5.0.

### 4.4 Validation and limits

- Zod validates all external input: AI config, per-call requests, messages,
  tool definitions.
- `maxInputChars` (default 1,000,000) and `maxMessages` (default 10,000)
  bound memory before any network work happens.
- Provider responses are structurally validated on the fields we consume;
  a malformed payload raises a non-retryable `ProviderError`.

### 4.5 Tools (0.2.0)

- `Tool.define({ name, description, parameters, execute })` validates the
  definition eagerly (name pattern, description limits, schema kind) — a bad
  tool fails at definition time, not mid-request.
- `parameters` accepts a Zod v4 schema (typed: `execute` receives the
  inferred argument type) or a plain JSON Schema (execute receives `unknown`).
  Zod schemas are converted via `z.toJSONSchema` **once** and cached on the
  instance; `$schema` is stripped for provider compatibility.
- Arguments from the model are **parsed as data and validated** before
  `execute` runs — never evaluated. `executeToolCalls` dispatches
  sequentially (deterministic ordering, no interleaved side effects) and
  serializes failures as `{"error": "<redacted message>"}` so the model can
  self-correct on the next turn; `{ onError: "throw" }` opts into strict
  failure. The caller's `signal` reaches every `execute`.

### 4.6 Chains (0.2.0)

- `Chain` sequences named, unique steps over an immutable-by-default state
  object: each step returns a partial state that is shallow-merged over the
  current state **outside** the error catch, so a contract violation (step
  returned a non-object) surfaces as a raw `ValidationError` instead of being
  wrapped as a step failure.
- Per-step timing traces (`ChainRunResult.steps`) are always recorded; step
  failures throw `ChainStepError` (code `CHAIN_STEP_ERROR`) naming the step
  and index. `{ onError: "continue" }` runs the remaining steps and reports
  all failures together at the end.
- Cancellation is checked before each step and after each await: an aborted
  signal raises `ExecutionCancelledError` between steps, so no step starts
  once the run is cancelled. Chains are reusable (no shared state between
  runs) and independent of providers — a chain step may call `ai.generate`
  or do plain computation.

### 4.7 Structured output (0.2.0)

Structured output is implemented **once, in the engine**, so it is portable
across every provider — including ones with no native JSON mode:

1. `generate({ output: schema })` / `generateObject(input, schema)` requires
   a Zod v4 schema (converted to JSON Schema eagerly; a non-convertible
   schema fails before any request is sent). `output` cannot be combined
   with `tools`, and `stream()` rejects it (validation needs the complete
   reply).
2. The engine appends a JSON-only instruction as a trailing user message
   (counted against input limits, closest to generation).
3. The reply is parsed tolerantly (raw JSON, fenced JSON, JSON embedded in
   prose) and validated. Success sets `response.object`.
4. A failure triggers a **bounded repair loop**: the model's own (truncated)
   reply is echoed back plus a message naming the concrete validation
   issues, capped at `STRUCTURED_OUTPUT_MAX_ATTEMPTS = 3` total attempts to
   bound token spend. Exhaustion throws `OutputValidationError`
   (`OUTPUT_VALIDATION`, non-retryable) carrying the issues, the attempt
   count, and a redacted preview of the last reply.
5. Each attempt is an ordinary request — its own execution context, events,
   retry, and stats — so timeouts, cancellation, and observability apply
   per attempt. `response.usage`/`latencyMs` are summed across attempts.

No provider capability gate applies: nothing new is sent over the wire, the
instruction travels as an ordinary message.

### 4.8 Agent runtime (0.3.0)

`Agent` is the bounded tool-calling loop, built entirely on the engine's
public surface (`ai.generate` + `executeToolCalls`) — no provider-specific
code paths:

1. **Seed**: system prompt → remembered history → this run's input. The
   system prompt is configuration: it is re-seeded on every run and never
   written to memory.
2. **Ask**: one `generate()` with the tool definitions attached. Zod → JSON
   Schema conversion happens once per tool at construction (cached in
   `tool.normalized`); each step reuses the plain JSON Schema.
3. **Act**: tool calls are executed sequentially through `executeToolCalls`
   (Zod-validated arguments, redacted failures); results are appended as
   `role: "tool"` messages and the loop repeats.
4. **Stop** when the model answers in text (`stopped: "completed"`) or the
   step budget is spent (`stopped: "max_steps"`).

Boundaries enforced in code:

- `maxSteps` (default `DEFAULT_AGENT_MAX_STEPS = 10`) caps model calls per
  run. The final step's tool calls are **not executed** — their results would
  have no consumer — and that trailing assistant message is **not persisted**
  to memory (a dangling tool call breaks provider message validation on the
  next run).
- The caller's `signal` is checked before every step (raising
  `ExecutionCancelledError`) and forwarded to every request and tool;
  `timeoutMs` applies per model call; `metadata` propagates to the underlying
  `request:start` events.
- An instance is **single-flight**: a concurrent `run()` throws
  `ValidationError`, because memory and the step budget are shared state —
  create one agent per concurrent task.
- Memory is updated only after a run that made progress; a failed run leaves
  history untouched.
- `onToolError` mirrors `executeToolCalls`: `"return"` (default) serializes
  the failure as `{"error": …}` so the model can recover on its own;
  `"throw"` fails the run fast.
- Events (`step`, `tool`, `finish`) are content-free like every engine event
  (names, counts, timings — never message content) and listener exceptions
  are isolated. `AgentResult.steps` is the per-step trace: response, tool
  calls, tool messages, and duration.

### 4.9 Memory (0.3.0)

`Memory` is an abstract class — validated by `Agent` with `instanceof` —
with a deliberately tiny contract: `messages`, `remember()`, `clear()`.
Two implementations ship:

- **`BufferMemory`** — bounded FIFO (default 100 messages). Retention is
  enforced by one shared helper, `evictOldestUnits`, which treats an
  assistant message that requested tool calls and its `role: "tool"` results
  as a single atomic unit: they are always evicted together. A split pair is
  a conversation most providers reject.
- **`SummarizingMemory`** — keeps `keepRecent` (default 10) messages verbatim
  and folds everything older into a rolling summary via a single `generate()`
  call whenever the bound (`maxMessages`, default 40) is exceeded. The
  summary travels as a labeled `user`-role message
  (`[Conversation summary so far]`): every provider accepts mid-conversation
  user messages, whereas mid-conversation system messages have
  provider-specific semantics. The summary itself is truncated to 4 000
  characters so a rambling model cannot grow it unboundedly; an empty summary
  raises `ValidationError`; a summarization failure propagates with the buffer
  left intact (no silent data loss).

Both implementations store message content in-process only; it leaves the
process solely as part of the next request to the configured provider.

---

## 5. Retrieval Layer (0.4.0)

`embed()` has existed since 0.1.0, but nothing consumed it. 0.4.0 builds the
retrieval layer on top of the existing engine — no new packages, no new error
codes, no provider-specific code paths:

```
Document[] ──chunkText──▶ Chunk[] ──ai.embedMany (batched)──▶ VectorStore
                                                        query ▲
Query ─────────────────── ai.embed ──────────────────────────┘
```

### 5.1 Engine-level embedding validation

Providers are fallible: a malformed embedding (a string, an empty array, a
`NaN` component, a batch whose length does not match its inputs, vectors of
mixed dimensions) is a **provider bug, never a transient failure**. The engine
validates every embedding result *after* the retry loop:

- `assertEmbedding` / `assertEmbeddingBatch` are module-private to `AI.ts` —
  not part of the public API.
- Failures throw a non-retryable `ProviderError` (`code: PROVIDER_ERROR`) with
  a **shape-only** message (`array(3)`, `string`) — never payload contents.
- Batch checks enforce `length === inputs.length` and one consistent dimension
  across all vectors, so dimension mismatches fail at the engine boundary
  instead of surfacing much later inside a vector store.
- Validation runs inside the request's `try`, so stats (`errors`) and
  `request:error` events still fire.

### 5.2 VectorStore contract

```ts
abstract class VectorStore<T> {
  abstract get size(): number;
  abstract upsert(items: readonly UpsertItem<T>[]): Promise<void>;
  abstract query(vector: readonly number[], options: QueryOptions<T>): Promise<VectorMatch<T>[]>;
  abstract delete(ids: readonly string[]): Promise<number>;
}
```

- **Async methods** so remote stores (pgvector, a service) implement the same
  contract as the in-memory one; `instanceof VectorStore` validates injection.
- **`VectorMatch` carries `{ id, score, payload }` — never the vector** —
  so callers cannot mutate stored state through a query result.
- **`filter` is a predicate** over the payload, applied before ranking. Remote
  adapters that cannot push a filter down must reject it rather than scan
  unbounded data (documented on `QueryOptions.filter`).
- **`InMemoryVectorStore`** is the shipped implementation: linear scan with
  cosine similarity, cached norms, defensive vector copies on `upsert`,
  validate-whole-batch-then-commit (an invalid item writes nothing), one
  dimension per store (established by the first batch, reset when emptied),
  zero-magnitude vectors rejected (cosine is undefined for them), a
  `maxVectors` cap (default 10 000), and deterministic tie-breaking
  (score descending, then id ascending) so tests and reruns are stable.

### 5.3 Retriever (RAG)

```ts
const retriever = new Retriever({ ai, store });            // chunkSize 1000, overlap 200
await retriever.index([{ id: "faq-1", text: "Reset your password …" }]);
const matches = await retriever.retrieve("how do I log in?", { topK: 3 });
matches[0].payload.text;                                   // most similar chunk
```

- **`chunkText`** is a naive fixed-size window splitter with overlap — a pure
  function; a final remainder fully contained in the previous chunk is dropped.
  Semantic splitting is out of scope: pass pre-chunked documents instead.
- **`index()` validates every document first**, embeds sequentially in bounded
  batches (`embedBatchSize`, default 100, capped at the engine's 1 000 limit),
  then deletes stale chunk ids and upserts. Embedding happens **before**
  deletion, so a failed embed never destroys existing data. Re-indexing a
  known id replaces its previous chunks (tracked in an internal
  `indexed: Map<docId, chunkIds>`).
- **`retrieve()`** fails fast on invalid input *before* spending an embedding
  call, then embeds the query and delegates to the store.
- **`remove(documentIds)`** validates all ids before mutating and returns the
  number of chunks removed.
- Documents and metadata are validated at the boundary (`ValidationError`),
  consistent with the rest of the engine.

### 5.4 What the retrieval layer does not do

- **No web loaders in 0.4.0** — fetching and parsing remote content (with
  private-IP/redirect controls) stays deferred; documents are supplied by the
  application.
- **No hybrid search, no rerankers, no persistence backends** — the contract
  is designed so they can be added without breaking changes.
- **No performance claims** — the in-memory scan is O(n) per query and is
  documented as such.

---

## 6. Provider Contract

```ts
interface ModelProvider {
  readonly name: string;
  readonly capabilities: ProviderCapabilities;
  generate(options: GenerateOptions): Promise<AIResponse>;
  stream(options: GenerateOptions): AsyncIterable<AIChunk>;
  embed(text: string, options?: EmbedOptions): Promise<number[]>;
  embedMany(texts: string[], options?: EmbedOptions): Promise<number[][]>;
}
```

- **Capability detection:** `ProviderCapabilities` advertises
  `streaming / toolCalling / embeddings / structuredOutput / systemMessage`.
  The engine throws `UnsupportedCapabilityError` *before* calling a provider
  that cannot serve a request — never a silent degradation.
- **Unified types:** one `Message`, `AIResponse`, `AIChunk`, `TokenUsage`
  shape for every provider. Switching providers is a one-line change.
- **Error normalization:** every adapter translates SDK failures into the
  MurphyChain error taxonomy (section 7), with secrets redacted from messages.
- **Context propagation:** `GenerateOptions` carries the abort signal;
  per-request metadata stays in the engine (events/tracing), not in payloads.

---

## 7. Error Model

```
MurphyChainError (base: code, retryable, details, cause)
├── ConfigurationError          developer error: bad config, unknown provider
├── ValidationError             invalid runtime input (zod issues, limits)
├── PromptTemplateError         missing/unknown template variables
├── UnsupportedCapabilityError  provider cannot serve this request
├── ExecutionTimeoutError       local deadline exceeded (not retried)
├── ExecutionCancelledError     caller aborted (not retried)
├── ChainStepError              a chain step failed (step/index; not retried)
├── OutputValidationError       structured output failed validation after the
│                               bounded repair loop (issues/attempts; not retried)
├── ProviderError               normalized provider failure
│   ├── ProviderAuthError       401/403 — never retried
│   ├── ProviderRateLimitError  429 — retried (retryAfterMs respected)
│   ├── ProviderBadRequestError 400/404 — never retried
│   ├── ProviderServerError     5xx — retried
│   ├── ProviderConnectionError DNS/sockets — retried
│   ├── ProviderTimeoutError    provider-side timeout — retried
│   └── ProviderContentFilterError — never retried
```

After retries are exhausted the **last error is thrown as-is** (so callers can
`instanceof ProviderRateLimitError`); attempt counts are visible through
`retry` events and `ai.stats()`.

The 0.3.0 agent runtime introduces **no new error codes**: input and
configuration failures are `ValidationError`, step-boundary cancellation and
per-call deadlines are `ExecutionCancelledError` / `ExecutionTimeoutError`,
and tool failures flow through `executeToolCalls`' existing serialization or
propagate unchanged with `onToolError: "throw"`.

The 0.4.0 retrieval layer likewise introduces **no new error codes**:
malformed provider embeddings are a non-retryable `ProviderError`
(`PROVIDER_ERROR`), and every retrieval input failure (bad documents, invalid
chunking/batching options, capacity exceeded) is a `ValidationError`.

---

## 8. Security Posture (0.1.0 threat model, extended in 0.2.0 through 0.4.0)

| Risk | Control (enforced in code) |
| --- | --- |
| API key leakage | keys never appear in events, stats, or error messages; `redactSecrets()` strips `sk-…`/`AIza…`/bearer tokens from any provider message before it is surfaced |
| Credential rotation | `apiKey` accepts `() => string`, re-read per request |
| Prompt/PII leakage via telemetry | events are content-free by design (counts and lengths only) |
| Resource exhaustion | `maxInputChars`, `maxMessages`, bounded retries, bounded backoff, deadlines, bounded agent loops (`maxSteps`, bounded summaries), bounded vector stores (`maxVectors`, default 10 000), bounded embedding batches (`embedBatchSize` ≤ 1 000) |
| Runaway requests | timeouts + cancellation on every call and stream; abortable backoff; agent checks `signal` between steps; retrieval forwards `signal` to every embedding call |
| Runaway agent loops | `maxSteps` hard-caps model calls per run; the final step's tool calls are never executed; single-flight per instance |
| Malformed/malicious provider output | structural validation of consumed fields; tool arguments parsed as data (never executed); no `eval` anywhere; embedding results validated at the engine boundary (shape, count, dimension consistency) with shape-only error messages |
| Model misbehavior (structured output) | JSON parsed + schema-validated before use; repair loop bounded to 3 attempts (no runaway token spend); `OutputValidationError` preview redacted |
| Tool failure leakage | tool errors serialized to the model as `{"error": …}` after `redactSecrets()`; secrets never reach the conversation |
| Prompt injection (partial) | tool definitions validated + capability-gated; the 0.3.0 agent bounds tool loops (`maxSteps`) and executes only registered tools with validated arguments; permission boundaries and approval gates remain deferred (1.0.0) |
| Store poisoning / memory abuse | vector stores validate every vector (finite components, non-zero magnitude, one dimension per store) before writing; whole batch validated before commit; re-indexing deletes stale chunks only after embedding succeeds |
| Supply chain | 3 runtime dependencies total (zod + 2 official SDKs), lockfile committed, CI dependency audit |
| SSRF | no user-supplied URL fetching exists in 0.1.0; `baseUrl` must be an http(s) URL; web loaders remain deferred — when added they must add private-IP/redirect controls |

---

## 9. Performance Principles

- Lazy provider loading; no provider code evaluated until first use.
- Single validation pass per call (config validated once at construction).
- No retries inside providers (single policy, no stacked backoff).
- Streaming avoids whole-response buffering; `raw` payload retention can be
  disabled per call (`includeRaw: false`).
- Zero-allocation hot path goals are measured, not assumed — benchmarks are a
  pre-1.0.0 deliverable, and **no performance claims are made until they are
  reproducible**.

---

## 10. Testing Strategy

- **Unit tests (mocked transport):** adapters inject a custom `fetch` into the
  official SDKs, so tests exercise the SDK's real parsing/error paths against
  canned HTTP responses — no network, no faked client interfaces.
- **Engine tests:** fake registered providers exercise retry, timeout,
  cancellation, capability gates, event ordering, and validation limits.
- **Agent/memory tests:** scripted fake providers drive multi-step loops,
  budget exhaustion (final-step calls unexecuted), cancellation between
  steps, single-flight guards, and memory retention invariants (atomic
  tool-unit eviction, system prompt never persisted, no partial state on
  summarization failure).
- **Retrieval tests:** deterministic fake embeddings drive chunk windowing,
  batch-bounded indexing, stale-chunk replacement, filter-before-rank,
  capacity bounds, dimension/zero-vector rejection, and the embed-before-delete
  failure invariant (a failed embed leaves existing data untouched).
- **Type tests:** `expectTypeOf` assertions on the public API surface.
- **Integration tests:** live calls, skipped unless `OPENAI_API_KEY` /
  `GEMINI_API_KEY` are present.
- Rules: tests run before any release claim; failures are fixed, never
  suppressed.

---

## 11. Public API Surface (0.2.0, extended in 0.3.0 and 0.4.0)

```ts
import {
  AI, Model, Prompt, Tool, Chain, Agent, BufferMemory,
  Retriever, InMemoryVectorStore, chunkText, z,
} from "murphychain";

const ai = new AI({ provider: "openai", model: "gpt-4.1", apiKey: process.env.OPENAI_API_KEY });

const res  = await ai.generate("Explain artificial intelligence.");   // → AIResponse
const strm = await ai.stream("Write a JavaScript tutorial.");         // → AsyncIterable<AIChunk>
const emb  = await ai.embed("hello world");                           // → number[]

const prompt = Prompt.template("Explain {topic} in {language}.");
await ai.generate(prompt.format({ topic: "AI", language: "simple English" }));

// Structured output (engine-level, provider-portable)
const obj = await ai.generateObject("Extract the person.", z.object({
  name: z.string(),
  age: z.number().int(),
}));                                                                  // → AIResponse & { object }

// Tools
const weather = Tool.define({
  name: "get_weather",
  description: "Get the current weather for a city",
  parameters: z.object({ city: z.string() }),
  execute: async ({ city }) => getWeather(city),                      // args are typed
});

// Chains
const chain = new Chain<{ city: string; report?: string }>()
  .step("fetch", async (state) => ({ data: await getWeather(state.city) }))
  .step("summarize", (state) => ({ report: summarize(state.data) }));
const { state, steps } = await chain.run({ city: "Oslo" });           // steps = timings

// Agents (bounded tool-calling loop) with memory across runs (0.3.0)
const agent = new Agent({
  name: "researcher",
  system: "You answer weather questions using the tools provided.",
  ai,
  tools: [weather],
  memory: new BufferMemory({ maxMessages: 100 }),
  maxSteps: 8,
});
const result = await agent.run("What is the weather in Oslo?");       // → AgentResult
agent.on("step", (e) => console.log(e.index, e.toolCalls));           // content-free events

// Retrieval / RAG (0.4.0)
const retriever = new Retriever({ ai, store: new InMemoryVectorStore() });
await retriever.index([{ id: "faq-1", text: "Reset your password …" }]);
const matches = await retriever.retrieve("how do I log in?", { topK: 3 }); // → VectorMatch<Chunk>[]
matches[0].payload.text;                                               // most similar chunk

ai.on("request:success", (e) => console.log(e.latencyMs, e.usage));
```

Advanced paths: pass a `ModelProvider` instance directly to `AI`, or
`registerProvider("name", factory)` for custom registries.

---

## 12. Release Mapping

| Milestone | Contents |
| --- | --- |
| **0.1.0** | core engine, provider contract, OpenAI + Gemini, generate/stream/embed, prompt templates, events, retry/timeout/cancel |
| **0.2.0** | `Tool` + `executeToolCalls`, `Chain`, engine-level structured output (`generateObject`) |
| **0.3.0** | agent runtime (`Agent`), memory (`BufferMemory`, `SummarizingMemory`), agent events, `onCallComplete` |
| **0.4.0 (this document)** | embedding validation at the engine boundary, `VectorStore` + `InMemoryVectorStore`, `Retriever` + `chunkText` (RAG) |
| 0.5.0 | MCP, advanced workflows, OpenTelemetry |
| 1.0.0 | security audit, stable APIs, comprehensive docs, NPM release |

Deviation note: the original milestone list placed "error handling" in 0.2.0;
foundational error types ship in 0.1.0 because the execution engine cannot be
correct without them.
