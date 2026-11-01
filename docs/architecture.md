# MurphyChain Architecture

**Version:** 0.1.0 (MVP)
**Status:** Approved for implementation
**Organization:** MurphTech

> "Build AI. Connect Intelligence."

This document defines the architecture of MurphyChain: the layer model, package
graph, execution engine, provider contract, error model, security posture, and
the decisions taken for version 0.1.0. It is the reference against which
implementation changes are reviewed.

---

## 1. Goals and Non-Goals

### 1.1 Goals (0.1.0)

1. A tiny, provider-agnostic core with a two-line developer experience:
   `new AI(...)` → `await ai.generate(...)`.
2. A standardized provider contract with capability detection, streaming,
   tool-calling pass-through, usage tracking, and error normalization.
3. Production-grade execution controls in the core: timeouts, cancellation,
   bounded retries with backoff, input limits, and typed events.
4. Strict TypeScript, minimal dependencies, lazy loading of optional
   providers, and security enforced by runtime code rather than documentation.

### 1.2 Non-Goals (0.1.0 — explicitly deferred)

| Deferred feature | Target release |
| --- | --- |
| Chains, Tool class, structured output | 0.2.0 |
| Agent runtime, memory | 0.3.0 |
| Embeddings infrastructure, RAG, vector stores | 0.4.0 |
| MCP, advanced workflows, OpenTelemetry | 0.5.0 |
| Benchmarks vs. other frameworks | before 1.0.0 (no performance claims until reproducible) |

Provider-level `embed()` ships in 0.1.0 because the provider contract requires
it; the retrieval infrastructure that consumes it does not.

---

## 2. Layer Model

```
┌───────────────────────────────────────────────────────────────┐
│ A. Developer API layer                                        │
│    AI · Model · Prompt · (0.2+: Tool, Chain, Agent, ...)      │
├───────────────────────────────────────────────────────────────┤
│ B. Core execution engine                                     │
│    lifecycle · context · events · retry · timeout ·           │
│    cancellation · validation · error normalization            │
├───────────────────────────────────────────────────────────────┤
│ C. Provider adapter layer                                     │
│    ModelProvider contract · registry · capability detection   │
│    OpenAI · Gemini · (0.2+: Anthropic, Ollama, custom)        │
├───────────────────────────────────────────────────────────────┤
│ D. Agent runtime            (0.3.0)                           │
│ E. Data / retrieval layer    (0.4.0)                           │
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

---

## 5. Provider Contract

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
  MurphyChain error taxonomy (section 6), with secrets redacted from messages.
- **Context propagation:** `GenerateOptions` carries the abort signal;
  per-request metadata stays in the engine (events/tracing), not in payloads.

---

## 6. Error Model

```
MurphyChainError (base: code, retryable, details, cause)
├── ConfigurationError          developer error: bad config, unknown provider
├── ValidationError             invalid runtime input (zod issues, limits)
├── PromptTemplateError         missing/unknown template variables
├── UnsupportedCapabilityError  provider cannot serve this request
├── ExecutionTimeoutError       local deadline exceeded (not retried)
├── ExecutionCancelledError     caller aborted (not retried)
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

---

## 7. Security Posture (0.1.0 threat model)

| Risk | Control (enforced in code) |
| --- | --- |
| API key leakage | keys never appear in events, stats, or error messages; `redactSecrets()` strips `sk-…`/`AIza…`/bearer tokens from any provider message before it is surfaced |
| Credential rotation | `apiKey` accepts `() => string`, re-read per request |
| Prompt/PII leakage via telemetry | events are content-free by design (counts and lengths only) |
| Resource exhaustion | `maxInputChars`, `maxMessages`, bounded retries, bounded backoff, deadlines |
| Runaway requests | timeouts + cancellation on every call and stream; abortable backoff |
| Malformed/malicious provider output | structural validation of consumed fields; tool arguments parsed as data (never executed); no `eval` anywhere |
| Prompt injection (partial, 0.1.0) | tool definitions validated + capability-gated; full mitigations (permission boundaries, approval gates) land with the agent runtime |
| Supply chain | 3 runtime dependencies total (zod + 2 official SDKs), lockfile committed, CI dependency audit |
| SSRF | no user-supplied URL fetching exists in 0.1.0; `baseUrl` must be an http(s) URL; web loaders (0.4.0) will add private-IP/redirect controls |

---

## 8. Performance Principles

- Lazy provider loading; no provider code evaluated until first use.
- Single validation pass per call (config validated once at construction).
- No retries inside providers (single policy, no stacked backoff).
- Streaming avoids whole-response buffering; `raw` payload retention can be
  disabled per call (`includeRaw: false`).
- Zero-allocation hot path goals are measured, not assumed — benchmarks are a
  pre-1.0.0 deliverable, and **no performance claims are made until they are
  reproducible**.

---

## 9. Testing Strategy

- **Unit tests (mocked transport):** adapters inject a custom `fetch` into the
  official SDKs, so tests exercise the SDK's real parsing/error paths against
  canned HTTP responses — no network, no faked client interfaces.
- **Engine tests:** fake registered providers exercise retry, timeout,
  cancellation, capability gates, event ordering, and validation limits.
- **Type tests:** `expectTypeOf` assertions on the public API surface.
- **Integration tests:** live calls, skipped unless `OPENAI_API_KEY` /
  `GEMINI_API_KEY` are present.
- Rules: tests run before any release claim; failures are fixed, never
  suppressed.

---

## 10. Public API Surface (0.1.0)

```ts
import { AI, Model, Prompt, z } from "murphychain";

const ai = new AI({ provider: "openai", model: "gpt-4.1", apiKey: process.env.OPENAI_API_KEY });

const res  = await ai.generate("Explain artificial intelligence.");   // → AIResponse
const strm = await ai.stream("Write a JavaScript tutorial.");         // → AsyncIterable<AIChunk>
const emb  = await ai.embed("hello world");                           // → number[]

const prompt = Prompt.template("Explain {topic} in {language}.");
await ai.generate(prompt.format({ topic: "AI", language: "simple English" }));

ai.on("request:success", (e) => console.log(e.latencyMs, e.usage));
```

Advanced paths: pass a `ModelProvider` instance directly to `AI`, or
`registerProvider("name", factory)` for custom registries.

---

## 11. Release Mapping

| Milestone | Contents |
| --- | --- |
| **0.1.0 (this document)** | core engine, provider contract, OpenAI + Gemini, generate/stream/embed, prompt templates, events, retry/timeout/cancel |
| 0.2.0 | chains, `Tool`, structured output |
| 0.3.0 | agent runtime, memory |
| 0.4.0 | embeddings infrastructure, RAG, vector stores |
| 0.5.0 | MCP, advanced workflows, OpenTelemetry |
| 1.0.0 | security audit, stable APIs, comprehensive docs, NPM release |

Deviation note: the original milestone list placed "error handling" in 0.2.0;
foundational error types ship in 0.1.0 because the execution engine cannot be
correct without them.
