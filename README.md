# MurphyChain

**Build AI. Connect Intelligence.**

A modern, fast, secure, modular AI framework for JavaScript and TypeScript.

```ts
import { AI } from "murphychain";

const ai = new AI({ provider: "openai", model: "gpt-4.1", apiKey: process.env.OPENAI_API_KEY });

const response = await ai.generate("Explain artificial intelligence.");
console.log(response.text);
```

```ts
const stream = await ai.stream("Count to ten.");
for await (const chunk of stream) process.stdout.write(chunk.text);
```

```ts
import { Prompt } from "murphychain";

const template = Prompt.template("Summarize {topic} in {words} words.");
const prompt = template.format({ topic: "gravity", words: 30 });
```

Structured output is **validated, portable, and self-repairing** — one Zod schema works on every provider:

```ts
import { z } from "murphychain";

const person = await ai.generateObject("Extract the person.", z.object({
  name: z.string(),
  age: z.number().int().positive(),
}));
console.log(person.object); // fully validated — no casts, no manual JSON.parse
```

Switching providers is a **one-line change** — the response contract is identical:

```ts
const ai = new AI({ provider: "gemini", model: "gemini-2.5-flash", apiKey: process.env.GEMINI_API_KEY });
```

## Packages

| Package | Purpose |
| --- | --- |
| [`murphychain`](./packages/murphychain) | Main entry point: re-exports the engine, lazily registers built-in providers |
| [`@murphychain/core`](./packages/core) | Engine: `AI` class, provider contract, prompts, tools, chains, structured output, events, errors, retry/timeout/cancellation |
| [`@murphychain/openai`](./packages/providers/openai) | OpenAI adapter (official `openai` SDK) |
| [`@murphychain/gemini`](./packages/providers/gemini) | Google Gemini adapter (official `@google/genai` SDK) |

Provider adapters are **optional peer dependencies** and are loaded lazily on first use — a provider you never call costs nothing.

## Install

```bash
pnpm add murphychain @murphychain/openai        # OpenAI
pnpm add murphychain @murphychain/gemini         # Gemini
```

```bash
npm install murphychain @murphychain/openai
yarn add murphychain @murphychain/openai
```

## Core concepts

### Generate, stream, embed

```ts
const response = await ai.generate({
  prompt: "Hello",
  messages: [{ role: "system", content: "Be brief." }, { role: "user", content: "Hello" }],
  temperature: 0.2,
  maxTokens: 500,
});

const stream = await ai.stream("Write a poem."); // AsyncIterable<AIChunk>, single-use
const vector = await ai.embed("search text");    // number[]
const vectors = await ai.embedMany(["a", "b"]);  // number[][]
```

`generate()` accepts a string, a `Message[]`, or a full request object.

### Tools

Give the model named, schema-validated functions it can request — argument types are inferred from your Zod schema:

```ts
import { Tool, executeToolCalls, z } from "murphychain";

const getWeather = Tool.define({
  name: "get_weather",
  description: "Get the current weather for a city",
  parameters: z.object({ city: z.string() }),
  execute: async ({ city }) => ({ city, tempC: 21 }),   // args are typed
});

// Dispatch the calls the model requested (from response.message.toolCalls):
const results = await executeToolCalls([getWeather], toolCalls);
// Results are role:"tool" messages — feed them straight back into generate().
```

Failures are serialized as `{"error": ...}` (redacted) so the model can see what went wrong and correct itself on the next turn; pass `{ onError: "throw" }` to fail fast instead. Tool schemas are converted once and cached.

### Chains

Sequence steps explicitly over shared state — no hidden orchestration, per-step timing included:

```ts
import { Chain } from "murphychain";

const chain = new Chain<{ city: string; summary?: string }>({ name: "report" })
  .step("fetch", async (state) => ({ weather: await getWeather.run({ city: state.city }) }))
  .step("summarize", (state) => ({ summary: `${state.weather.city}: ${state.weather.tempC}°C` }));

const { state, steps } = await chain.run({ city: "Oslo" });
// steps = [{ name: "fetch", index: 0, durationMs: 12 }, ...]
```

A failing step throws `ChainStepError` naming the step and index; `{ onError: "continue" }` runs the rest and reports all failures together. Steps can be cancelled via `signal`.

### Structured output

```ts
const response = await ai.generate({ prompt: "Extract Ada.", output: personSchema });
response.object;                       // schema-validated value
```

- The engine appends a JSON-only instruction, parses the reply (tolerating code fences and prose), and validates it against the schema — **implemented once in the engine, so it works with every provider**.
- An invalid reply triggers a **bounded repair loop** (3 total attempts) that re-asks with the concrete validation issues; exhaustion throws `OutputValidationError` carrying the issues.
- Cannot be combined with `tools` (use one or the other per request) and not supported by `stream()` (validation needs the complete reply).

### Prompt templates

```ts
const t = Prompt.template("Hello {name}, you have {count} messages.");
t.variables;                    // ["name", "count"]
t.format({ name: "Ada", count: 3 });
t.partial({ name: "Ada" });     // bind some variables, format the rest later
```

Missing or unknown variables throw `PromptTemplateError` — templates fail at the edges, not deep inside a provider call.

### Events (content-free by design)

```ts
ai.on("request:start", (e) => log.info({ provider: e.provider, chars: e.totalChars }));
ai.on("request:success", (e) => log.info({ latency: e.latencyMs, tokens: e.usage.totalTokens }));
ai.on("request:error", (e) => log.error({ code: e.error.code, retryable: e.error.retryable }));
ai.on("retry", (e) => log.warn({ attempt: e.attempt, delay: e.delayMs }));
ai.on("chunk", (e) => metrics.increment("chunks", { index: e.index }));
ai.on("abort", (e) => log.warn({ reason: e.reason }));
```

Events carry **counts and lengths only — never prompt or response content** — so they are safe to ship to a log aggregator. Listener exceptions are isolated and can never break a request.

### Retry, timeout, cancellation

```ts
const ai = new AI({
  provider: "openai",
  model: "gpt-4.1",
  timeoutMs: 30_000,   // generate: total deadline; stream: time-to-first-chunk. 0 disables.
  retries: 3,          // or a full RetryPolicy: { retries, baseDelayMs, maxDelayMs, jitter }
  signal: undefined,   // per-call: await ai.generate("hi", { signal: controller.signal })
});
```

- Retry is **single-sourced in the engine** (provider SDK retries are disabled) with full jitter, capped exponential backoff, and only `retryable` errors.
- `timeoutMs` for `stream` is a **time-to-first-chunk** deadline; the clock stops at the first delta.
- Breaking out of a stream loop **cancels** the upstream request.
- The engine races every attempt against the abort signal, so timeouts and cancellation are honored even if a provider ignores its signal.

### Errors

Every failure is a `MurphyChainError` with a stable `code`, a `retryable` flag, and **redacted messages** (API keys are never echoed):

```ts
import { MurphyChainError, ProviderRateLimitError, ConfigurationError } from "murphychain";

try {
  await ai.generate("hi");
} catch (error) {
  if (error instanceof MurphyChainError) {
    console.error(error.code, error.retryable, error.message);
  }
}
```

| Error | Meaning | Retried |
| --- | --- | --- |
| `ConfigurationError` | Bad config, unknown provider | no |
| `ValidationError` | Invalid input (empty prompt, limits exceeded) | no |
| `PromptTemplateError` | Missing/unknown template variables | no |
| `ProviderAuthError` | 401/403 | no |
| `ProviderRateLimitError` | 429 (carries `retryAfterMs`) | yes |
| `ProviderBadRequestError` | 400/404/422 | no |
| `ProviderServerError` | 5xx | yes |
| `ProviderConnectionError` | DNS/TLS/socket | yes |
| `ProviderTimeoutError` | Server-side timeout | yes |
| `ProviderContentFilterError` | Safety block / 451 | no |
| `ChainStepError` | A chain step failed (carries `step`/`index`) | no |
| `OutputValidationError` | Model reply failed schema validation after the repair loop (carries `issues`/`attempts`) | no |

### Custom providers

Implement the `ModelProvider` contract and register it — you get the whole engine (retry, timeout, events, validation, error taxonomy) for free:

```ts
import { AI, registerProvider, type ModelProvider } from "murphychain";

class MyProvider implements ModelProvider {
  readonly name = "my-provider";
  readonly capabilities = { generate: true, streaming: true, toolCalling: false, embeddings: false, systemMessage: true, structuredOutput: false };
  async generate(options) { /* call your backend */ }
  stream(options) { /* return AsyncIterable<AIChunk> */ }
  async embed(text) { throw new Error("unsupported"); }
  async embedMany(texts) { throw new Error("unsupported"); }
}

registerProvider("my-provider", () => new MyProvider());
const ai = new AI({ provider: "my-provider", model: "my-model" });
```

A complete, runnable version lives in [`examples/custom-provider`](./examples/custom-provider) — it runs with **no API key and no network**.

### Strict configuration

Config objects are validated with Zod `.strict()` schemas: a typo like `modle:` or `temprature:` fails loudly at construction or request time instead of being silently ignored. Provider options (`providerOptions`) are validated by each provider the same way.

## Examples

```bash
pnpm example:basic-chat        # generate + prompts + events + stats
pnpm example:streaming         # streaming, early break, single-use streams
pnpm example:provider-switch   # identical code on OpenAI and Gemini
pnpm example:custom-provider   # implement ModelProvider — no key required
pnpm example:workflows         # tools + chains + structured output — no key required
```

The first three need an API key (`OPENAI_API_KEY` or `GEMINI_API_KEY`) and print setup instructions if none is set; `custom-provider` and `workflows` always run.

## Development

```bash
pnpm install
pnpm typecheck   # tsc --noEmit across all packages and tests
pnpm test        # vitest (306 tests; 6 live API tests skip without keys)
pnpm build       # tsup ESM + CJS + d.ts for every package
```

Live integration tests run only when their key is present:

```bash
OPENAI_API_KEY=sk-... pnpm --filter @murphychain/openai test
GEMINI_API_KEY=AIza... pnpm --filter @murphychain/gemini test
```

See [`docs/architecture.md`](./docs/architecture.md) for the design rationale.

## Requirements

- Node.js >= 22.18
- TypeScript 5+ (the packages ship their own types)

## License

MIT © MurphTech
