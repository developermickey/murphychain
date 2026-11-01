/**
 * Custom provider — implement the ModelProvider contract, get the whole
 * engine for free. Runs fully offline: no API key, no network.
 *
 *   pnpm example:custom-provider
 *
 * Demonstrates: `registerProvider`, capability declarations, generate/stream/
 * embed implementations, engine-owned retry (the provider just throws a
 * retryable error), the unified error taxonomy, and content-free events.
 */
import {
  AI,
  ProviderBadRequestError,
  ProviderServerError,
  registerProvider,
  type AIChunk,
  type AIResponse,
  type GenerateOptions,
  type ModelProvider,
  type ProviderCapabilities,
} from "murphychain";

// ---------------------------------------------------------------------------
// 1. A provider. This one is deterministic and local — a real integration
//    would call its vendor's official SDK here (see @murphychain/openai and
//    @murphychain/gemini for production examples).
// ---------------------------------------------------------------------------

const CAPABILITIES: ProviderCapabilities = Object.freeze({
  generate: true,
  streaming: true,
  toolCalling: false,
  embeddings: true,
  systemMessage: true,
  structuredOutput: false,
});

/** Deterministic 8-dimension vector: hash-bucketed word counts. */
function embedLocally(text: string): number[] {
  const vector = new Array<number>(8).fill(0);
  for (const word of text.toLowerCase().split(/\s+/)) {
    if (word.length === 0) continue;
    let hash = 0;
    for (let i = 0; i < word.length; i++) hash = (hash * 31 + word.charCodeAt(i)) >>> 0;
    vector[hash % 8] = (vector[hash % 8] ?? 0) + 1;
  }
  const norm = Math.sqrt(vector.reduce((sum, v) => sum + v * v, 0)) || 1;
  return vector.map((v) => v / norm);
}

/** Counts words as a stand-in for real token accounting. */
function countWords(text: string): number {
  return text.trim().length === 0 ? 0 : text.trim().split(/\s+/).length;
}

class LocalProvider implements ModelProvider {
  readonly name = "local";
  readonly capabilities = CAPABILITIES;

  /** Demo hook: fails the first N generate calls with a retryable error. */
  private failuresRemaining: number;

  constructor(failures = 0) {
    this.failuresRemaining = failures;
  }

  async generate(options: GenerateOptions): Promise<AIResponse> {
    // A real adapter maps SDK failures onto this taxonomy; the engine decides
    // whether to retry based on `retryable`. This one simulates a 503 twice.
    if (this.failuresRemaining > 0) {
      this.failuresRemaining -= 1;
      throw new ProviderServerError(this.name, "simulated upstream outage");
    }

    const prompt = options.messages.at(-1)?.content ?? "";
    if (prompt.length === 0) {
      // Never retry a caller error.
      throw new ProviderBadRequestError(this.name, "empty prompt");
    }

    const model = options.model ?? "local-1";
    const text = `Local echo (${model}): ${prompt}`;
    const promptTokens = countWords(prompt);
    const completionTokens = countWords(text);

    return {
      text,
      provider: this.name,
      model,
      message: { role: "assistant", content: text },
      finishReason: "stop",
      usage: { promptTokens, completionTokens, totalTokens: promptTokens + completionTokens },
      latencyMs: 0, // the engine stamps its own measured latency
      // Adapters attach the raw payload unless includeRaw: false was requested.
      ...(options.includeRaw === false ? {} : { raw: { simulated: true } }),
    };
  }

  async *stream(options: GenerateOptions): AsyncIterable<AIChunk> {
    const response = await this.generate(options);
    // Yield the text in word-sized deltas, then a terminal summary chunk.
    const words = response.text.split(/(?<=\s)/);
    for (const word of words) {
      yield { text: word, index: 0 }; // the engine re-stamps `index`
    }
    yield {
      text: "",
      index: 0,
      finishReason: "stop",
      usage: response.usage,
    };
  }

  async embed(text: string): Promise<number[]> {
    return embedLocally(text);
  }

  async embedMany(texts: string[]): Promise<number[][]> {
    return texts.map(embedLocally);
  }
}

// ---------------------------------------------------------------------------
// 2. Register it under a name, then use it exactly like any built-in provider.
// ---------------------------------------------------------------------------

// The factory receives engine config (apiKey accessor, baseUrl, …) — a real
// adapter validates those with a Zod schema before constructing its client.
registerProvider("local", () => new LocalProvider(2 /* fail twice, then succeed */));

const ai = new AI({
  provider: "local", // <-- same constructor shape as "openai" / "gemini"
  model: "local-1",
  temperature: 0.2,
  retries: 3,
});

ai.on("retry", (event) => {
  console.log(`[event] retry attempt=${event.attempt} in ${event.delayMs}ms`);
});
ai.on("request:success", (event) => {
  console.log(
    `[event] request:success latency=${event.latencyMs}ms tokens=${event.usage.totalTokens} chunks=${event.chunkCount}`,
  );
});

// ---------------------------------------------------------------------------
// 3. Generate — the engine retried the two simulated 503s automatically.
// ---------------------------------------------------------------------------

console.log("--- generate ---");
const response = await ai.generate("hello from a custom provider");
console.log(`provider : ${response.provider}`);
console.log(`model    : ${response.model}`);
console.log(`finish   : ${response.finishReason}`);
console.log(`usage    : ${JSON.stringify(response.usage)}`);
console.log(`raw      : ${JSON.stringify(response.raw)}`);
console.log(`text     : ${response.text}`);

// ---------------------------------------------------------------------------
// 4. Stream — identical async-iterable API as built-in providers.
// ---------------------------------------------------------------------------

console.log("\n--- stream ---");
const stream = await ai.stream("streaming works too");
let assembled = "";
for await (const chunk of stream) {
  assembled += chunk.text;
  process.stdout.write(chunk.text);
}
process.stdout.write("\n");
console.log(`assembled length: ${assembled.length}`);

// ---------------------------------------------------------------------------
// 5. Embed — capability-gated: declaring `embeddings: false` would make
//    ai.embed() throw UnsupportedCapabilityError before any network I/O.
// ---------------------------------------------------------------------------

console.log("\n--- embed ---");
const [first, second] = await Promise.all([
  ai.embed("vector search"),
  ai.embed("vector search"),
]);
console.log(`dimensions      : ${first.length}`);
console.log(`deterministic   : ${JSON.stringify(first) === JSON.stringify(second)}`);

const batch = await ai.embedMany(["alpha", "beta", "gamma"]);
console.log(`batch vectors   : ${batch.length}`);

// ---------------------------------------------------------------------------
// 6. Errors use the same taxonomy as every other provider.
// ---------------------------------------------------------------------------

console.log("\n--- errors ---");

// An unknown provider name fails fast with actionable guidance.
try {
  await new AI({ provider: "definitely-not-registered", model: "m" }).generate("x");
} catch (error) {
  console.log(`unknown provider: ${(error as Error).message.split("\n")[0]}`);
}

// Validation happens before any provider is even constructed.
try {
  await ai.generate("");
} catch (error) {
  console.log(`empty prompt    : ${(error as Error).message}`);
}

console.log(`\nstats: ${JSON.stringify(ai.stats())}`);
