import type {
  AIChunk,
  AIResponse,
  EmbedOptions,
  GenerateOptions,
  ModelProvider,
  ProviderCapabilities,
} from "@murphychain/core";

/** Every capability enabled (capability-gate tests override individual flags). */
export const ALL_CAPABILITIES: ProviderCapabilities = Object.freeze({
  generate: true,
  streaming: true,
  toolCalling: true,
  embeddings: true,
  systemMessage: true,
  structuredOutput: false,
});

/** Builds a plausible AIResponse for tests. */
export function makeResponse(overrides: Partial<AIResponse> = {}): AIResponse {
  return {
    text: "Hello!",
    provider: "fake",
    model: "fake-1",
    message: { role: "assistant", content: "Hello!" },
    finishReason: "stop",
    usage: { promptTokens: 3, completionTokens: 2, totalTokens: 5 },
    latencyMs: 0,
    ...overrides,
  };
}

/** Default three-chunk stream: two text chunks plus a terminal summary chunk. */
export async function* defaultStream(): AsyncGenerator<AIChunk> {
  yield { text: "Hel", index: 0 };
  yield { text: "lo", index: 1 };
  yield {
    text: "",
    index: 0,
    finishReason: "stop",
    usage: { promptTokens: 1, completionTokens: 1, totalTokens: 2 },
  };
}

export interface FakeProviderOptions {
  name?: string;
  capabilities?: Partial<ProviderCapabilities>;
  generateImpl?: (options: GenerateOptions) => Promise<AIResponse>;
  streamImpl?: (options: GenerateOptions) => AsyncIterable<AIChunk>;
  embedImpl?: (text: string, options?: EmbedOptions) => Promise<number[]>;
}

/**
 * In-memory ModelProvider used by engine tests. Records every call so tests
 * can assert what the engine handed to the provider (options, signal, …).
 */
export class FakeProvider implements ModelProvider {
  readonly name: string;
  readonly capabilities: ProviderCapabilities;
  readonly generateCalls: GenerateOptions[] = [];
  readonly streamCalls: GenerateOptions[] = [];
  readonly embedCalls: Array<{ text: string; options?: EmbedOptions | undefined }> = [];

  private readonly options: FakeProviderOptions;

  constructor(options: FakeProviderOptions = {}) {
    this.options = options;
    this.name = options.name ?? "fake";
    this.capabilities = { ...ALL_CAPABILITIES, ...options.capabilities };
  }

  async generate(options: GenerateOptions): Promise<AIResponse> {
    this.generateCalls.push(options);
    if (this.options.generateImpl !== undefined) return this.options.generateImpl(options);
    const response = makeResponse({ provider: this.name, model: options.model ?? "fake-1" });
    // Real adapters attach the raw provider payload unless asked not to.
    if (options.includeRaw !== false) response.raw = { echoed: true };
    return response;
  }

  stream(options: GenerateOptions): AsyncIterable<AIChunk> {
    this.streamCalls.push(options);
    return this.options.streamImpl !== undefined
      ? this.options.streamImpl(options)
      : defaultStream();
  }

  async embed(text: string, options?: EmbedOptions): Promise<number[]> {
    this.embedCalls.push({ text, options });
    if (this.options.embedImpl !== undefined) return this.options.embedImpl(text, options);
    return [0.1, 0.2];
  }

  async embedMany(texts: string[], options?: EmbedOptions): Promise<number[][]> {
    const out: number[][] = [];
    for (const text of texts) out.push(await this.embed(text, options));
    return out;
  }
}

/** Fast deterministic retry policy for tests (no jitter, ~1ms delays). */
export const FAST_RETRY = {
  retries: 2,
  initialDelayMs: 1,
  maxDelayMs: 5,
  backoffFactor: 2,
  jitter: false,
} as const;
