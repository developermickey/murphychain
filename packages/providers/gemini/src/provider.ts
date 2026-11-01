import { GoogleGenAI } from "@google/genai";
import {
  ConfigurationError,
  ProviderError,
  type AIChunk,
  type AIResponse,
  type EmbedOptions,
  type FinishReason,
  type GenerateOptions,
  type ModelProvider,
  type ProviderCapabilities,
  type TokenUsage,
} from "@murphychain/core";
import type { GeminiProviderConfig } from "./config";
import { mapGeminiError } from "./errors";
import {
  assertUsableResponse,
  buildConfig,
  extractText,
  extractToolCalls,
  firstCandidate,
  mapFinishReason,
  mapUsage,
  toGeminiContents,
} from "./mapping";

const PROVIDER = "gemini";

/** Bounded per-key client cache (credential rotation must not leak memory). */
const MAX_CACHED_CLIENTS = 8;

/** Max parallel embedContent calls in embedMany (Gemini has no batch API). */
const EMBED_CONCURRENCY = 8;

/** Capabilities of the Gemini API. */
const CAPABILITIES: ProviderCapabilities = Object.freeze({
  generate: true,
  streaming: true,
  toolCalling: true,
  embeddings: true,
  systemMessage: true,
  // Response-schema / structured output lands with the 0.2.0 work.
  structuredOutput: false,
});

/**
 * Google Gemini provider adapter.
 *
 * Wraps the official `@google/genai` SDK. SDK-internal retries are disabled
 * (`retryOptions.attempts: 1` — the SDK defaults to 5) because the
 * MurphyChain engine owns the retry policy.
 */
export class GeminiProvider implements ModelProvider {
  readonly name = PROVIDER;
  readonly capabilities = CAPABILITIES;

  private readonly config: GeminiProviderConfig;
  private readonly clients = new Map<string, GoogleGenAI>();

  constructor(config: GeminiProviderConfig) {
    this.config = config;
  }

  // -------------------------------------------------------------------------
  // Generation
  // -------------------------------------------------------------------------

  async generate(options: GenerateOptions): Promise<AIResponse> {
    const model = this.requireModel(options.model);
    const { contents, systemInstruction } = toGeminiContents(options.messages);
    const config = buildConfig(options, systemInstruction);
    let response;
    try {
      response = await this.client().models.generateContent({ model, contents, config });
    } catch (error) {
      throw mapGeminiError(error);
    }
    assertUsableResponse(response);
    const candidate = firstCandidate(response);
    /* c8 ignore next -- assertUsableResponse guarantees a candidate */
    if (candidate === undefined) {
      throw new ProviderError(PROVIDER, "Gemini returned no candidate.", {
        code: "PROVIDER_MALFORMED_RESPONSE",
        retryable: false,
      });
    }
    const text = extractText(candidate.content?.parts);
    const toolCalls = extractToolCalls(candidate.content?.parts, "gemini_call");
    const response2: AIResponse = {
      text,
      provider: PROVIDER,
      model: response.modelVersion ?? model,
      message: {
        role: "assistant",
        content: text,
        ...(toolCalls !== undefined ? { toolCalls } : {}),
      },
      finishReason: mapFinishReason(candidate.finishReason),
      usage: mapUsage(response),
      latencyMs: 0, // stamped by the engine.
    };
    if (options.includeRaw !== false) response2.raw = response;
    return response2;
  }

  async *stream(options: GenerateOptions): AsyncIterable<AIChunk> {
    const model = this.requireModel(options.model);
    const { contents, systemInstruction } = toGeminiContents(options.messages);
    const config = buildConfig(options, systemInstruction);
    let iterator;
    try {
      iterator = await this.client().models.generateContentStream({ model, contents, config });
    } catch (error) {
      throw mapGeminiError(error);
    }

    let finishReason: FinishReason | undefined;
    let usage: TokenUsage | undefined;
    let chunkIndex = 0;
    try {
      for await (const chunk of iterator) {
        assertUsableResponse(chunk);
        const candidate = firstCandidate(chunk);
        if (candidate === undefined) continue;
        if (candidate.finishReason !== undefined) {
          finishReason = mapFinishReason(candidate.finishReason);
        }
        if (chunk.usageMetadata !== undefined) usage = mapUsage(chunk);
        const text = extractText(candidate.content?.parts);
        if (text.length > 0) {
          // The engine stamps `index` on every chunk.
          yield { text, index: 0 };
        }
        const toolCalls = extractToolCalls(
          candidate.content?.parts,
          `gemini_call_${chunkIndex}`,
        );
        if (toolCalls !== undefined) {
          yield {
            text: "",
            index: 0,
            toolCalls,
            ...(finishReason !== undefined ? { finishReason } : {}),
            ...(usage !== undefined ? { usage } : {}),
          };
        }
        chunkIndex += 1;
      }
    } catch (error) {
      throw mapGeminiError(error);
    }
    // Terminal chunk: finish reason and usage (tool calls ride earlier chunks).
    yield {
      text: "",
      index: 0,
      finishReason: finishReason ?? "unknown",
      usage,
    };
  }

  // -------------------------------------------------------------------------
  // Embeddings
  // -------------------------------------------------------------------------

  async embed(text: string, options: EmbedOptions = {}): Promise<number[]> {
    const [first] = await this.embedBatch([text], options);
    /* c8 ignore next -- embedBatch always returns one vector per input */
    if (first === undefined) {
      throw new ProviderError(PROVIDER, "Gemini returned no embedding.", {
        code: "PROVIDER_MALFORMED_RESPONSE",
        retryable: false,
      });
    }
    return first;
  }

  async embedMany(texts: string[], options: EmbedOptions = {}): Promise<number[][]> {
    return this.embedBatch(texts, options);
  }

  /**
   * Gemini 2.32 exposes no batch embedding endpoint, so inputs are embedded
   * through `embedContent` with bounded concurrency; results keep input order.
   */
  private async embedBatch(texts: string[], options: EmbedOptions): Promise<number[][]> {
    const results: number[][] = new Array(texts.length);
    let cursor = 0;
    const worker = async (): Promise<void> => {
      for (;;) {
        const index = cursor;
        cursor += 1;
        if (index >= texts.length) return;
        const text = texts[index];
        /* c8 ignore next -- index is always in range here */
        if (text === undefined) continue;
        try {
          const response = await this.client().models.embedContent({
            model: this.config.embeddingModel,
            contents: text,
            ...(options.signal !== undefined
              ? { config: { abortSignal: options.signal } }
              : {}),
          });
          const values = response.embeddings?.[0]?.values;
          if (values === undefined || values.length === 0) {
            throw new ProviderError(PROVIDER, "Gemini returned an empty embedding.", {
              code: "PROVIDER_MALFORMED_RESPONSE",
              retryable: false,
            });
          }
          results[index] = values;
        } catch (error) {
          throw mapGeminiError(error);
        }
      }
    };
    const workers = Array.from(
      { length: Math.min(EMBED_CONCURRENCY, Math.max(texts.length, 1)) },
      () => worker(),
    );
    await Promise.all(workers);
    return results;
  }

  // -------------------------------------------------------------------------
  // Internals
  // -------------------------------------------------------------------------

  private requireModel(model: string | undefined): string {
    const resolved = model ?? this.config.defaultModel;
    if (resolved === undefined || resolved.length === 0) {
      throw new ConfigurationError(
        `Gemini: no model configured. Pass "model" to new AI({ model: "gemini-2.5-flash" }) or per request.`,
        { provider: PROVIDER },
      );
    }
    return resolved;
  }

  /**
   * Returns the client for the currently resolved API key. Clients are cached
   * per key so that `apiKey: () => rotate()` reuses connections while a key is
   * active; the cache is bounded to bound memory under rotation.
   */
  private client(): GoogleGenAI {
    const apiKey = this.resolveApiKey();
    const existing = this.clients.get(apiKey);
    if (existing !== undefined) return existing;
    const client = new GoogleGenAI({
      apiKey,
      httpOptions: {
        // The engine owns retry policy — never stack SDK retries (default: 5).
        retryOptions: { attempts: 1 },
        ...(this.config.baseUrl !== undefined ? { baseUrl: this.config.baseUrl } : {}),
        ...(this.config.apiVersion !== undefined ? { apiVersion: this.config.apiVersion } : {}),
        ...(this.config.headers !== undefined ? { headers: this.config.headers } : {}),
        ...(this.config.fetch !== undefined ? { fetch: this.config.fetch } : {}),
      },
    });
    this.clients.set(apiKey, client);
    if (this.clients.size > MAX_CACHED_CLIENTS) {
      const oldest = this.clients.keys().next().value;
      if (oldest !== undefined) this.clients.delete(oldest);
    }
    return client;
  }

  /**
   * Resolves the API key per request: accessor (rotation) → static fallback →
   * `GEMINI_API_KEY` / `GOOGLE_API_KEY` environment variables.
   */
  private resolveApiKey(): string {
    const key =
      this.config.apiKey?.() ??
      this.config.fallbackApiKey ??
      process.env.GEMINI_API_KEY ??
      process.env.GOOGLE_API_KEY;
    if (key === undefined || key.length === 0) {
      throw new ConfigurationError(
        "Gemini API key is required. Pass apiKey to new AI({ apiKey }) or set GEMINI_API_KEY.",
        { provider: PROVIDER },
      );
    }
    return key;
  }
}
