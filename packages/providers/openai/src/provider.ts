import OpenAI from "openai";
import type { CreateEmbeddingResponse } from "openai/resources/embeddings";
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
import type { OpenAIProviderConfig } from "./config";
import { mapOpenAIError } from "./errors";
import {
  ToolCallAccumulator,
  buildAIResponse,
  buildNonStreamingParams,
  buildStreamingParams,
  mapFinishReason,
  mapUsage,
} from "./mapping";

const PROVIDER = "openai";

/** Bounded per-key client cache (credential rotation must not leak memory). */
const MAX_CACHED_CLIENTS = 8;

/** Capabilities of the OpenAI Chat Completions API. */
const CAPABILITIES: ProviderCapabilities = Object.freeze({
  generate: true,
  streaming: true,
  toolCalling: true,
  embeddings: true,
  systemMessage: true,
  // JSON mode / structured outputs land with the 0.2.0 structured-output work.
  structuredOutput: false,
});

/**
 * OpenAI provider adapter.
 *
 * Wraps the official `openai` SDK. SDK-internal retries are disabled
 * (`maxRetries: 0`) because the MurphyChain engine owns the retry policy.
 */
export class OpenAIProvider implements ModelProvider {
  readonly name = PROVIDER;
  readonly capabilities = CAPABILITIES;

  private readonly config: OpenAIProviderConfig;
  private readonly clients = new Map<string, OpenAI>();

  constructor(config: OpenAIProviderConfig) {
    this.config = config;
  }

  // -------------------------------------------------------------------------
  // Generation
  // -------------------------------------------------------------------------

  async generate(options: GenerateOptions): Promise<AIResponse> {
    const model = this.requireModel(options.model);
    const params = buildNonStreamingParams(options, model);
    const includeRaw = options.includeRaw !== false;
    let completion;
    try {
      completion = await this.client().chat.completions.create(params, {
        ...(options.signal !== undefined ? { signal: options.signal } : {}),
      });
    } catch (error) {
      throw mapOpenAIError(error);
    }
    return buildAIResponse(completion, model, includeRaw, completion);
  }

  async *stream(options: GenerateOptions): AsyncIterable<AIChunk> {
    const model = this.requireModel(options.model);
    const params = buildStreamingParams(options, model);
    let stream;
    try {
      stream = await this.client().chat.completions.create(params, {
        ...(options.signal !== undefined ? { signal: options.signal } : {}),
      });
    } catch (error) {
      throw mapOpenAIError(error);
    }

    const accumulator = new ToolCallAccumulator();
    let finishReason: FinishReason | undefined;
    let usage: TokenUsage | undefined;
    try {
      for await (const chunk of stream) {
        if (chunk.usage != null) usage = mapUsage(chunk.usage);
        const choice = chunk.choices[0];
        if (choice === undefined) continue; // usage-only tail chunk.
        if (choice.finish_reason != null) finishReason = mapFinishReason(choice.finish_reason);
        if (choice.delta.tool_calls !== undefined) accumulator.add(choice.delta.tool_calls);
        const text = choice.delta.content;
        if (text !== undefined && text !== null && text.length > 0) {
          // The engine stamps `index` on every chunk.
          yield { text, index: 0 };
        }
      }
    } catch (error) {
      throw mapOpenAIError(error);
    }
    // Terminal chunk: finish reason, usage, and any accumulated tool calls.
    yield {
      text: "",
      index: 0,
      finishReason: finishReason ?? "unknown",
      usage,
      toolCalls: accumulator.toToolCalls(),
    };
  }

  // -------------------------------------------------------------------------
  // Embeddings
  // -------------------------------------------------------------------------

  async embed(text: string, options: EmbedOptions = {}): Promise<number[]> {
    const [first] = await this.embedBatch([text], options);
    /* c8 ignore next -- embedBatch always returns one vector per input */
    if (first === undefined) {
      throw new ProviderError(PROVIDER, "OpenAI returned no embedding.", {
        code: "PROVIDER_MALFORMED_RESPONSE",
        retryable: false,
      });
    }
    return first;
  }

  async embedMany(texts: string[], options: EmbedOptions = {}): Promise<number[][]> {
    return this.embedBatch(texts, options);
  }

  private async embedBatch(texts: string[], options: EmbedOptions): Promise<number[][]> {
    let response: CreateEmbeddingResponse;
    try {
      response = await this.client().embeddings.create(
        { model: this.config.embeddingModel, input: texts },
        { ...(options.signal !== undefined ? { signal: options.signal } : {}) },
      );
    } catch (error) {
      throw mapOpenAIError(error);
    }
    const sorted = [...response.data].sort((a, b) => a.index - b.index);
    if (sorted.length !== texts.length) {
      throw new ProviderError(
        PROVIDER,
        `Embedding count mismatch: requested ${texts.length}, received ${sorted.length}.`,
        { code: "PROVIDER_MALFORMED_RESPONSE", retryable: false },
      );
    }
    return sorted.map((entry) => entry.embedding);
  }

  // -------------------------------------------------------------------------
  // Internals
  // -------------------------------------------------------------------------

  private requireModel(model: string | undefined): string {
    const resolved = model ?? this.config.defaultModel;
    if (resolved === undefined || resolved.length === 0) {
      throw new ConfigurationError(
        `OpenAI: no model configured. Pass "model" to new AI({ model: "gpt-4.1" }) or per request.`,
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
  private client(): OpenAI {
    const apiKey = this.resolveApiKey();
    const existing = this.clients.get(apiKey);
    if (existing !== undefined) return existing;
    const client = new OpenAI({
      apiKey,
      maxRetries: 0, // The engine owns retry policy — never stack SDK retries.
      ...(this.config.baseUrl !== undefined ? { baseURL: this.config.baseUrl } : {}),
      ...(this.config.organization !== undefined ? { organization: this.config.organization } : {}),
      ...(this.config.project !== undefined ? { project: this.config.project } : {}),
      ...(this.config.defaultHeaders !== undefined
        ? { defaultHeaders: this.config.defaultHeaders }
        : {}),
      ...(this.config.fetch !== undefined ? { fetch: this.config.fetch } : {}),
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
   * `OPENAI_API_KEY` environment variable.
   */
  private resolveApiKey(): string {
    const key = this.config.apiKey?.() ?? this.config.fallbackApiKey ?? process.env.OPENAI_API_KEY;
    if (key === undefined || key.length === 0) {
      throw new ConfigurationError(
        "OpenAI API key is required. Pass apiKey to new AI({ apiKey }) or set OPENAI_API_KEY.",
        { provider: PROVIDER },
      );
    }
    return key;
  }
}
