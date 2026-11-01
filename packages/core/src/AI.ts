import { z } from "zod";
import { ExecutionContext } from "./Context";
import { EventBus } from "./Events";
import type { AIEventListener, AIEventMap, AIEventName } from "./Events";
import {
  ConfigurationError,
  ensureMurphyChainError,
  UnsupportedCapabilityError,
  ValidationError,
} from "./Errors";
import type { MurphyChainError } from "./Errors";
import { Model } from "./Model";
import type { GenerateOptions, ModelProvider, ProviderFactoryConfig } from "./provider";
import { listProviders, resolveProviderFactory } from "./registry";
import { withRetry, type RetryAttemptInfo, type RetryPolicy } from "./retry";
import {
  ZERO_USAGE,
  type AIChunk,
  type AIResponse,
  type AIStats,
  type FinishReason,
  type Message,
  type NormalizedTool,
  type TokenUsage,
  type ToolChoice,
  type ToolDefinition,
} from "./types";
import {
  assertInputLimits,
  assertMetadata,
  assertNonNegativeNumber,
  DEFAULT_MAX_INPUT_CHARS,
  DEFAULT_MAX_MESSAGES,
  formatIssues,
  generateRequestSchema,
  messageSchema,
  normalizeRetryPolicy,
  normalizeTool,
  parseOrThrow,
  retryPolicySchema,
} from "./validation";

// ---------------------------------------------------------------------------
// Public configuration and call types
// ---------------------------------------------------------------------------

/** Credential source: a literal key or an accessor re-read per request. */
type ApiKeySource = string | (() => string | undefined);

export interface AIConfig {
  /** Registered provider name (e.g. `"openai"`) or a provider instance. */
  provider: string | ModelProvider;
  /** Default model name (or a {@link Model} with default parameters). */
  model?: string | Model | undefined;
  /** API key. A function is re-evaluated per request (credential rotation). */
  apiKey?: ApiKeySource | undefined;
  /** API endpoint override (http/https URL). */
  baseUrl?: string | undefined;
  temperature?: number | undefined;
  maxTokens?: number | undefined;
  topP?: number | undefined;
  stop?: string[] | undefined;
  /**
   * Deadline in ms for `generate`; time-to-first-chunk deadline for
   * `stream`. `0` disables it. Default: 120_000.
   */
  timeoutMs?: number | undefined;
  /** Retry count (number) or full policy override. Default: 2 retries. */
  retries?: number | Partial<RetryPolicy> | undefined;
  /** Maximum messages per request. Default: 10_000. */
  maxMessages?: number | undefined;
  /** Maximum total characters per request. Default: 1_000_000. */
  maxInputChars?: number | undefined;
  /** Attach raw provider payloads to responses. Default: `true`. */
  includeRaw?: boolean | undefined;
  /** System message prepended to every request. */
  systemPrompt?: string | undefined;
  /** Provider-specific options, validated by the provider. */
  providerOptions?: Record<string, unknown> | undefined;
  /** Context propagated to `request:start` events (never log secrets here). */
  metadata?: Record<string, unknown> | undefined;
}

/** Per-call execution controls (not model parameters). */
export interface CallOptions {
  signal?: AbortSignal | undefined;
  timeoutMs?: number | undefined;
  metadata?: Record<string, unknown> | undefined;
  includeRaw?: boolean | undefined;
}

/** Structured form of a generation request. Exactly one of messages/prompt. */
export interface GenerateRequest {
  messages?: Message[] | undefined;
  prompt?: string | undefined;
  system?: string | undefined;
  model?: string | Model | undefined;
  temperature?: number | undefined;
  maxTokens?: number | undefined;
  topP?: number | undefined;
  stop?: string[] | undefined;
  tools?: ToolDefinition[] | undefined;
  toolChoice?: ToolChoice | undefined;
}

/** Accepted inputs of `ai.generate()` / `ai.stream()`. */
export type GenerateInput = string | Message[] | GenerateRequest;

// ---------------------------------------------------------------------------
// Schemas
// ---------------------------------------------------------------------------

const apiKeySchema = z.union([
  z.string().min(1),
  z.custom<ApiKeySource>((value) => typeof value === "function", {
    message: "apiKey must be a non-empty string or a function returning a string",
  }),
]);

const aiConfigSchema = z
  .object({
    provider: z.unknown(),
    model: z.union([z.string().min(1), z.instanceof(Model)]).optional(),
    apiKey: apiKeySchema.optional(),
    baseUrl: z.string().min(1).optional(),
    temperature: z.number().finite().min(0).optional(),
    maxTokens: z.number().int().positive().optional(),
    topP: z.number().min(0).max(1).optional(),
    stop: z.array(z.string().min(1).max(4096)).max(16).optional(),
    timeoutMs: z.number().int().min(0).max(86_400_000).default(120_000),
    retries: z
      .union([z.number().int().min(0).max(20), retryPolicySchema.partial().strict()])
      .optional(),
    maxMessages: z.number().int().min(1).max(1_000_000).default(DEFAULT_MAX_MESSAGES),
    maxInputChars: z.number().int().min(1).max(100_000_000).default(DEFAULT_MAX_INPUT_CHARS),
    includeRaw: z.boolean().default(true),
    systemPrompt: z.string().optional(),
    providerOptions: z.record(z.string(), z.unknown()).default({}),
    metadata: z.record(z.string(), z.unknown()).default({}),
  })
  .strict();

interface NormalizedConfig {
  provider: string | ModelProvider;
  model: string | Model | undefined;
  apiKey: ApiKeySource | undefined;
  baseUrl: string | undefined;
  temperature: number | undefined;
  maxTokens: number | undefined;
  topP: number | undefined;
  stop: string[] | undefined;
  timeoutMs: number;
  retries: RetryPolicy;
  maxMessages: number;
  maxInputChars: number;
  includeRaw: boolean;
  systemPrompt: string | undefined;
  providerOptions: Record<string, unknown>;
  metadata: Record<string, unknown>;
}

/** Validated per-call options. */
interface NormalizedCallOptions {
  signal: AbortSignal | undefined;
  timeoutMs: number | undefined;
  metadata: Record<string, unknown> | undefined;
  includeRaw: boolean | undefined;
}

interface ResolvedRequest {
  messages: Message[];
  model: string;
  temperature: number | undefined;
  maxTokens: number | undefined;
  topP: number | undefined;
  stop: string[] | undefined;
  toolChoice: ToolChoice | undefined;
  tools: NormalizedTool[] | undefined;
  totalChars: number;
  includeRaw: boolean;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function assertProvider(value: unknown): string | ModelProvider {
  if (typeof value === "string") {
    if (value.trim().length === 0) {
      throw new ConfigurationError("provider must be a non-empty string or a ModelProvider.");
    }
    return value.trim().toLowerCase();
  }
  if (typeof value === "object" && value !== null) {
    const candidate = value as Partial<ModelProvider>;
    if (
      typeof candidate.name === "string" &&
      typeof candidate.generate === "function" &&
      typeof candidate.stream === "function" &&
      typeof candidate.embed === "function" &&
      typeof candidate.embedMany === "function" &&
      typeof candidate.capabilities === "object" &&
      candidate.capabilities !== null
    ) {
      return value as ModelProvider;
    }
    throw new ConfigurationError(
      "provider object must implement the ModelProvider interface " +
        "(name, capabilities, generate, stream, embed, embedMany).",
    );
  }
  throw new ConfigurationError("provider must be a registered name or a ModelProvider.");
}

function assertHttpUrl(value: string, field: string): void {
  let url: URL;
  try {
    url = new URL(value);
  } catch {
    throw new ConfigurationError(`${field} must be a valid URL, got "${value}".`, { field });
  }
  if (url.protocol !== "http:" && url.protocol !== "https:") {
    throw new ConfigurationError(`${field} must use http(s), got "${url.protocol}".`, { field });
  }
}

// ---------------------------------------------------------------------------
// AI
// ---------------------------------------------------------------------------

/**
 * The primary MurphyChain entry point.
 *
 * ```ts
 * const ai = new AI({ provider: "openai", model: "gpt-4.1", apiKey: process.env.OPENAI_API_KEY });
 * const response = await ai.generate("Explain artificial intelligence.");
 * console.log(response.text);
 * ```
 */
export class AI {
  private readonly config: NormalizedConfig;
  private readonly bus = new EventBus<AIEventMap>();
  private readonly counters: AIStats = {
    requests: 0,
    errors: 0,
    retries: 0,
    chunks: 0,
    promptTokens: 0,
    completionTokens: 0,
    totalTokens: 0,
    totalLatencyMs: 0,
  };
  private cachedProvider: ModelProvider | undefined;
  private pendingProvider: Promise<ModelProvider> | undefined;

  constructor(config: AIConfig) {
    if (typeof config !== "object" || config === null) {
      throw new ConfigurationError("AI requires a configuration object.");
    }
    const parsed = aiConfigSchema.safeParse(config);
    if (!parsed.success) {
      throw new ConfigurationError("Invalid AI configuration.", {
        issues: formatIssues(parsed.error),
      });
    }
    const data = parsed.data;
    if (data.baseUrl !== undefined) assertHttpUrl(data.baseUrl, "baseUrl");

    this.config = {
      provider: assertProvider(data.provider),
      model: data.model,
      apiKey: data.apiKey,
      baseUrl: data.baseUrl,
      temperature: data.temperature,
      maxTokens: data.maxTokens,
      topP: data.topP,
      stop: data.stop,
      timeoutMs: data.timeoutMs,
      retries: normalizeRetryPolicy(data.retries),
      maxMessages: data.maxMessages,
      maxInputChars: data.maxInputChars,
      includeRaw: data.includeRaw,
      systemPrompt: data.systemPrompt,
      providerOptions: data.providerOptions,
      metadata: data.metadata,
    };
  }

  /** The provider name this instance is bound to. */
  get providerName(): string {
    return typeof this.config.provider === "string"
      ? this.config.provider
      : this.config.provider.name;
  }

  // -------------------------------------------------------------------------
  // Events
  // -------------------------------------------------------------------------

  /** Subscribes to an engine event. Returns an unsubscribe function. */
  on<K extends AIEventName>(event: K, listener: AIEventListener<K>): () => void {
    return this.bus.on(event, listener);
  }

  /** Subscribes to the next occurrence of an event. */
  once<K extends AIEventName>(event: K, listener: AIEventListener<K>): () => void {
    return this.bus.once(event, listener);
  }

  /** Removes a listener registered with {@link on}. */
  off<K extends AIEventName>(event: K, listener: AIEventListener<K>): void {
    this.bus.off(event, listener);
  }

  // -------------------------------------------------------------------------
  // Stats
  // -------------------------------------------------------------------------

  /** Snapshot of cumulative counters (frozen copy). */
  stats(): AIStats {
    return Object.freeze({ ...this.counters });
  }

  /** Resets cumulative counters to zero. */
  resetStats(): void {
    for (const key of Object.keys(this.counters) as (keyof AIStats)[]) {
      this.counters[key] = 0;
    }
  }

  // -------------------------------------------------------------------------
  // Generation
  // -------------------------------------------------------------------------

  /** Runs a non-streaming generation. */
  async generate(input: GenerateInput, options: CallOptions = {}): Promise<AIResponse> {
    const call = this.normalizeCallOptions(options);
    const request = this.resolveRequest(input, call);
    const provider = await this.getProvider();
    this.assertCapabilities(provider, request, false);
    const ctx = this.createContext(provider.name, request.model, call);
    const genOptions = this.buildGenerateOptions(request, ctx);
    this.counters.requests += 1;
    this.bus.emit("request:start", {
      requestId: ctx.requestId,
      provider: provider.name,
      model: request.model,
      messageCount: request.messages.length,
      totalChars: request.totalChars,
      metadata: ctx.metadata,
    });
    try {
      const result = await withRetry(() => provider.generate(genOptions), this.config.retries, {
        signal: ctx.signal,
        onRetry: (info) => this.emitRetry(ctx, info),
        createAbortError: () => ctx.abortError(),
      });
      const response: AIResponse = {
        ...result,
        provider: provider.name,
        model: result.model.length > 0 ? result.model : request.model,
        latencyMs: ctx.elapsedMs,
      };
      this.recordSuccess(ctx, response.usage, response.finishReason, 1);
      return response;
    } catch (error) {
      const mapped = this.mapCallError(error, ctx);
      this.counters.errors += 1;
      if (ctx.reason !== undefined) {
        this.bus.emit("abort", { requestId: ctx.requestId, reason: ctx.reason });
      }
      this.bus.emit("request:error", {
        requestId: ctx.requestId,
        provider: provider.name,
        model: request.model,
        latencyMs: ctx.elapsedMs,
        error: mapped,
      });
      throw mapped;
    } finally {
      ctx.dispose();
    }
  }

  /**
   * Starts a streaming generation. Returns a single-use async iterable.
   * Retry (if needed) applies only until the first chunk is emitted; after
   * that, failures propagate immediately (half a stream is never replayed).
   */
  async stream(input: GenerateInput, options: CallOptions = {}): Promise<AsyncIterable<AIChunk>> {
    const call = this.normalizeCallOptions(options);
    const request = this.resolveRequest(input, call);
    const provider = await this.getProvider();
    this.assertCapabilities(provider, request, true);
    const ctx = this.createContext(provider.name, request.model, call);
    const genOptions = this.buildGenerateOptions(request, ctx);
    this.counters.requests += 1;
    this.bus.emit("request:start", {
      requestId: ctx.requestId,
      provider: provider.name,
      model: request.model,
      messageCount: request.messages.length,
      totalChars: request.totalChars,
      metadata: ctx.metadata,
    });
    return this.iterateStream(ctx, provider, genOptions, request);
  }

  // -------------------------------------------------------------------------
  // Embeddings
  // -------------------------------------------------------------------------

  /** Embeds a single text. */
  async embed(text: string, options: CallOptions = {}): Promise<number[]> {
    const call = this.normalizeCallOptions(options);
    if (typeof text !== "string" || text.length === 0) {
      throw new ValidationError("embed() expects a non-empty string.");
    }
    if (text.length > this.config.maxInputChars) {
      throw new ValidationError(
        `Embedding input exceeds the configured limit of ${this.config.maxInputChars} characters.`,
        { limit: this.config.maxInputChars },
      );
    }
    const provider = await this.getProvider();
    if (!provider.capabilities.embeddings) {
      throw new UnsupportedCapabilityError(provider.name, "embeddings");
    }
    const ctx = this.createContext(provider.name, "(embeddings)", call);
    this.counters.requests += 1;
    try {
      return await withRetry(
        () => provider.embed(text, { signal: ctx.signal }),
        this.config.retries,
        {
          signal: ctx.signal,
          onRetry: () => {
            this.counters.retries += 1;
          },
          createAbortError: () => ctx.abortError(),
        },
      );
    } catch (error) {
      this.counters.errors += 1;
      throw this.mapCallError(error, ctx);
    } finally {
      ctx.dispose();
    }
  }

  /** Embeds a batch of texts (bounded: at most 1,000 per call). */
  async embedMany(texts: string[], options: CallOptions = {}): Promise<number[][]> {
    const call = this.normalizeCallOptions(options);
    if (!Array.isArray(texts) || texts.length === 0) {
      throw new ValidationError("embedMany() expects a non-empty array of strings.");
    }
    if (texts.length > 1_000) {
      throw new ValidationError("embedMany() accepts at most 1000 texts per call.", {
        count: texts.length,
        limit: 1_000,
      });
    }
    for (let i = 0; i < texts.length; i++) {
      const text = texts[i];
      if (typeof text !== "string" || text.length === 0) {
        throw new ValidationError(`embedMany(): item ${i} must be a non-empty string.`, {
          index: i,
        });
      }
      if (text.length > this.config.maxInputChars) {
        throw new ValidationError(
          `embedMany(): item ${i} exceeds the configured limit of ${this.config.maxInputChars} characters.`,
          { index: i, limit: this.config.maxInputChars },
        );
      }
    }
    const provider = await this.getProvider();
    if (!provider.capabilities.embeddings) {
      throw new UnsupportedCapabilityError(provider.name, "embeddings");
    }
    const ctx = this.createContext(provider.name, "(embeddings)", call);
    this.counters.requests += 1;
    try {
      return await withRetry(
        () => provider.embedMany(texts, { signal: ctx.signal }),
        this.config.retries,
        {
          signal: ctx.signal,
          onRetry: () => {
            this.counters.retries += 1;
          },
          createAbortError: () => ctx.abortError(),
        },
      );
    } catch (error) {
      this.counters.errors += 1;
      throw this.mapCallError(error, ctx);
    } finally {
      ctx.dispose();
    }
  }

  // -------------------------------------------------------------------------
  // Internals
  // -------------------------------------------------------------------------

  /** Validates per-call options exactly once. */
  private normalizeCallOptions(options: CallOptions): NormalizedCallOptions {
    if (typeof options !== "object" || options === null) {
      throw new ValidationError("Call options must be an object.");
    }
    const timeoutMs =
      options.timeoutMs !== undefined
        ? assertNonNegativeNumber(options.timeoutMs, "timeoutMs")
        : undefined;
    const metadata =
      options.metadata !== undefined ? assertMetadata(options.metadata, "metadata") : undefined;
    let includeRaw: boolean | undefined;
    if (options.includeRaw !== undefined) {
      if (typeof options.includeRaw !== "boolean") {
        throw new ValidationError('"includeRaw" must be a boolean.', { field: "includeRaw" });
      }
      includeRaw = options.includeRaw;
    }
    let signal: AbortSignal | undefined;
    if (options.signal !== undefined) {
      const candidate: unknown = options.signal;
      if (
        typeof candidate !== "object" ||
        candidate === null ||
        typeof (candidate as AbortSignal).addEventListener !== "function"
      ) {
        throw new ValidationError('"signal" must be an AbortSignal.', { field: "signal" });
      }
      signal = options.signal;
    }
    return { signal, timeoutMs, metadata, includeRaw };
  }

  /** Resolves (and caches) the provider: instance, registry, or lazy loader. */
  private async getProvider(): Promise<ModelProvider> {
    const configured = this.config.provider;
    if (typeof configured !== "string") return configured;
    if (this.cachedProvider !== undefined) return this.cachedProvider;
    if (this.pendingProvider === undefined) {
      this.pendingProvider = this.loadProvider(configured).then(
        (provider) => {
          this.cachedProvider = provider;
          this.pendingProvider = undefined;
          return provider;
        },
        (error: unknown) => {
          this.pendingProvider = undefined;
          throw error;
        },
      );
    }
    return this.pendingProvider;
  }

  private async loadProvider(name: string): Promise<ModelProvider> {
    const factory = resolveProviderFactory(name);
    if (factory === undefined) {
      const available = listProviders();
      throw new ConfigurationError(
        `Unknown provider "${name}". ${
          available.length > 0
            ? `Registered providers: ${available.join(", ")}.`
            : "No providers are registered."
        } Import "murphychain" (it registers built-in providers) or call registerProvider().`,
        { provider: name, registered: available },
      );
    }
    const modelName = this.resolveModelName();
    const apiKey = this.resolveApiKeyAccessor();
    const config: ProviderFactoryConfig = {
      ...(modelName !== undefined ? { model: modelName } : {}),
      ...(apiKey !== undefined ? { apiKey } : {}),
      ...(this.config.baseUrl !== undefined ? { baseUrl: this.config.baseUrl } : {}),
      ...(this.config.timeoutMs > 0 ? { timeoutMs: this.config.timeoutMs } : {}),
      options: this.config.providerOptions,
    };
    try {
      return await factory(config);
    } catch (error) {
      if (error instanceof ConfigurationError) throw error;
      const message = error instanceof Error ? error.message : String(error);
      const code = (error as { code?: string } | null | undefined)?.code;
      const notFound = code === "ERR_MODULE_NOT_FOUND" || code === "MODULE_NOT_FOUND";
      if (notFound && message.includes("@murphychain/")) {
        throw new ConfigurationError(
          `Provider "${name}" is not installed. Install it with: pnpm add @murphychain/${name}`,
          { provider: name },
        );
      }
      throw ensureMurphyChainError(error, `Failed to load provider "${name}"`, {
        provider: name,
      });
    }
  }

  private resolveModelName(): string | undefined {
    const model = this.config.model;
    if (typeof model === "string") return model;
    if (model instanceof Model) return model.name;
    return undefined;
  }

  private resolveApiKeyAccessor(): (() => string | undefined) | undefined {
    const key = this.config.apiKey;
    if (key === undefined) return undefined;
    if (typeof key === "function") return key;
    return () => key;
  }

  /**
   * Validates and merges an input into a fully resolved request
   * (call > Model defaults > AI config) and enforces input limits.
   */
  private resolveRequest(input: GenerateInput, call: NormalizedCallOptions): ResolvedRequest {
    let raw: GenerateRequest;
    if (typeof input === "string") {
      if (input.length === 0) {
        throw new ValidationError("generate()/stream() expects a non-empty prompt string.");
      }
      raw = { prompt: input };
    } else if (Array.isArray(input)) {
      raw = {
        messages: parseOrThrow(z.array(messageSchema).min(1), input, "Invalid messages."),
      };
    } else if (typeof input === "object" && input !== null) {
      // The schema validates messages exactly once, here.
      raw = parseOrThrow(generateRequestSchema, input, "Invalid generation request.");
    } else {
      throw new ValidationError(
        "generate()/stream() expects a string, an array of messages, or a request object.",
      );
    }

    // Model resolution: request > AI config (Model instances add defaults).
    const modelRef = raw.model ?? this.config.model;
    const modelDefaults = modelRef instanceof Model ? modelRef.defaults : undefined;
    const modelName =
      typeof modelRef === "string"
        ? modelRef
        : modelRef instanceof Model
          ? modelRef.name
          : undefined;
    if (modelName === undefined) {
      throw new ConfigurationError(
        'No model configured. Pass "model" to new AI({ model }) or provide it per request.',
      );
    }

    // Messages: system from config, system from request, then conversation.
    const messages: Message[] = [];
    if (this.config.systemPrompt !== undefined) {
      messages.push({ role: "system", content: this.config.systemPrompt });
    }
    if (raw.system !== undefined) {
      messages.push({ role: "system", content: raw.system });
    }
    if (raw.messages !== undefined) {
      messages.push(...raw.messages);
    } else if (raw.prompt !== undefined) {
      messages.push({ role: "user", content: raw.prompt });
    }

    assertInputLimits(messages, {
      maxMessages: this.config.maxMessages,
      maxInputChars: this.config.maxInputChars,
    });

    let tools: NormalizedTool[] | undefined;
    if (raw.tools !== undefined && raw.tools.length > 0) {
      tools = raw.tools.map((tool) => normalizeTool(tool));
    }
    let toolChoice: ToolChoice | undefined;
    if (raw.toolChoice !== undefined) {
      if (tools === undefined) {
        throw new ValidationError('"toolChoice" requires at least one tool.', {
          field: "toolChoice",
        });
      }
      toolChoice = raw.toolChoice;
    }

    return {
      messages,
      model: modelName,
      temperature: raw.temperature ?? modelDefaults?.temperature ?? this.config.temperature,
      maxTokens: raw.maxTokens ?? modelDefaults?.maxTokens ?? this.config.maxTokens,
      topP: raw.topP ?? modelDefaults?.topP ?? this.config.topP,
      stop: raw.stop ?? modelDefaults?.stop ?? this.config.stop,
      toolChoice,
      tools,
      totalChars: messages.reduce((total, message) => total + message.content.length, 0),
      includeRaw: call.includeRaw ?? this.config.includeRaw,
    };
  }

  private assertCapabilities(
    provider: ModelProvider,
    request: ResolvedRequest,
    streaming: boolean,
  ): void {
    const caps = provider.capabilities;
    if (!caps.generate) {
      throw new UnsupportedCapabilityError(provider.name, "generate");
    }
    if (streaming && !caps.streaming) {
      throw new UnsupportedCapabilityError(provider.name, "streaming");
    }
    if (request.tools !== undefined && !caps.toolCalling) {
      throw new UnsupportedCapabilityError(provider.name, "tool calling");
    }
    if (!caps.systemMessage && request.messages.some((m) => m.role === "system")) {
      throw new UnsupportedCapabilityError(provider.name, "system messages");
    }
  }

  private createContext(
    provider: string,
    model: string,
    call: NormalizedCallOptions,
  ): ExecutionContext {
    return new ExecutionContext({
      provider,
      model,
      timeoutMs: call.timeoutMs ?? this.config.timeoutMs,
      ...(call.signal !== undefined ? { signal: call.signal } : {}),
      metadata: { ...this.config.metadata, ...(call.metadata ?? {}) },
    });
  }

  private buildGenerateOptions(request: ResolvedRequest, ctx: ExecutionContext): GenerateOptions {
    return {
      messages: request.messages,
      model: request.model,
      ...(request.temperature !== undefined ? { temperature: request.temperature } : {}),
      ...(request.maxTokens !== undefined ? { maxTokens: request.maxTokens } : {}),
      ...(request.topP !== undefined ? { topP: request.topP } : {}),
      ...(request.stop !== undefined ? { stop: request.stop } : {}),
      ...(request.tools !== undefined ? { tools: request.tools } : {}),
      ...(request.toolChoice !== undefined ? { toolChoice: request.toolChoice } : {}),
      includeRaw: request.includeRaw,
      signal: ctx.signal,
    };
  }

  /** Maps abort state first, then wraps unknown failures log-safely. */
  private mapCallError(error: unknown, ctx: ExecutionContext): MurphyChainError {
    if (ctx.reason !== undefined) return ctx.abortError();
    return ensureMurphyChainError(error, "Request failed", { provider: ctx.provider });
  }

  private emitRetry(ctx: ExecutionContext, info: RetryAttemptInfo): void {
    this.counters.retries += 1;
    this.bus.emit("retry", {
      requestId: ctx.requestId,
      attempt: info.attempt,
      delayMs: info.delayMs,
      error: info.error,
    });
  }

  private recordSuccess(
    ctx: ExecutionContext,
    usage: TokenUsage,
    finishReason: FinishReason,
    chunkCount: number,
  ): void {
    this.counters.promptTokens += usage.promptTokens;
    this.counters.completionTokens += usage.completionTokens;
    this.counters.totalTokens += usage.totalTokens;
    this.counters.totalLatencyMs += ctx.elapsedMs;
    this.bus.emit("request:success", {
      requestId: ctx.requestId,
      provider: ctx.provider,
      model: ctx.model,
      latencyMs: ctx.elapsedMs,
      usage,
      finishReason,
      chunkCount,
    });
  }

  /** Wraps the provider stream with engine semantics (single-use, events). */
  private iterateStream(
    ctx: ExecutionContext,
    provider: ModelProvider,
    genOptions: GenerateOptions,
    request: ResolvedRequest,
  ): AsyncIterable<AIChunk> {
    let consumed = false;
    return {
      [Symbol.asyncIterator]: () => {
        if (consumed) {
          throw new ConfigurationError(
            "This stream has already been consumed; streams are single-use.",
          );
        }
        consumed = true;
        return this.runStream(ctx, provider, genOptions, request);
      },
    };
  }

  private async *runStream(
    ctx: ExecutionContext,
    provider: ModelProvider,
    genOptions: GenerateOptions,
    request: ResolvedRequest,
  ): AsyncGenerator<AIChunk> {
    let iterator: AsyncIterator<AIChunk> | undefined;
    let completed = false;
    let chunkCount = 0;
    let usage: TokenUsage | undefined;
    let finishReason: FinishReason = "unknown";
    try {
      // Open the stream with retry. Nothing is observable before the first
      // chunk, so replaying here is safe.
      const opened = await withRetry(
        async () => {
          const iterable = provider.stream(genOptions);
          const inner = iterable[Symbol.asyncIterator]();
          try {
            const first = await inner.next();
            return { inner, first };
          } catch (error) {
            try {
              await inner.return?.();
            } catch {
              // Best-effort cleanup of a failed stream.
            }
            throw error;
          }
        },
        this.config.retries,
        {
          signal: ctx.signal,
          onRetry: (info) => this.emitRetry(ctx, info),
          createAbortError: () => ctx.abortError(),
        },
      );
      iterator = opened.inner;
      let result = opened.first;
      while (!result.done) {
        const chunk = result.value;
        if (chunkCount === 0) {
          // First chunk arrived: the deadline covered time-to-first-chunk.
          ctx.clearDeadline();
        }
        if (chunk.usage !== undefined) usage = chunk.usage;
        if (chunk.finishReason !== undefined) finishReason = chunk.finishReason;
        this.counters.chunks += 1;
        this.bus.emit("chunk", {
          requestId: ctx.requestId,
          index: chunkCount,
          textLength: chunk.text.length,
        });
        // The engine owns chunk indexing and passes provider payloads through.
        yield { ...chunk, index: chunkCount };
        chunkCount += 1;
        result = await iterator.next();
      }
      completed = true;
      const finalUsage = usage ?? ZERO_USAGE;
      this.counters.promptTokens += finalUsage.promptTokens;
      this.counters.completionTokens += finalUsage.completionTokens;
      this.counters.totalTokens += finalUsage.totalTokens;
      this.counters.totalLatencyMs += ctx.elapsedMs;
      this.bus.emit("request:success", {
        requestId: ctx.requestId,
        provider: ctx.provider,
        model: request.model,
        latencyMs: ctx.elapsedMs,
        usage: finalUsage,
        finishReason,
        chunkCount,
      });
    } catch (error) {
      const mapped = this.mapCallError(error, ctx);
      this.counters.errors += 1;
      if (ctx.reason !== undefined) {
        this.bus.emit("abort", { requestId: ctx.requestId, reason: ctx.reason });
      }
      this.bus.emit("request:error", {
        requestId: ctx.requestId,
        provider: ctx.provider,
        model: request.model,
        latencyMs: ctx.elapsedMs,
        error: mapped,
      });
      throw mapped;
    } finally {
      if (iterator !== undefined) {
        try {
          await iterator.return?.();
        } catch {
          // Best-effort cleanup.
        }
      }
      if (!completed) {
        // Early break or failure: release the underlying connection.
        ctx.cancel();
      }
      ctx.dispose();
    }
  }
}
