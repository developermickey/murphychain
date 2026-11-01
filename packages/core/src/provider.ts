import type {
  AIChunk,
  AIResponse,
  EmbedOptions,
  Message,
  NormalizedTool,
  ToolChoice,
} from "./types";

/**
 * The provider contract.
 *
 * Every LLM integration implements this interface, which makes providers
 * interchangeable: switching providers is a one-line change in application
 * code. Adapters map their official SDKs onto MurphyChain's unified types,
 * translate SDK failures into the MurphyChain error taxonomy, and disable
 * built-in SDK retries (the engine owns retry policy).
 */

/** What a provider can do. The engine fails fast on unsupported features. */
export interface ProviderCapabilities {
  /** Non-streaming generation. Always required. */
  generate: boolean;
  /** Streaming generation via `stream()`. */
  streaming: boolean;
  /** Tool (function) definitions in requests and tool calls in responses. */
  toolCalling: boolean;
  /** `embed()` / `embedMany()`. */
  embeddings: boolean;
  /** `role: "system"` messages. */
  systemMessage: boolean;
  /** Structured output / JSON mode. Not yet implemented by the engine. */
  structuredOutput: boolean;
}

/** Normalized request passed to {@link ModelProvider.generate} / `stream`. */
export interface GenerateOptions {
  messages: Message[];
  /** Resolved model name. Direct provider users may instead set it in config. */
  model?: string | undefined;
  temperature?: number | undefined;
  maxTokens?: number | undefined;
  topP?: number | undefined;
  stop?: string[] | undefined;
  tools?: NormalizedTool[] | undefined;
  toolChoice?: ToolChoice | undefined;
  /** Cancellation signal. Providers must pass it to their SDK. */
  signal?: AbortSignal | undefined;
  /** Attach `raw` to the response (default: `true`). */
  includeRaw?: boolean | undefined;
}

/** The standardized provider interface of MurphyChain. */
export interface ModelProvider {
  /** Stable identifier (e.g. `"openai"`). */
  readonly name: string;
  readonly capabilities: ProviderCapabilities;
  generate(options: GenerateOptions): Promise<AIResponse>;
  stream(options: GenerateOptions): AsyncIterable<AIChunk>;
  embed(text: string, options?: EmbedOptions): Promise<number[]>;
  embedMany(texts: string[], options?: EmbedOptions): Promise<number[][]>;
}

/** Configuration the engine passes to a registered provider factory. */
export interface ProviderFactoryConfig {
  /** Default model from `AIConfig.model`, when it is a plain string. */
  model?: string | undefined;
  /** Accessor invoked per request so credentials can rotate at runtime. */
  apiKey?: (() => string | undefined) | undefined;
  /** Override of the provider's API endpoint (must be http(s)). */
  baseUrl?: string | undefined;
  /** Client-level timeout hint in milliseconds, when configured. */
  timeoutMs?: number | undefined;
  /** Provider-specific options, validated by the provider itself. */
  options?: Record<string, unknown> | undefined;
}

/** Factory registered under a provider name. May be async (lazy loading). */
export type ProviderFactory = (
  config: ProviderFactoryConfig,
) => ModelProvider | Promise<ModelProvider>;
