import type { ZodType } from "zod";

/**
 * Shared public types of the MurphyChain core.
 *
 * Every provider adapter maps its native payloads onto these types so that
 * application code is portable across providers.
 */

/** Roles supported in a conversation. */
export type Role = "system" | "user" | "assistant" | "tool";

/** A tool invocation requested by the model. */
export interface ToolCall {
  /** Provider-assigned identifier correlating the call with its result. */
  id: string;
  /** Name of the tool the model wants to invoke. */
  name: string;
  /** Parsed JSON arguments. Never executed by the framework. */
  arguments: Record<string, unknown>;
}

/** A single conversation message. */
export interface Message {
  role: Role;
  content: string;
  /** Tool name. Required on `role: "tool"` messages. */
  name?: string | undefined;
  /** Id of the tool call this message responds to. Required on `role: "tool"`. */
  toolCallId?: string | undefined;
  /** Tool calls requested by an assistant message. */
  toolCalls?: ToolCall[] | undefined;
}

/** Token accounting for a completed generation. */
export interface TokenUsage {
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
}

/** Normalized finish reason across providers. */
export type FinishReason =
  | "stop"
  | "length"
  | "tool_calls"
  | "content_filter"
  | "error"
  | "unknown";

/** Result of a non-streaming generation. */
export interface AIResponse {
  /** Concatenated text of the assistant message (may be empty for tool calls). */
  text: string;
  /**
   * Schema-validated structured output. Set only when the request carried an
   * `output` schema (or via `generateObject()`); absent otherwise. The value
   * has been parsed from the model's reply and validated against the schema,
   * so it is safe to use without further checks.
   */
  object?: unknown;
  /** Provider that served the request (e.g. `"openai"`). */
  provider: string;
  /** Model that served the request. */
  model: string;
  /** Full assistant message, including any tool calls. */
  message: Message;
  finishReason: FinishReason;
  usage: TokenUsage;
  /** Wall-clock latency measured by the engine, including retries. */
  latencyMs: number;
  /** Raw provider payload. Omit when `includeRaw: false`. */
  raw?: unknown;
}

/** One incremental update of a streaming generation. */
export interface AIChunk {
  /**
   * Text delta for this chunk. Empty on the terminal chunk, which carries
   * `finishReason` / `usage` / `toolCalls` instead.
   */
  text: string;
  /** Zero-based index, stamped by the engine. */
  index: number;
  /** Present on the terminal chunk when the provider reported it. */
  finishReason?: FinishReason | undefined;
  /** Present on the terminal chunk when the provider reported usage. */
  usage?: TokenUsage | undefined;
  /** Present on the terminal chunk when the model requested tool calls. */
  toolCalls?: ToolCall[] | undefined;
}

/** A free-form JSON Schema object (draft 2020-12 subset is sufficient). */
export type JsonSchema = Record<string, unknown>;

/**
 * User-facing tool definition. `parameters` accepts either a Zod schema
 * (recommended) or a plain JSON Schema object.
 */
export interface ToolDefinition {
  name: string;
  description: string;
  parameters: ZodType | JsonSchema;
}

/**
 * Engine-normalized tool definition. Provider adapters receive this shape —
 * Zod-to-JSON-Schema conversion happens once, in the engine.
 */
export interface NormalizedTool {
  name: string;
  description: string;
  jsonSchema: JsonSchema;
}

/** How the model should be asked to (not) call tools. */
export type ToolChoice = "auto" | "none" | "required" | { readonly name: string };

/** Options shared by embedding calls. */
export interface EmbedOptions {
  signal?: AbortSignal | undefined;
}

/** Cumulative counters exposed by {@link AI.stats}. */
export interface AIStats {
  /** Requests that entered the execution lifecycle (generate/stream/embed). */
  requests: number;
  /** Requests that finished with an error. */
  errors: number;
  /** Retry attempts performed across all requests. */
  retries: number;
  /** Stream chunks yielded across all requests. */
  chunks: number;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  /** Aggregate wall-clock latency of all requests, in milliseconds. */
  totalLatencyMs: number;
}

/** A zeroed usage object. */
export const ZERO_USAGE: TokenUsage = Object.freeze({
  promptTokens: 0,
  completionTokens: 0,
  totalTokens: 0,
});
