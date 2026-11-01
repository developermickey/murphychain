/**
 * MurphyChain core — engine, provider contract, prompts, and errors.
 *
 * Most applications should import from `murphychain` (the facade), which
 * additionally registers the built-in provider adapters.
 */

export { AI } from "./AI";
export type {
  AIConfig,
  CallOptions,
  GenerateInput,
  GenerateRequest,
} from "./AI";

export { Model } from "./Model";
export type { ModelDefaults } from "./Model";

export { Prompt, PromptTemplate } from "./Prompt";
export type { PromptValue } from "./Prompt";

export { EventBus } from "./Events";
export type {
  AbortEvent,
  AIEventMap,
  AIEventName,
  AIEventListener,
  ChunkEvent,
  RequestErrorEvent,
  RequestStartEvent,
  RequestSuccessEvent,
  RetryEvent,
} from "./Events";

export { listProviders, registerProvider, resolveProviderFactory, unregisterProvider } from "./registry";
export type { RegisterProviderOptions } from "./registry";

export type {
  GenerateOptions,
  ModelProvider,
  ProviderCapabilities,
  ProviderFactory,
  ProviderFactoryConfig,
} from "./provider";

export { DEFAULT_RETRY_POLICY, computeDelay, withRetry } from "./retry";
export type { RetryAttemptInfo, RetryHooks, RetryPolicy } from "./retry";

export {
  ConfigurationError,
  ensureMurphyChainError,
  ExecutionCancelledError,
  ExecutionTimeoutError,
  MurphyChainError,
  ProviderAuthError,
  ProviderBadRequestError,
  ProviderConnectionError,
  ProviderContentFilterError,
  ProviderError,
  ProviderRateLimitError,
  ProviderServerError,
  ProviderTimeoutError,
  PromptTemplateError,
  redactSecrets,
  truncateMessage,
  UnsupportedCapabilityError,
  ValidationError,
} from "./Errors";
export type { MurphyChainErrorOptions, ProviderErrorOptions } from "./Errors";

export { normalizeTool } from "./validation";
export type { InputLimits, ValidationIssue } from "./validation";

export {
  ZERO_USAGE,
  type AIChunk,
  type AIResponse,
  type AIStats,
  type EmbedOptions,
  type FinishReason,
  type JsonSchema,
  type Message,
  type NormalizedTool,
  type Role,
  type TokenUsage,
  type ToolCall,
  type ToolChoice,
  type ToolDefinition,
} from "./types";

export { z } from "zod";
