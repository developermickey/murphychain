/**
 * MurphyChain error taxonomy.
 *
 * Every error raised by the framework extends {@link MurphyChainError} and
 * carries a stable `code` plus a `retryable` flag that drives the engine's
 * retry policy. All messages are passed through {@link redactSecrets} so that
 * credentials can never surface in logs or error handlers.
 */

const SECRET_PATTERNS: readonly RegExp[] = [
  // OpenAI / Anthropic style keys
  /sk-[A-Za-z0-9_-]{8,}/g,
  // Google API keys
  /AIza[A-Za-z0-9_-]{8,}/g,
  // Authorization headers
  /Bearer\s+[A-Za-z0-9._~+/=-]{8,}/gi,
  // GitHub tokens
  /gh[pousr]_[A-Za-z0-9]{10,}/g,
  // Slack tokens
  /xox[baprs]-[A-Za-z0-9-]{10,}/g,
  // Connection strings with embedded credentials
  /(?:postgres|postgresql|mysql|mongodb|redis|amqp):\/\/[^\s:/@]+:[^\s@]+@/gi,
];

const REDACTED = "[REDACTED]";

/**
 * Replaces anything that looks like a credential with a redacted marker.
 * The first three characters of a matched secret are kept so that operators
 * can still identify *which* key leaked without exposing it.
 */
export function redactSecrets(input: string): string {
  let output = input;
  for (const pattern of SECRET_PATTERNS) {
    output = output.replace(pattern, (match) => {
      const prefix = match.slice(0, 3);
      return `${prefix === "Bea" ? "Bearer" : prefix}…${REDACTED}`;
    });
  }
  return output;
}

/** Truncates long provider messages so error logs stay readable. */
export function truncateMessage(input: string, maxLength = 1_000): string {
  if (input.length <= maxLength) return input;
  return `${input.slice(0, maxLength)}… [truncated]`;
}

export interface MurphyChainErrorOptions {
  /** Stable machine-readable identifier (e.g. `"PROVIDER_AUTH"`). */
  code: string;
  /** Whether the engine may retry this error automatically. */
  retryable?: boolean | undefined;
  /** Structured, log-safe context. Must never contain credentials. */
  details?: Record<string, unknown> | undefined;
  /** Underlying cause, preserved for debugging. */
  cause?: unknown;
}

/** Base class of every error thrown by MurphyChain. */
export class MurphyChainError extends Error {
  readonly code: string;
  readonly retryable: boolean;
  readonly details: Readonly<Record<string, unknown>>;

  constructor(message: string, options: MurphyChainErrorOptions) {
    super(
      redactSecrets(message),
      options.cause !== undefined ? { cause: options.cause } : {},
    );
    this.name = "MurphyChainError";
    this.code = options.code;
    this.retryable = options.retryable ?? false;
    this.details = Object.freeze({ ...(options.details ?? {}) });
  }
}

/** Developer-facing configuration problem (bad config, unknown provider…). */
export class ConfigurationError extends MurphyChainError {
  constructor(message: string, details?: Record<string, unknown>) {
    super(message, { code: "CONFIGURATION_ERROR", ...(details ? { details } : {}) });
    this.name = "ConfigurationError";
  }
}

/** Invalid runtime input (schema violations, exceeded limits). */
export class ValidationError extends MurphyChainError {
  constructor(message: string, details?: Record<string, unknown>) {
    super(message, { code: "VALIDATION_ERROR", ...(details ? { details } : {}) });
    this.name = "ValidationError";
  }
}

/** A prompt template was formatted with missing or unknown variables. */
export class PromptTemplateError extends MurphyChainError {
  constructor(message: string, details?: Record<string, unknown>) {
    super(message, { code: "PROMPT_TEMPLATE_ERROR", ...(details ? { details } : {}) });
    this.name = "PromptTemplateError";
  }
}

/** The provider cannot serve the requested capability. */
export class UnsupportedCapabilityError extends MurphyChainError {
  constructor(provider: string, capability: string) {
    super(`Provider "${provider}" does not support "${capability}".`, {
      code: "UNSUPPORTED_CAPABILITY",
      details: { provider, capability },
    });
    this.name = "UnsupportedCapabilityError";
  }
}

/** The local deadline for a request was exceeded. Never retried. */
export class ExecutionTimeoutError extends MurphyChainError {
  constructor(message = "Request timed out.", details?: Record<string, unknown>) {
    super(message, { code: "EXECUTION_TIMEOUT", ...(details ? { details } : {}) });
    this.name = "ExecutionTimeoutError";
  }
}

/** The caller aborted a request via `AbortSignal`. Never retried. */
export class ExecutionCancelledError extends MurphyChainError {
  constructor(message = "Request cancelled by caller.", details?: Record<string, unknown>) {
    super(message, { code: "EXECUTION_CANCELLED", ...(details ? { details } : {}) });
    this.name = "ExecutionCancelledError";
  }
}

export interface ProviderErrorOptions {
  code?: string | undefined;
  retryable?: boolean | undefined;
  /** HTTP status reported by the provider, when available. */
  status?: number | undefined;
  /** Provider request id, when available (for support tickets). */
  requestId?: string | undefined;
  /** Structured, log-safe context. */
  details?: Record<string, unknown> | undefined;
  cause?: unknown;
}

/** Normalized failure reported by a provider adapter. */
export class ProviderError extends MurphyChainError {
  readonly provider: string;
  readonly status?: number;
  readonly requestId?: string;

  constructor(provider: string, message: string, options: ProviderErrorOptions = {}) {
    const details: Record<string, unknown> = { ...options.details, provider };
    if (options.status !== undefined) details.status = options.status;
    if (options.requestId !== undefined) details.requestId = options.requestId;
    super(message, {
      code: options.code ?? "PROVIDER_ERROR",
      retryable: options.retryable ?? false,
      details,
      ...(options.cause !== undefined ? { cause: options.cause } : {}),
    });
    this.name = "ProviderError";
    this.provider = provider;
    if (options.status !== undefined) this.status = options.status;
    if (options.requestId !== undefined) this.requestId = options.requestId;
  }
}

/** Authentication or authorization failed (401/403). Never retried. */
export class ProviderAuthError extends ProviderError {
  constructor(provider: string, message: string, options: ProviderErrorOptions = {}) {
    super(provider, message, { ...options, code: "PROVIDER_AUTH", retryable: false });
    this.name = "ProviderAuthError";
  }
}

/** The provider rate-limited the request (429). Retried with backoff. */
export class ProviderRateLimitError extends ProviderError {
  /** Server-provided wait hint in milliseconds, when available. */
  readonly retryAfterMs?: number;

  constructor(
    provider: string,
    message: string,
    options: ProviderErrorOptions & { retryAfterMs?: number } = {},
  ) {
    super(provider, message, { ...options, code: "PROVIDER_RATE_LIMIT", retryable: true });
    this.name = "ProviderRateLimitError";
    if (options.retryAfterMs !== undefined) this.retryAfterMs = options.retryAfterMs;
  }
}

/** The request was rejected as invalid (400/404/422). Never retried. */
export class ProviderBadRequestError extends ProviderError {
  constructor(provider: string, message: string, options: ProviderErrorOptions = {}) {
    super(provider, message, { ...options, code: "PROVIDER_BAD_REQUEST", retryable: false });
    this.name = "ProviderBadRequestError";
  }
}

/** The provider failed internally (5xx). Retried with backoff. */
export class ProviderServerError extends ProviderError {
  constructor(provider: string, message: string, options: ProviderErrorOptions = {}) {
    super(provider, message, { ...options, code: "PROVIDER_SERVER_ERROR", retryable: true });
    this.name = "ProviderServerError";
  }
}

/** The provider could not be reached (DNS/TLS/socket). Retried with backoff. */
export class ProviderConnectionError extends ProviderError {
  constructor(provider: string, message: string, options: ProviderErrorOptions = {}) {
    super(provider, message, { ...options, code: "PROVIDER_CONNECTION", retryable: true });
    this.name = "ProviderConnectionError";
  }
}

/** The provider timed out server-side. Retried with backoff. */
export class ProviderTimeoutError extends ProviderError {
  constructor(provider: string, message: string, options: ProviderErrorOptions = {}) {
    super(provider, message, { ...options, code: "PROVIDER_TIMEOUT", retryable: true });
    this.name = "ProviderTimeoutError";
  }
}

/** Output was blocked by the provider's safety systems. Never retried. */
export class ProviderContentFilterError extends ProviderError {
  constructor(provider: string, message: string, options: ProviderErrorOptions = {}) {
    super(provider, message, {
      ...options,
      code: "PROVIDER_CONTENT_FILTER",
      retryable: false,
    });
    this.name = "ProviderContentFilterError";
  }
}

/**
 * Wraps an unexpected value into a log-safe MurphyChainError.
 * Used as the last line of defense so unknown failures still carry the
 * framework's shape (and redacted messages).
 */
export function ensureMurphyChainError(
  error: unknown,
  fallbackMessage: string,
  details?: Record<string, unknown>,
): MurphyChainError {
  if (error instanceof MurphyChainError) return error;
  const message =
    error instanceof Error
      ? `${fallbackMessage}: ${error.message}`
      : `${fallbackMessage}: ${String(error)}`;
  return new MurphyChainError(message, {
    code: "INTERNAL_ERROR",
    ...(details ? { details } : {}),
    cause: error,
  });
}
