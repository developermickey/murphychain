import {
  APIConnectionError,
  APIConnectionTimeoutError,
  APIError,
  APIUserAbortError,
  OpenAIError,
} from "openai";
import {
  MurphyChainError,
  ProviderAuthError,
  ProviderBadRequestError,
  ProviderConnectionError,
  ProviderContentFilterError,
  ProviderError,
  ProviderRateLimitError,
  ProviderServerError,
  ProviderTimeoutError,
} from "@murphychain/core";

const PROVIDER = "openai";

/**
 * Parses a `retry-after` header (seconds or HTTP-date) into milliseconds.
 * Returns `undefined` when absent or unparseable.
 */
export function parseRetryAfterMs(headers: Headers | undefined): number | undefined {
  const value = headers?.get("retry-after");
  if (value === null || value === undefined || value.length === 0) return undefined;
  const seconds = Number(value);
  if (Number.isFinite(seconds)) return Math.max(0, Math.round(seconds * 1000));
  const dateMs = Date.parse(value);
  if (!Number.isNaN(dateMs)) return Math.max(0, dateMs - Date.now());
  return undefined;
}

/**
 * Translates OpenAI SDK failures into the MurphyChain error taxonomy.
 *
 * - `APIUserAbortError` is rethrown as-is: it is only raised by the SDK when
 *   *our* abort signal fired, and the engine maps it (via the abort reason) to
 *   `ExecutionTimeoutError` / `ExecutionCancelledError`.
 * - All other SDK errors become non-retryable or retryable `ProviderError`
 *   subclasses that drive the engine's retry policy.
 */
export function mapOpenAIError(error: unknown): MurphyChainError {
  if (error instanceof MurphyChainError) return error;
  if (error instanceof APIUserAbortError) throw error;
  if (error instanceof APIConnectionTimeoutError) {
    return new ProviderTimeoutError(PROVIDER, `OpenAI request timed out: ${error.message}`, {
      cause: error,
    });
  }
  if (error instanceof APIConnectionError) {
    return new ProviderConnectionError(PROVIDER, `Could not reach OpenAI: ${error.message}`, {
      cause: error,
    });
  }
  if (error instanceof APIError) {
    return mapApiError(error);
  }
  if (error instanceof OpenAIError) {
    // SDK-level misuse (bad params before the request is sent).
    return new ProviderError(PROVIDER, `OpenAI SDK error: ${error.message}`, {
      code: "PROVIDER_SDK_ERROR",
      retryable: false,
      cause: error,
    });
  }
  const message = error instanceof Error ? error.message : String(error);
  return new ProviderError(PROVIDER, `Unexpected OpenAI failure: ${message}`, {
    code: "PROVIDER_UNEXPECTED",
    retryable: false,
    cause: error,
  });
}

function mapApiError(error: APIError): MurphyChainError {
  const status = error.status;
  const message = `OpenAI error ${status ?? "unknown"}: ${error.message}`;
  const base = {
    status: status ?? undefined,
    requestId: error.requestID ?? undefined,
  } as const;

  if (status === 401 || status === 403) {
    return new ProviderAuthError(PROVIDER, message, { ...base, cause: error });
  }
  if (status === 429) {
    const retryAfterMs = parseRetryAfterMs(error.headers);
    return new ProviderRateLimitError(PROVIDER, message, {
      ...base,
      ...(retryAfterMs !== undefined ? { retryAfterMs } : {}),
      cause: error,
    });
  }
  if (status === 400 || status === 404 || status === 422) {
    return new ProviderBadRequestError(PROVIDER, message, { ...base, cause: error });
  }
  if (status === 451) {
    return new ProviderContentFilterError(PROVIDER, message, { ...base, cause: error });
  }
  if (status !== undefined && status >= 500) {
    return new ProviderServerError(PROVIDER, message, { ...base, cause: error });
  }
  return new ProviderError(PROVIDER, message, {
    code: "PROVIDER_ERROR",
    retryable: false,
    ...base,
    cause: error,
  });
}

