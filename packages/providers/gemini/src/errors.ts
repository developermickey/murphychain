import { ApiError } from "@google/genai";
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

const PROVIDER = "gemini";

/**
 * Error classes the Gemini SDK throws for transport failures are internal
 * (not exported), so they are identified by their stable `name`.
 */
const TRANSPORT_TIMEOUT_NAMES = new Set(["RequestTimeoutError", "APIConnectionTimeoutError"]);
const TRANSPORT_ABORT_NAMES = new Set(["RequestAbortedError", "APIUserAbortError"]);
const TRANSPORT_CONNECTION_NAMES = new Set(["ConnectionError", "APIConnectionError"]);

/**
 * Translates Gemini SDK failures into the MurphyChain error taxonomy.
 *
 * - SDK abort errors are rethrown as-is: they fire only when *our* abort
 *   signal aborted the request, and the engine maps them (via the recorded
 *   abort reason) to `ExecutionTimeoutError` / `ExecutionCancelledError`.
 * - HTTP failures arrive as `ApiError` with a `status`.
 * - Transport failures (DNS/socket/client timeout) are mapped to retryable
 *   connection/timeout errors so the engine can retry them.
 */
export function mapGeminiError(error: unknown): MurphyChainError {
  if (error instanceof MurphyChainError) return error;
  if (error instanceof ApiError) return mapApiError(error);
  const name = error instanceof Error ? error.name : "";
  if (TRANSPORT_ABORT_NAMES.has(name)) throw error;
  const message = error instanceof Error ? error.message : String(error);
  if (TRANSPORT_TIMEOUT_NAMES.has(name)) {
    return new ProviderTimeoutError(PROVIDER, `Gemini request timed out: ${message}`, {
      cause: error,
    });
  }
  if (TRANSPORT_CONNECTION_NAMES.has(name)) {
    return new ProviderConnectionError(PROVIDER, `Could not reach Gemini: ${message}`, {
      cause: error,
    });
  }
  if (name === "InvalidRequestError") {
    return new ProviderBadRequestError(PROVIDER, `Gemini SDK rejected the request: ${message}`, {
      cause: error,
    });
  }
  return new ProviderError(PROVIDER, `Unexpected Gemini failure: ${message}`, {
    code: "PROVIDER_UNEXPECTED",
    retryable: false,
    cause: error,
  });
}

function mapApiError(error: ApiError): MurphyChainError {
  const status = error.status;
  // ApiError.message embeds the JSON error body returned by the API.
  const message = `Gemini error ${status}: ${error.message}`;
  const base = { status } as const;

  if (status === 401 || status === 403) {
    return new ProviderAuthError(PROVIDER, message, { ...base, cause: error });
  }
  if (status === 429) {
    return new ProviderRateLimitError(PROVIDER, message, { ...base, cause: error });
  }
  if (status === 400 || status === 404 || status === 422) {
    return new ProviderBadRequestError(PROVIDER, message, { ...base, cause: error });
  }
  if (status === 451) {
    return new ProviderContentFilterError(PROVIDER, message, { ...base, cause: error });
  }
  if (status >= 500) {
    return new ProviderServerError(PROVIDER, message, { ...base, cause: error });
  }
  return new ProviderError(PROVIDER, message, {
    code: "PROVIDER_ERROR",
    retryable: false,
    ...base,
    cause: error,
  });
}
