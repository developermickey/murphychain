import { ExecutionCancelledError, MurphyChainError, ProviderRateLimitError } from "./Errors";

/**
 * Bounded retry with exponential backoff and full jitter.
 *
 * Retry policy lives exclusively in the engine: provider adapters configure
 * their official SDKs with `maxRetries: 0` so that attempts are never
 * stacked. Only errors classified as `retryable` are retried.
 */

export interface RetryPolicy {
  /** Additional attempts after the first (0 disables retries). */
  retries: number;
  /** Delay before the first retry, in milliseconds. */
  initialDelayMs: number;
  /** Upper bound for any computed delay, in milliseconds. */
  maxDelayMs: number;
  /** Multiplier applied to the delay after each retry. */
  backoffFactor: number;
  /** Apply full jitter (random delay in `[0, computed]`). */
  jitter: boolean;
}

/** Default policy: 2 retries, 250ms → ×2, 4s cap, full jitter. */
export const DEFAULT_RETRY_POLICY: Readonly<RetryPolicy> = Object.freeze({
  retries: 2,
  initialDelayMs: 250,
  maxDelayMs: 4_000,
  backoffFactor: 2,
  jitter: true,
});

/** Hard cap on honouring a server-provided `Retry-After` hint. */
const MAX_RETRY_AFTER_MS = 60_000;

export interface RetryAttemptInfo {
  /** 1-based number of the attempt about to run. */
  attempt: number;
  delayMs: number;
  error: MurphyChainError;
}

export interface RetryHooks {
  signal?: AbortSignal | undefined;
  onRetry?: ((info: RetryAttemptInfo) => void) | undefined;
  /** Builds the error thrown when the signal aborts during a backoff delay. */
  createAbortError?: (() => Error) | undefined;
}

/**
 * Computes the backoff delay before retry `retryIndex` (0-based).
 * A server-provided `Retry-After` hint takes precedence, capped at 60s.
 */
export function computeDelay(policy: RetryPolicy, retryIndex: number, error?: unknown): number {
  const base = Math.min(
    policy.initialDelayMs * Math.pow(policy.backoffFactor, retryIndex),
    policy.maxDelayMs,
  );
  const backoff = policy.jitter ? Math.round(Math.random() * base) : Math.round(base);
  if (error instanceof ProviderRateLimitError && error.retryAfterMs !== undefined) {
    return Math.max(backoff, Math.min(error.retryAfterMs, MAX_RETRY_AFTER_MS));
  }
  return backoff;
}

/**
 * Runs `operation`, retrying retryable failures according to `policy`.
 * Backoff sleeps are abortable, so cancellation takes effect immediately.
 *
 * Each attempt is also raced against the abort signal: even if a provider
 * ignores its signal and hangs, the engine still honours its timeout and
 * cancellation contract (the abandoned operation's later settlement is
 * swallowed to avoid unhandled rejections).
 */
export async function withRetry<T>(
  operation: (attempt: number) => Promise<T>,
  policy: RetryPolicy,
  hooks: RetryHooks = {},
): Promise<T> {
  const maxAttempts = policy.retries + 1;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    if (hooks.signal?.aborted === true) throw makeAbortError(hooks.createAbortError);
    try {
      return await raceAbort(operation(attempt), hooks.signal, hooks.createAbortError);
    } catch (error) {
      const retryable = error instanceof MurphyChainError && error.retryable;
      const isLastAttempt = attempt === maxAttempts - 1;
      if (!retryable || isLastAttempt) throw error;
      const delayMs = computeDelay(policy, attempt, error);
      hooks.onRetry?.({ attempt: attempt + 2, delayMs, error: error as MurphyChainError });
      await sleep(delayMs, hooks.signal, hooks.createAbortError);
    }
  }
  /* c8 ignore next -- unreachable: the loop always returns or throws */
  throw new ExecutionCancelledError();
}

/** Builds the error raised when the signal aborts (timeout vs. cancellation). */
function makeAbortError(create?: () => Error): Error {
  return create !== undefined ? create() : new ExecutionCancelledError();
}

/**
 * Settles with the operation, or rejects as soon as the signal aborts.
 * The abandoned operation keeps running (the signal is also handed to the
 * provider), but its eventual outcome can never surface or leak.
 */
function raceAbort<T>(
  operation: Promise<T>,
  signal: AbortSignal | undefined,
  createAbortError?: () => Error,
): Promise<T> {
  if (signal === undefined) return operation;
  if (signal.aborted) {
    operation.catch(() => undefined);
    return Promise.reject(makeAbortError(createAbortError));
  }
  return new Promise<T>((resolve, reject) => {
    const onAbort = (): void => {
      operation.catch(() => undefined);
      reject(makeAbortError(createAbortError));
    };
    signal.addEventListener("abort", onAbort, { once: true });
    operation.then(
      (value) => {
        signal.removeEventListener("abort", onAbort);
        resolve(value);
      },
      (error: unknown) => {
        signal.removeEventListener("abort", onAbort);
        reject(error);
      },
    );
  });
}

/** Abortable sleep. Rejects as soon as the signal aborts. */
function sleep(ms: number, signal?: AbortSignal, createAbortError?: () => Error): Promise<void> {
  if (ms <= 0 && signal?.aborted !== true) return Promise.resolve();
  return new Promise<void>((resolve, reject) => {
    const settle = (fn: () => void): void => {
      clearTimeout(handle);
      signal?.removeEventListener("abort", onAbort);
      fn();
    };
    const onAbort = (): void => {
      settle(() => reject(makeAbortError(createAbortError)));
    };
    const handle = setTimeout(() => {
      settle(resolve);
    }, ms);
    if (signal !== undefined) {
      if (signal.aborted) {
        onAbort();
        return;
      }
      signal.addEventListener("abort", onAbort, { once: true });
    }
  });
}
