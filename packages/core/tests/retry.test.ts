import { afterEach, describe, expect, it } from "vitest";
import {
  DEFAULT_RETRY_POLICY,
  ExecutionCancelledError,
  ProviderAuthError,
  ProviderRateLimitError,
  ProviderServerError,
  computeDelay,
  withRetry,
} from "@murphychain/core";
import type { RetryPolicy } from "@murphychain/core";

const NO_RETRY: RetryPolicy = { retries: 0, initialDelayMs: 0, maxDelayMs: 0, backoffFactor: 1, jitter: false };
const FAST: RetryPolicy = { retries: 2, initialDelayMs: 1, maxDelayMs: 5, backoffFactor: 2, jitter: false };

const unhandled: unknown[] = [];
const onUnhandled = (reason: unknown): void => {
  unhandled.push(reason);
};
process.on("unhandledRejection", onUnhandled);

afterEach(() => {
  unhandled.length = 0;
});

describe("DEFAULT_RETRY_POLICY", () => {
  it("is frozen and matches the documented defaults", () => {
    expect(Object.isFrozen(DEFAULT_RETRY_POLICY)).toBe(true);
    expect(DEFAULT_RETRY_POLICY).toEqual({
      retries: 2,
      initialDelayMs: 250,
      maxDelayMs: 4_000,
      backoffFactor: 2,
      jitter: true,
    });
  });
});

describe("computeDelay", () => {
  it("grows exponentially and caps at maxDelayMs (no jitter)", () => {
    const policy: RetryPolicy = { retries: 5, initialDelayMs: 100, maxDelayMs: 350, backoffFactor: 2, jitter: false };
    expect(computeDelay(policy, 0)).toBe(100);
    expect(computeDelay(policy, 1)).toBe(200);
    expect(computeDelay(policy, 2)).toBe(350); // 400 capped
    expect(computeDelay(policy, 3)).toBe(350);
  });

  it("stays within [0, base] with full jitter", () => {
    const policy: RetryPolicy = { retries: 5, initialDelayMs: 100, maxDelayMs: 10_000, backoffFactor: 2, jitter: true };
    for (let i = 0; i < 50; i++) {
      const delay = computeDelay(policy, 1);
      expect(delay).toBeGreaterThanOrEqual(0);
      expect(delay).toBeLessThanOrEqual(200);
    }
  });

  it("honours a Retry-After hint, capped at 60s", () => {
    const policy: RetryPolicy = { retries: 5, initialDelayMs: 10, maxDelayMs: 1_000, backoffFactor: 2, jitter: false };
    const hint = (retryAfterMs: number): number =>
      computeDelay(policy, 0, new ProviderRateLimitError("p", "429", { retryAfterMs }));
    expect(hint(5_000)).toBe(5_000);
    expect(hint(5)).toBe(10); // never shorter than the backoff itself
    expect(hint(120_000)).toBe(60_000); // hard cap
  });

  it("ignores Retry-After on non-rate-limit errors", () => {
    const policy: RetryPolicy = { retries: 5, initialDelayMs: 10, maxDelayMs: 1_000, backoffFactor: 2, jitter: false };
    expect(computeDelay(policy, 0, new ProviderServerError("p", "500"))).toBe(10);
  });
});

describe("withRetry", () => {
  it("returns the first successful attempt without retries", async () => {
    let calls = 0;
    const result = await withRetry(
      async () => {
        calls += 1;
        return "ok";
      },
      FAST,
    );
    expect(result).toBe("ok");
    expect(calls).toBe(1);
  });

  it("retries retryable errors until success and reports each attempt", async () => {
    let calls = 0;
    const attempts: number[] = [];
    const result = await withRetry(
      async () => {
        calls += 1;
        if (calls < 3) throw new ProviderServerError("p", `fail ${calls}`);
        return "ok";
      },
      FAST,
      { onRetry: (info) => attempts.push(info.attempt) },
    );
    expect(result).toBe("ok");
    expect(calls).toBe(3);
    expect(attempts).toEqual([2, 3]);
  });

  it("throws the last error when retries are exhausted", async () => {
    let calls = 0;
    await expect(
      withRetry(
        async () => {
          calls += 1;
          throw new ProviderServerError("p", "always failing");
        },
        { ...FAST, retries: 2 },
      ),
    ).rejects.toThrowError(ProviderServerError);
    expect(calls).toBe(3); // 1 initial + 2 retries
  });

  it("does not retry non-retryable errors", async () => {
    let calls = 0;
    await expect(
      withRetry(
        async () => {
          calls += 1;
          throw new ProviderAuthError("p", "401");
        },
        { ...FAST, retries: 5 },
      ),
    ).rejects.toThrowError(ProviderAuthError);
    expect(calls).toBe(1);
  });

  it("runs exactly one attempt when retries: 0", async () => {
    let calls = 0;
    await expect(
      withRetry(
        async () => {
          calls += 1;
          throw new ProviderServerError("p", "boom");
        },
        NO_RETRY,
      ),
    ).rejects.toThrowError(ProviderServerError);
    expect(calls).toBe(1);
  });

  it("fails fast when the signal is already aborted", async () => {
    let calls = 0;
    const controller = new AbortController();
    controller.abort();
    await expect(
      withRetry(
        async () => {
          calls += 1;
          return "never";
        },
        FAST,
        { signal: controller.signal, createAbortError: () => new Error("pre-aborted") },
      ),
    ).rejects.toThrow("pre-aborted");
    expect(calls).toBe(0);
  });

  it("rejects as soon as the signal aborts during a backoff sleep", async () => {
    const controller = new AbortController();
    const promise = withRetry(
      async () => {
        throw new ProviderServerError("p", "retry me");
      },
      { retries: 5, initialDelayMs: 10_000, maxDelayMs: 10_000, backoffFactor: 1, jitter: false },
      { signal: controller.signal, createAbortError: () => new ExecutionCancelledError() },
    );
    setTimeout(() => controller.abort(), 10);
    await expect(promise).rejects.toThrowError(ExecutionCancelledError);
  });

  // raceAbort: the engine's timeout contract must hold even when a provider
  // ignores the abort signal and hangs forever.
  it("rejects when the signal aborts while the operation hangs", async () => {
    const controller = new AbortController();
    const hanging = new Promise<string>(() => undefined); // never settles
    const promise = withRetry(() => hanging, NO_RETRY, {
      signal: controller.signal,
      createAbortError: () => new Error("hung-out"),
    });
    controller.abort();
    await expect(promise).rejects.toThrow("hung-out");
  });

  it("swallows a late rejection of the abandoned operation", async () => {
    const controller = new AbortController();
    let rejectLater: ((error: Error) => void) | undefined;
    const hanging = new Promise<string>((_resolve, reject) => {
      rejectLater = reject;
    });
    const promise = withRetry(() => hanging, NO_RETRY, {
      signal: controller.signal,
      createAbortError: () => new Error("aborted"),
    });
    controller.abort();
    await expect(promise).rejects.toThrow("aborted");
    rejectLater?.(new Error("late failure"));
    await new Promise((resolve) => setTimeout(resolve, 20));
    expect(unhandled).toEqual([]);
  });

  it("returns the operation result directly when no signal is given", async () => {
    const result = await withRetry(() => Promise.resolve(42), NO_RETRY);
    expect(result).toBe(42);
  });
});
