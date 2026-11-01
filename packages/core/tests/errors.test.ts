import { describe, expect, it } from "vitest";
import {
  ConfigurationError,
  MurphyChainError,
  ProviderAuthError,
  ProviderBadRequestError,
  ProviderConnectionError,
  ProviderContentFilterError,
  ProviderError,
  ProviderRateLimitError,
  ProviderServerError,
  ProviderTimeoutError,
  UnsupportedCapabilityError,
  ValidationError,
  ensureMurphyChainError,
  redactSecrets,
  truncateMessage,
} from "@murphychain/core";

describe("redactSecrets", () => {
  it("redacts OpenAI-style keys, keeping a 3-char prefix", () => {
    const out = redactSecrets("key is sk-abcdef1234567890ABCD here");
    expect(out).toBe("key is sk-…[REDACTED] here");
  });

  it("redacts Google API keys", () => {
    expect(redactSecrets("AIzaSyA1234567890abcdef")).toBe("AIz…[REDACTED]");
  });

  it("redacts bearer tokens", () => {
    expect(redactSecrets("Authorization: Bearer abcdef123456")).toBe(
      "Authorization: Bearer…[REDACTED]",
    );
  });

  it("redacts GitHub and Slack tokens", () => {
    expect(redactSecrets("ghp_abcdefghijklmno")).toContain("[REDACTED]");
    expect(redactSecrets("xoxb-1234567890-abcdef")).toContain("[REDACTED]");
  });

  it("redacts credentials embedded in connection strings", () => {
    const out = redactSecrets("postgres://user:pass@host:5432/db");
    expect(out).not.toContain("pass@");
    expect(out).toContain("[REDACTED]");
  });

  it("leaves innocent text untouched", () => {
    expect(redactSecrets("hello world 123")).toBe("hello world 123");
  });
});

describe("truncateMessage", () => {
  it("keeps short messages as-is", () => {
    expect(truncateMessage("short")).toBe("short");
  });
  it("truncates long messages with a marker", () => {
    const out = truncateMessage("x".repeat(1_100));
    expect(out.length).toBeLessThan(1_100);
    expect(out).toContain("[truncated]");
  });
});

describe("MurphyChainError", () => {
  it("carries code, retryable, frozen details, and cause", () => {
    const cause = new Error("inner");
    const error = new MurphyChainError("boom", {
      code: "X",
      retryable: true,
      details: { a: 1 },
      cause,
    });
    expect(error).toBeInstanceOf(Error);
    expect(error.code).toBe("X");
    expect(error.retryable).toBe(true);
    expect(error.details).toEqual({ a: 1 });
    expect(Object.isFrozen(error.details)).toBe(true);
    expect(error.cause).toBe(cause);
  });

  it("defaults retryable to false", () => {
    expect(new MurphyChainError("x", { code: "X" }).retryable).toBe(false);
  });

  it("redacts secrets in the message at construction time", () => {
    const error = new MurphyChainError("leaked sk-abcdef1234567890", { code: "X" });
    expect(error.message).not.toContain("sk-abcdef1234567890");
    expect(error.message).toContain("[REDACTED]");
  });
});

describe("error taxonomy", () => {
  it("assigns stable codes and retry flags", () => {
    const cases: Array<[MurphyChainError, string, boolean]> = [
      [new ConfigurationError("x"), "CONFIGURATION_ERROR", false],
      [new ValidationError("x"), "VALIDATION_ERROR", false],
      [new UnsupportedCapabilityError("p", "streaming"), "UNSUPPORTED_CAPABILITY", false],
      [new ProviderAuthError("p", "401"), "PROVIDER_AUTH", false],
      [new ProviderRateLimitError("p", "429"), "PROVIDER_RATE_LIMIT", true],
      [new ProviderBadRequestError("p", "400"), "PROVIDER_BAD_REQUEST", false],
      [new ProviderServerError("p", "500"), "PROVIDER_SERVER_ERROR", true],
      [new ProviderConnectionError("p", "dns"), "PROVIDER_CONNECTION", true],
      [new ProviderTimeoutError("p", "slow"), "PROVIDER_TIMEOUT", true],
      [new ProviderContentFilterError("p", "blocked"), "PROVIDER_CONTENT_FILTER", false],
    ];
    for (const [error, code, retryable] of cases) {
      expect(error.code).toBe(code);
      expect(error.retryable).toBe(retryable);
      expect(error).toBeInstanceOf(MurphyChainError);
    }
  });

  it("ProviderError records provider, status, and requestId in details", () => {
    const error = new ProviderError("openai", "msg", { status: 500, requestId: "req_1" });
    expect(error.provider).toBe("openai");
    expect(error.status).toBe(500);
    expect(error.requestId).toBe("req_1");
    expect(error.details.status).toBe(500);
  });

  it("ProviderRateLimitError keeps retryAfterMs", () => {
    expect(new ProviderRateLimitError("p", "429", { retryAfterMs: 2_000 }).retryAfterMs).toBe(2_000);
  });
});

describe("ensureMurphyChainError", () => {
  it("passes MurphyChainError through unchanged", () => {
    const original = new ValidationError("bad");
    expect(ensureMurphyChainError(original, "fallback")).toBe(original);
  });

  it("wraps unknown errors with INTERNAL_ERROR and preserves the cause", () => {
    const cause = new Error("kaboom");
    const wrapped = ensureMurphyChainError(cause, "Request failed", { provider: "p" });
    expect(wrapped).toBeInstanceOf(MurphyChainError);
    expect(wrapped.code).toBe("INTERNAL_ERROR");
    expect(wrapped.message).toContain("kaboom");
    expect(wrapped.cause).toBe(cause);
    expect(wrapped.details.provider).toBe("p");
  });

  it("wraps non-Error values", () => {
    const wrapped = ensureMurphyChainError("plain string", "Request failed");
    expect(wrapped.message).toContain("plain string");
  });
});
