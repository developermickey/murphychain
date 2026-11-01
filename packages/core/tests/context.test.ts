import { describe, expect, it } from "vitest";
import {
  ExecutionCancelledError,
  ExecutionTimeoutError,
} from "@murphychain/core";
import { ExecutionContext } from "../src/Context";

describe("ExecutionContext", () => {
  it("generates a req_-prefixed request id and records basics", () => {
    const ctx = new ExecutionContext({ provider: "p", model: "m" });
    expect(ctx.requestId).toMatch(/^req_[0-9a-f-]{36}$/);
    expect(ctx.provider).toBe("p");
    expect(ctx.model).toBe("m");
    expect(ctx.aborted).toBe(false);
    expect(ctx.reason).toBeUndefined();
    expect(ctx.elapsedMs).toBeGreaterThanOrEqual(0);
  });

  it("freezes a copy of the metadata", () => {
    const metadata = { a: 1 };
    const ctx = new ExecutionContext({ provider: "p", model: "m", metadata });
    metadata.a = 2; // mutating the source must not affect the context
    expect(ctx.metadata.a).toBe(1);
    expect(Object.isFrozen(ctx.metadata)).toBe(true);
  });

  it("has no deadline when timeoutMs is 0 or omitted", () => {
    const ctx = new ExecutionContext({ provider: "p", model: "m", timeoutMs: 0 });
    expect(ctx.deadline).toBeUndefined();
    expect(ctx.remainingMs()).toBeUndefined();
  });

  it("aborts with reason 'timeout' and a timeout error when the deadline fires", async () => {
    const ctx = new ExecutionContext({ provider: "p", model: "m", timeoutMs: 20 });
    expect(ctx.deadline).toBeGreaterThan(Date.now());
    expect(ctx.remainingMs()).toBeLessThanOrEqual(20);
    await new Promise((resolve) => setTimeout(resolve, 40));
    expect(ctx.aborted).toBe(true);
    expect(ctx.reason).toBe("timeout");
    const error = ctx.abortError();
    expect(error).toBeInstanceOf(ExecutionTimeoutError);
    expect(error.message).toContain("20ms");
    expect(() => ctx.throwIfAborted()).toThrowError(ExecutionTimeoutError);
  });

  it("aborts with reason 'user' when the external signal fires", async () => {
    const controller = new AbortController();
    const ctx = new ExecutionContext({ provider: "p", model: "m", signal: controller.signal });
    controller.abort();
    expect(ctx.reason).toBe("user");
    expect(ctx.abortError()).toBeInstanceOf(ExecutionCancelledError);
  });

  it("is immediately aborted for an already-aborted external signal", () => {
    const controller = new AbortController();
    controller.abort();
    const ctx = new ExecutionContext({ provider: "p", model: "m", signal: controller.signal });
    expect(ctx.aborted).toBe(true);
    expect(ctx.reason).toBe("user");
  });

  it("cancel() aborts as a user cancellation", () => {
    const ctx = new ExecutionContext({ provider: "p", model: "m" });
    ctx.cancel();
    expect(ctx.reason).toBe("user");
  });

  it("keeps the first abort reason (abort is one-shot)", () => {
    const ctx = new ExecutionContext({ provider: "p", model: "m" });
    ctx.abort("user");
    ctx.abort("timeout");
    expect(ctx.reason).toBe("user");
  });

  it("clearDeadline() stops the timer so no late abort fires", async () => {
    const ctx = new ExecutionContext({ provider: "p", model: "m", timeoutMs: 20 });
    ctx.clearDeadline();
    expect(ctx.deadline).toBeUndefined();
    await new Promise((resolve) => setTimeout(resolve, 40));
    expect(ctx.aborted).toBe(false);
  });

  it("dispose() is idempotent and releases the timer", async () => {
    const ctx = new ExecutionContext({ provider: "p", model: "m", timeoutMs: 20 });
    ctx.dispose();
    ctx.dispose();
    await new Promise((resolve) => setTimeout(resolve, 40));
    expect(ctx.aborted).toBe(false);
  });

  it("does not accumulate external listeners across aborts", () => {
    const controller = new AbortController();
    const ctx = new ExecutionContext({ provider: "p", model: "m", signal: controller.signal });
    ctx.cancel(); // aborts internally; must unlink the external listener
    controller.abort(); // must be a no-op for this context
    expect(ctx.reason).toBe("user");
  });
});
