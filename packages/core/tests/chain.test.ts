import { describe, expect, it } from "vitest";
import {
  Chain,
  ChainStepError,
  ExecutionCancelledError,
  ValidationError,
} from "@murphychain/core";
import type { ChainStepContext } from "@murphychain/core";

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

interface State {
  topic?: string | undefined;
  summary?: string | undefined;
  uppercase?: string | undefined;
  [key: string]: unknown;
}

describe("Chain.step registration", () => {
  it("registers steps in order and exposes names/length", () => {
    const chain = new Chain<State>()
      .step("draft", () => ({}))
      .step("polish", () => ({}));
    expect(chain.stepNames).toEqual(["draft", "polish"]);
    expect(chain.length).toBe(2);
  });

  it("is chainable and returns the same instance", () => {
    const chain = new Chain<State>();
    expect(chain.step("a", () => ({}))).toBe(chain);
  });

  it("rejects an empty step name", () => {
    expect(() => new Chain<State>().step("  ", () => ({}))).toThrow(ValidationError);
  });

  it("rejects a non-function step", () => {
    // @ts-expect-error -- deliberately invalid for the runtime check
    expect(() => new Chain<State>().step("bad", "not a function")).toThrow(ValidationError);
  });

  it("rejects duplicate step names", () => {
    const chain = new Chain<State>().step("dup", () => ({}));
    expect(() => chain.step("dup", () => ({}))).toThrow(ValidationError);
    expect(() => chain.step("dup", () => ({}))).toThrow("already registered");
  });
});

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

describe("Chain.run", () => {
  it("runs steps sequentially, threading state", async () => {
    const chain = new Chain<State>()
      .step("draft", (state) => ({ summary: `summary of ${state.topic}` }))
      .step("polish", (state) => ({ uppercase: state.summary?.toUpperCase() }));

    const { state } = await chain.run({ topic: "gravity" });
    expect(state).toEqual({
      topic: "gravity",
      summary: "summary of gravity",
      uppercase: "SUMMARY OF GRAVITY",
    });
  });

  it("passes step name and index in the context", async () => {
    const seen: ChainStepContext[] = [];
    const chain = new Chain<State>()
      .step("a", (_state, ctx) => {
        seen.push(ctx);
        return {};
      })
      .step("b", (_state, ctx) => {
        seen.push(ctx);
        return {};
      });
    await chain.run({});
    expect(seen.map((ctx) => [ctx.name, ctx.index])).toEqual([
      ["a", 0],
      ["b", 1],
    ]);
  });

  it("records a timing trace per executed step", async () => {
    const chain = new Chain<State>().step("fast", () => ({}));
    const { steps } = await chain.run({});
    expect(steps).toHaveLength(1);
    expect(steps[0]).toMatchObject({ name: "fast", index: 0 });
    expect(steps[0]!.durationMs).toBeGreaterThanOrEqual(0);
  });

  it("supports async steps", async () => {
    const chain = new Chain<State>().step("slow", async () => {
      await new Promise((resolve) => setTimeout(resolve, 5));
      return { summary: "done" };
    });
    const { state } = await chain.run({});
    expect(state.summary).toBe("done");
  });

  it("does not mutate the initial state object", async () => {
    const initial: State = { topic: "x" };
    const chain = new Chain<State>().step("add", () => ({ summary: "y" }));
    await chain.run(initial);
    expect(initial).toEqual({ topic: "x" });
  });

  it("is reusable across runs (no shared state)", async () => {
    let counter = 0;
    const chain = new Chain<State>().step("count", () => ({ summary: `run-${++counter}` }));
    const first = await chain.run({});
    const second = await chain.run({});
    expect(first.state.summary).toBe("run-1");
    expect(second.state.summary).toBe("run-2");
  });

  it("rejects a non-object initial state", async () => {
    const chain = new Chain<State>().step("a", () => ({}));
    // @ts-expect-error -- deliberately invalid for the runtime check
    await expect(chain.run("nope")).rejects.toThrow(ValidationError);
  });

  it("rejects a chain with no steps", async () => {
    await expect(new Chain<State>().run({})).rejects.toThrow(ValidationError);
    await expect(new Chain<State>().run({})).rejects.toThrow("no steps");
  });

  it("rejects a step that returns a non-object", async () => {
    const chain = new Chain<State>().step("bad", () => 42 as unknown as State);
    await expect(chain.run({})).rejects.toThrow(ValidationError);
    await expect(chain.run({})).rejects.toThrow('Chain step "bad" must return state');
  });
});

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

describe("Chain error handling", () => {
  it('throws ChainStepError naming the step (onError: "throw", the default)', async () => {
    const chain = new Chain<State>()
      .step("ok", () => ({ summary: "fine" }))
      .step("boom", () => {
        throw new Error("step exploded");
      });
    try {
      await chain.run({});
      expect.unreachable("should have thrown");
    } catch (error) {
      expect(error).toBeInstanceOf(ChainStepError);
      const stepError = error as ChainStepError;
      expect(stepError.step).toBe("boom");
      expect(stepError.index).toBe(1);
      expect(stepError.code).toBe("CHAIN_STEP_ERROR");
      expect(stepError.retryable).toBe(false);
      expect(stepError.message).toContain("step exploded");
    }
  });

  it("stops the chain at the first failure", async () => {
    const ran: string[] = [];
    const chain = new Chain<State>()
      .step("boom", () => {
        throw new Error("stop here");
      })
      .step("never", () => {
        ran.push("never");
        return {};
      });
    await expect(chain.run({})).rejects.toThrow(ChainStepError);
    expect(ran).toEqual([]);
  });

  it('continues past failures with onError: "continue" and reports them at the end', async () => {
    const ran: string[] = [];
    const chain = new Chain<State>({ onError: "continue" })
      .step("boom", () => {
        throw new Error("first failure");
      })
      .step("survives", () => {
        ran.push("survives");
        return { summary: "kept going" };
      })
      .step("boom2", () => {
        throw new Error("second failure");
      });

    try {
      await chain.run({});
      expect.unreachable("should have thrown");
    } catch (error) {
      expect(error).toBeInstanceOf(ChainStepError);
      const stepError = error as ChainStepError;
      expect(stepError.message).toContain("boom, boom2");
      expect(stepError.details?.["failedSteps"]).toEqual(["boom", "boom2"]);
    }
    expect(ran).toEqual(["survives"]);
  });

  it('onError: "continue" returns state normally when every step succeeds', async () => {
    const chain = new Chain<State>({ onError: "continue" }).step("a", () => ({ summary: "ok" }));
    const { state } = await chain.run({});
    expect(state.summary).toBe("ok");
  });

  it("does not wrap an already-wrapped ChainStepError", async () => {
    const inner = new Chain<State>().step("inner", () => {
      throw new Error("inner failure");
    });
    const outer = new Chain<State>().step("outer", async () => {
      await inner.run({});
      return {};
    });
    try {
      await outer.run({});
      expect.unreachable("should have thrown");
    } catch (error) {
      // The original ChainStepError surfaces unchanged (with its own step info).
      expect(error).toBeInstanceOf(ChainStepError);
      expect((error as ChainStepError).step).toBe("inner");
    }
  });
});

// ---------------------------------------------------------------------------
// Cancellation
// ---------------------------------------------------------------------------

describe("Chain cancellation", () => {
  it("throws ExecutionCancelledError before a step when already aborted", async () => {
    const controller = new AbortController();
    controller.abort();
    const chain = new Chain<State>().step("a", () => ({}));
    await expect(chain.run({}, { signal: controller.signal })).rejects.toThrow(
      ExecutionCancelledError,
    );
  });

  it("stops between steps when aborted mid-run", async () => {
    const controller = new AbortController();
    const ran: string[] = [];
    const chain = new Chain<State>()
      .step("first", () => {
        ran.push("first");
        controller.abort();
        return {};
      })
      .step("second", () => {
        ran.push("second");
        return {};
      });
    await expect(chain.run({}, { signal: controller.signal })).rejects.toThrow(
      ExecutionCancelledError,
    );
    expect(ran).toEqual(["first"]);
  });

  it("surfaces cancellation from inside a step", async () => {
    const controller = new AbortController();
    const chain = new Chain<State>().step("self-cancel", (_state, ctx) => {
      controller.abort();
      ctx.signal?.throwIfAborted();
      return {};
    });
    await expect(chain.run({}, { signal: controller.signal })).rejects.toThrow(
      ExecutionCancelledError,
    );
  });

  it("passes the run signal into every step context", async () => {
    const controller = new AbortController();
    const seen: Array<AbortSignal | undefined> = [];
    const chain = new Chain<State>()
      .step("a", (_state, ctx) => {
        seen.push(ctx.signal);
        return {};
      })
      .step("b", (_state, ctx) => {
        seen.push(ctx.signal);
        return {};
      });
    await chain.run({}, { signal: controller.signal });
    expect(seen).toEqual([controller.signal, controller.signal]);
  });

  it("omits the signal from context when none was provided", async () => {
    let seen: AbortSignal | undefined = new AbortController().signal; // sentinel
    const chain = new Chain<State>().step("a", (_state, ctx) => {
      seen = ctx.signal;
      return {};
    });
    await chain.run({});
    expect(seen).toBeUndefined();
  });
});
