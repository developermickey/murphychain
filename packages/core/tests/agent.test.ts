import { describe, expect, it } from "vitest";
import { z } from "zod";
import { AI, Agent, BufferMemory, ExecutionCancelledError, Tool, ValidationError } from "@murphychain/core";
import type { AIResponse, Message } from "@murphychain/core";
import { FakeProvider, makeResponse } from "./helpers";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

function makeAI(provider: FakeProvider): AI {
  return new AI({ provider, model: "fake-1", retries: 0 });
}

/** Provider whose generateImpl replies with the scripted response per step. */
function scripted(script: (step: number) => AIResponse): FakeProvider {
  let step = 0;
  return new FakeProvider({ generateImpl: async () => script(step++) });
}

function assistant(text: string): AIResponse {
  return makeResponse({ text, message: { role: "assistant", content: text } });
}

function toolAssistant(
  name: string,
  args: Record<string, unknown>,
  id = "call_1",
): AIResponse {
  return makeResponse({
    text: "",
    finishReason: "tool_calls",
    message: {
      role: "assistant",
      content: "",
      toolCalls: [{ id, name, arguments: args }],
    },
  });
}

const calculator = Tool.define({
  name: "calculator",
  description: "Adds two numbers.",
  parameters: z.object({ a: z.number(), b: z.number() }),
  execute: async ({ a, b }) => a + b,
});

// ---------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------

describe("Agent loop", () => {
  it("answers directly without executing anything when the model returns text", async () => {
    const provider = scripted(() => assistant("42"));
    const agent = new Agent({ ai: makeAI(provider), name: "direct" });

    const result = await agent.run("What is the answer?");

    expect(result.response.text).toBe("42");
    expect(result.stopped).toBe("completed");
    expect(result.steps).toHaveLength(1);
    expect(result.usage.totalTokens).toBe(5);
    // No tools → no tools in the request.
    expect(provider.generateCalls[0]!.tools).toBeUndefined();
  });

  it("executes a requested tool and feeds the result back in the next step", async () => {
    const provider = scripted((step) =>
      step === 0 ? toolAssistant("calculator", { a: 2, b: 3 }) : assistant("2 + 3 = 5"),
    );
    const agent = new Agent({ ai: makeAI(provider), name: "math", tools: [calculator] });

    const result = await agent.run("What is 2 + 3?");

    expect(result.response.text).toBe("2 + 3 = 5");
    expect(result.stopped).toBe("completed");
    expect(result.steps).toHaveLength(2);
    expect(result.steps[0]!.toolCalls).toHaveLength(1);
    expect(result.steps[0]!.toolMessages[0]).toEqual({
      role: "tool",
      name: "calculator",
      toolCallId: "call_1",
      content: "5",
    });
    expect(result.steps[1]!.toolMessages).toHaveLength(0);

    // The follow-up request carries the full exchange: user → assistant+call → tool.
    const followUp = provider.generateCalls[1]!.messages;
    expect(followUp.map((m) => m.role)).toEqual(["user", "assistant", "tool"]);
    expect(followUp[2]!.content).toBe("5");
    // Tool definitions were sent on both steps.
    expect(provider.generateCalls[0]!.tools).toHaveLength(1);
    expect(provider.generateCalls[1]!.tools).toHaveLength(1);
  });

  it("stops at maxSteps and does NOT execute the final step's tool calls", async () => {
    let executed = 0;
    const loopTool = Tool.define({
      name: "loop",
      description: "Loops forever.",
      parameters: z.object({ x: z.number() }),
      execute: async () => {
        executed += 1;
        return "again";
      },
    });
    const provider = scripted(() => toolAssistant("loop", { x: 1 }));
    const agent = new Agent({
      ai: makeAI(provider),
      tools: [loopTool],
      maxSteps: 3,
      name: "looper",
    });

    const result = await agent.run("keep going");

    expect(result.stopped).toBe("max_steps");
    expect(result.steps).toHaveLength(3);
    expect(provider.generateCalls).toHaveLength(3);
    // Steps 0 and 1 executed the tool; step 2's call was never executed.
    expect(executed).toBe(2);
    expect(result.steps[2]!.toolMessages).toHaveLength(0);
  });

  it("links the caller's signal into every model call", async () => {
    const controller = new AbortController();
    let seen: AbortSignal | undefined;
    const provider = new FakeProvider({
      generateImpl: async (options) => {
        seen = options.signal;
        // Abort while the request is in flight: the linked signal must follow.
        controller.abort();
        return assistant("ok");
      },
    });
    const agent = new Agent({ ai: makeAI(provider) });

    // The provider ignores the abort and answers anyway; the engine maps the
    // cancellation. Either way, the caller's signal reached the provider.
    await agent.run("hi", { signal: controller.signal }).catch((error: unknown) => {
      expect(error).toBeInstanceOf(ExecutionCancelledError);
    });

    expect(seen).toBeDefined();
    expect(seen!.aborted).toBe(true);
  });

  it("applies the per-run timeoutMs as a deadline on model calls", async () => {
    // A provider that hangs forever: only the engine's deadline can end this.
    const provider = new FakeProvider({
      generateImpl: () => new Promise<AIResponse>(() => {}),
    });
    const agent = new Agent({ ai: makeAI(provider) });

    await expect(agent.run("hi", { timeoutMs: 30 })).rejects.toThrow(/timed out after 30ms/);
  });

  it("sums usage across steps", async () => {
    const provider = scripted((step) =>
      step === 0 ? toolAssistant("calculator", { a: 1, b: 1 }) : assistant("2"),
    );
    const agent = new Agent({ ai: makeAI(provider), tools: [calculator] });

    const result = await agent.run("1 + 1?");

    // Two calls, each 3+2+5 tokens (FakeProvider usage).
    expect(result.usage).toEqual({
      promptTokens: 6,
      completionTokens: 4,
      totalTokens: 10,
    });
  });

  it("sends the system prompt first, and never stores it in memory", async () => {
    const provider = scripted(() => assistant("done"));
    const memory = new BufferMemory();
    const agent = new Agent({
      ai: makeAI(provider),
      system: "You are terse.",
      memory,
      name: "sys",
    });

    const result = await agent.run("hi");

    expect(result.messages[0]).toEqual({ role: "system", content: "You are terse." });
    // Memory holds the executed conversation only.
    expect(memory.messages.map((m) => m.role)).toEqual(["user", "assistant"]);
    expect(memory.messages.some((m) => m.role === "system")).toBe(false);
  });

  it("emits step, tool, and finish events (content-free)", async () => {
    const provider = scripted((step) =>
      step === 0 ? toolAssistant("calculator", { a: 4, b: 5 }) : assistant("9"),
    );
    const agent = new Agent({ ai: makeAI(provider), tools: [calculator], name: "ev" });
    const steps: string[] = [];
    const tools: Array<{ name: string; ok: boolean }> = [];
    const finishes: Array<{ steps: number; stopped: string }> = [];

    agent.on("step", (event) => steps.push(`${event.index}:${event.toolCalls}`));
    agent.on("tool", (event) => tools.push({ name: event.name, ok: event.ok }));
    agent.on("finish", (event) => finishes.push({ steps: event.steps, stopped: event.stopped }));

    await agent.run("4 + 5?");

    expect(steps).toEqual(["0:1", "1:0"]);
    expect(tools).toEqual([{ name: "calculator", ok: true }]);
    expect(finishes).toEqual([{ steps: 2, stopped: "completed" }]);
  });

  it("cancels between steps when the signal aborts during tool execution", async () => {
    const controller = new AbortController();
    const abortingTool = Tool.define({
      name: "aborter",
      description: "Aborts the run.",
      parameters: z.object({}),
      execute: async () => {
        controller.abort();
        return "done";
      },
    });
    const provider = scripted(() => toolAssistant("aborter", {}));
    const agent = new Agent({ ai: makeAI(provider), tools: [abortingTool] });

    await expect(agent.run("go", { signal: controller.signal })).rejects.toThrow(
      ExecutionCancelledError,
    );
    // Cancelled between steps → the second model call never happened.
    expect(provider.generateCalls).toHaveLength(1);
  });

  it("rejects concurrent runs on the same agent (single-flight)", async () => {
    const provider = scripted(() => assistant("ok"));
    const agent = new Agent({ ai: makeAI(provider) });

    const first = agent.run("first");
    await expect(agent.run("second")).rejects.toThrow(ValidationError);
    await first;
  });

  it("forwards tool errors as results by default so the model can recover", async () => {
    const failing = Tool.define({
      name: "boom",
      description: "Always fails.",
      parameters: z.object({}),
      execute: async () => {
        throw new Error("kaboom");
      },
    });
    const provider = scripted((step) =>
      step === 0 ? toolAssistant("boom", {}) : assistant("recovered"),
    );
    const agent = new Agent({ ai: makeAI(provider), tools: [failing] });

    const result = await agent.run("try");

    expect(result.response.text).toBe("recovered");
    const toolMessage = result.steps[0]!.toolMessages[0]!;
    expect(toolMessage.role).toBe("tool");
    expect(toolMessage.content).toContain('"error"');
    expect(toolMessage.content).toContain("kaboom");
  });

  it("throws tool errors immediately with onToolError: throw", async () => {
    const failing = Tool.define({
      name: "boom",
      description: "Always fails.",
      parameters: z.object({}),
      execute: async () => {
        throw new Error("kaboom");
      },
    });
    const provider = scripted(() => toolAssistant("boom", {}));
    const agent = new Agent({
      ai: makeAI(provider),
      tools: [failing],
      onToolError: "throw",
    });

    await expect(agent.run("try")).rejects.toThrow("kaboom");
  });
});

// ---------------------------------------------------------------------------
// Memory integration
// ---------------------------------------------------------------------------

describe("Agent memory", () => {
  it("replays remembered history on the next run", async () => {
    const provider = scripted(() => assistant("noted"));
    const memory = new BufferMemory();
    const agent = new Agent({ ai: makeAI(provider), memory, name: "rememberer" });

    await agent.run("my name is Alice");
    await agent.run("remember anything?");

    expect(provider.generateCalls).toHaveLength(2);
    const second = provider.generateCalls[1]!.messages;
    // user: my name is Alice / assistant: noted / user: remember anything?
    expect(second.map((m) => m.role)).toEqual(["user", "assistant", "user"]);
    expect(second[0]!.content).toBe("my name is Alice");
    expect(second[2]!.content).toBe("remember anything?");
  });

  it("persists only the executed conversation when the final step requested tools", async () => {
    const provider = scripted(() => toolAssistant("calculator", { a: 1, b: 1 }));
    const memory = new BufferMemory();
    const agent = new Agent({
      ai: makeAI(provider),
      tools: [calculator],
      memory,
      maxSteps: 1,
    });

    await agent.run("1 + 1?");

    // Memory must not contain the trailing unanswered assistant tool-call
    // message — it would break provider message validation on the next run.
    expect(memory.messages.map((m) => m.role)).toEqual(["user"]);
    expect(memory.messages[0]!.toolCalls).toBeUndefined();
  });

  it("leaves memory untouched when the run is cancelled", async () => {
    const controller = new AbortController();
    const abortingTool = Tool.define({
      name: "aborter",
      description: "Aborts the run.",
      parameters: z.object({}),
      execute: async () => {
        controller.abort();
        return "done";
      },
    });
    const provider = scripted(() => toolAssistant("aborter", {}));
    const memory = new BufferMemory();
    const agent = new Agent({ ai: makeAI(provider), tools: [abortingTool], memory });

    await expect(
      agent.run("go", { signal: controller.signal }),
    ).rejects.toThrow(ExecutionCancelledError);

    expect(memory.messages).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Configuration & input validation
// ---------------------------------------------------------------------------

describe("Agent validation", () => {
  function bareAI(): AI {
    return new AI({ provider: new FakeProvider(), model: "fake-1", retries: 0 });
  }

  it("requires an AI instance", () => {
    expect(() => new Agent({ ai: {} as unknown as AI })).toThrow(ValidationError);
  });

  it("rejects duplicate tool names", () => {
    expect(() => new Agent({ ai: bareAI(), tools: [calculator, calculator] })).toThrow(
      /Duplicate tool name/,
    );
  });

  it("rejects non-Tool entries in tools", () => {
    expect(
      () => new Agent({ ai: bareAI(), tools: [{ name: "fake" } as unknown as Tool] }),
    ).toThrow(/Tool\.define/);
  });

  it("validates maxSteps", () => {
    expect(() => new Agent({ ai: bareAI(), maxSteps: 0 })).toThrow(ValidationError);
    expect(() => new Agent({ ai: bareAI(), maxSteps: 2.5 })).toThrow(ValidationError);
  });

  it("requires a Memory instance", () => {
    expect(
      () => new Agent({ ai: bareAI(), memory: {} as unknown as BufferMemory }),
    ).toThrow(/Memory instance/);
  });

  it("validates onToolError", () => {
    expect(
      () => new Agent({ ai: bareAI(), onToolError: "explode" as unknown as "throw" }),
    ).toThrow(ValidationError);
  });

  it("rejects an empty prompt", async () => {
    const provider = new FakeProvider();
    const agent = new Agent({ ai: makeAI(provider) });
    await expect(agent.run("")).rejects.toThrow(ValidationError);
    expect(provider.generateCalls).toHaveLength(0);
  });

  it("rejects invalid message input", async () => {
    const provider = new FakeProvider();
    const agent = new Agent({ ai: makeAI(provider) });
    const bad = [{ role: "narrator", content: "hello" }] as unknown as Message[];
    await expect(agent.run(bad)).rejects.toThrow(ValidationError);
    expect(provider.generateCalls).toHaveLength(0);
  });

  it("accepts a message array as input", async () => {
    const provider = scripted(() => assistant("hi"));
    const agent = new Agent({ ai: makeAI(provider) });

    const result = await agent.run([{ role: "user", content: "hello" }]);

    expect(result.response.text).toBe("hi");
    expect(provider.generateCalls[0]!.messages[0]!.content).toBe("hello");
  });
});
