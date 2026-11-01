import { describe, expect, it, vi } from "vitest";
import { Tool, ValidationError, executeToolCalls, z } from "@murphychain/core";
import type { ToolCall } from "@murphychain/core";

// ---------------------------------------------------------------------------
// Definition & validation
// ---------------------------------------------------------------------------

describe("Tool.define", () => {
  it("defines a tool with name, description, and execute", () => {
    const tool = Tool.define({
      name: "get_weather",
      description: "Get the current weather for a city",
      parameters: z.object({ city: z.string() }),
      execute: ({ city }) => `sunny in ${city}`,
    });
    expect(tool.name).toBe("get_weather");
    expect(tool.description).toContain("weather");
    expect(tool.definition).toMatchObject({ name: "get_weather" });
  });

  it("converts Zod parameters to JSON Schema exactly once (cached)", () => {
    const tool = Tool.define({
      name: "lookup",
      description: "Look up an item",
      parameters: z.object({ id: z.string().min(1) }),
      execute: () => null,
    });
    const first = tool.normalized;
    const second = tool.normalized;
    expect(first).toBe(second); // same cached object
    expect(first.jsonSchema).toMatchObject({ type: "object" });
    expect(first.jsonSchema).not.toHaveProperty("$schema");
  });

  it("rejects a missing execute function", () => {
    expect(() =>
      Tool.define({
        name: "bad",
        description: "",
        parameters: z.object({}),
        // @ts-expect-error -- deliberately invalid for the runtime check
        execute: "not a function",
      }),
    ).toThrow(ValidationError);
  });

  it("rejects an invalid tool name at definition time", () => {
    expect(() =>
      Tool.define({
        name: "not a valid name!",
        description: "x",
        parameters: z.object({}),
        execute: () => null,
      }),
    ).toThrow(ValidationError);
  });

  it("accepts a plain JSON Schema as parameters", () => {
    const tool = Tool.define({
      name: "raw_schema",
      description: "Uses raw JSON Schema",
      parameters: { type: "object", properties: { a: { type: "string" } } },
      execute: () => null,
    });
    expect(tool.normalized.jsonSchema).toMatchObject({ type: "object" });
  });
});

// ---------------------------------------------------------------------------
// Argument parsing
// ---------------------------------------------------------------------------

describe("Tool.parseArguments", () => {
  const tool = Tool.define({
    name: "greet",
    description: "Greets someone",
    parameters: z.object({ name: z.string().min(1), excited: z.boolean().optional() }),
    execute: ({ name, excited }) => `Hi ${name}${excited ? "!" : "."}`,
  });

  it("parses valid arguments", () => {
    expect(tool.parseArguments({ name: "Ada" })).toEqual({ name: "Ada" });
  });

  it("throws ValidationError with tool name and issues on bad arguments", () => {
    try {
      tool.parseArguments({ name: 42 });
      expect.unreachable("should have thrown");
    } catch (error) {
      expect(error).toBeInstanceOf(ValidationError);
      const validation = error as ValidationError;
      expect(validation.message).toContain('Tool "greet"');
      expect(validation.details?.["tool"]).toBe("greet");
      expect(Array.isArray(validation.details?.["issues"])).toBe(true);
    }
  });

  it("rejects non-object arguments for JSON Schema tools", () => {
    const rawTool = Tool.define({
      name: "raw",
      description: "raw",
      parameters: { type: "object", properties: {} },
      execute: () => null,
    });
    expect(() => rawTool.parseArguments("oops")).toThrow(ValidationError);
    expect(() => rawTool.parseArguments([1, 2])).toThrow(ValidationError);
  });
});

// ---------------------------------------------------------------------------
// Execution & result serialization
// ---------------------------------------------------------------------------

describe("Tool.run", () => {
  it("parses arguments and awaits execute", async () => {
    const tool = Tool.define({
      name: "add",
      description: "Adds two numbers",
      parameters: z.object({ a: z.number(), b: z.number() }),
      execute: async ({ a, b }) => a + b,
    });
    await expect(tool.run({ a: 2, b: 3 })).resolves.toBe(5);
  });

  it("propagates execute failures to the caller", async () => {
    const tool = Tool.define({
      name: "boom",
      description: "Throws",
      parameters: z.object({}),
      execute: () => {
        throw new Error("kaboom");
      },
    });
    await expect(tool.run({})).rejects.toThrow("kaboom");
  });

  it("forwards the cancellation signal to execute", async () => {
    const controller = new AbortController();
    let seen: AbortSignal | undefined;
    const tool = Tool.define({
      name: "watch",
      description: "Receives signal",
      parameters: z.object({}),
      execute: (_args, options) => {
        seen = options.signal;
      },
    });
    await tool.run({}, { signal: controller.signal });
    expect(seen).toBe(controller.signal);
  });
});

describe("Tool.serializeResult", () => {
  const tool = Tool.define({
    name: "t",
    description: "t",
    parameters: z.object({}),
    execute: () => null,
  });

  it("passes strings through", () => {
    expect(tool.serializeResult("plain text")).toBe("plain text");
  });

  it("JSON-stringifies objects and arrays", () => {
    expect(tool.serializeResult({ a: 1 })).toBe('{"a":1}');
    expect(tool.serializeResult([1, "x"])).toBe('[1,"x"]');
  });

  it('serializes undefined as "null"', () => {
    expect(tool.serializeResult(undefined)).toBe("null");
  });

  it("survives circular results", () => {
    const circular: Record<string, unknown> = {};
    circular["self"] = circular;
    expect(typeof tool.serializeResult(circular)).toBe("string");
  });
});

// ---------------------------------------------------------------------------
// executeToolCalls
// ---------------------------------------------------------------------------

function call(id: string, name: string, args: Record<string, unknown>): ToolCall {
  return { id, name, arguments: args };
}

const weather = Tool.define({
  name: "get_weather",
  description: "Weather lookup",
  parameters: z.object({ city: z.string() }),
  execute: ({ city }) => ({ tempC: 21, city }),
});

const slower = Tool.define({
  name: "slow_tool",
  description: "Runs after get_weather",
  parameters: z.object({}),
  execute: () => "done",
});

describe("executeToolCalls", () => {
  it("executes calls in order and returns role:tool messages", async () => {
    const messages = await executeToolCalls(
      [weather, slower],
      [call("c1", "get_weather", { city: "Oslo" }), call("c2", "slow_tool", {})],
    );
    expect(messages).toHaveLength(2);
    expect(messages[0]).toMatchObject({
      role: "tool",
      name: "get_weather",
      toolCallId: "c1",
      content: '{"tempC":21,"city":"Oslo"}',
    });
    expect(messages[1]).toMatchObject({ role: "tool", name: "slow_tool", toolCallId: "c2" });
  });

  it("serializes failures as an error object for model self-recovery by default", async () => {
    const messages = await executeToolCalls(
      [weather],
      [call("c1", "get_weather", { city: 42 }), call("c2", "missing_tool", {})],
    );
    expect(messages).toHaveLength(2);
    const invalidArgs = JSON.parse(messages[0]!.content) as { error?: string };
    expect(invalidArgs.error).toContain('Tool "get_weather"');
    const unknown = JSON.parse(messages[1]!.content) as { error?: string };
    expect(unknown.error).toContain("Unknown tool");
    expect(unknown.error).toContain("get_weather"); // lists available tools
  });

  it("redacts secrets from serialized failures", async () => {
    const leaky = Tool.define({
      name: "leaky",
      description: "Leaks",
      parameters: z.object({}),
      execute: () => {
        throw new Error("failed with key sk-abc123XYZsecret and token ghp_SECRETVALUE123");
      },
    });
    const messages = await executeToolCalls([leaky], [call("c1", "leaky", {})]);
    const parsed = JSON.parse(messages[0]!.content) as { error?: string };
    expect(parsed.error).not.toContain("sk-abc123XYZsecret");
    expect(parsed.error).not.toContain("ghp_SECRETVALUE123");
  });

  it('throws when onError is "throw"', async () => {
    await expect(
      executeToolCalls(
        [weather],
        [call("c1", "get_weather", { city: 42 })],
        { onError: "throw" },
      ),
    ).rejects.toThrow(ValidationError);
  });

  it("executes tools sequentially (no interleaving)", async () => {
    const order: string[] = [];
    const first = Tool.define({
      name: "first",
      description: "first",
      parameters: z.object({}),
      execute: async () => {
        order.push("first-start");
        await new Promise((resolve) => setTimeout(resolve, 5));
        order.push("first-end");
        return "ok";
      },
    });
    const second = Tool.define({
      name: "second",
      description: "second",
      parameters: z.object({}),
      execute: () => {
        order.push("second");
        return "ok";
      },
    });
    await executeToolCalls(
      [first, second],
      [call("c1", "first", {}), call("c2", "second", {})],
    );
    expect(order).toEqual(["first-start", "first-end", "second"]);
  });

  it("forwards the signal to every tool", async () => {
    const controller = new AbortController();
    const seen: Array<AbortSignal | undefined> = [];
    const spy = Tool.define({
      name: "spy",
      description: "records signal",
      parameters: z.object({}),
      execute: (_args, options) => {
        seen.push(options.signal);
      },
    });
    await executeToolCalls(
      [spy],
      [call("c1", "spy", {}), call("c2", "spy", {})],
      { signal: controller.signal },
    );
    expect(seen).toEqual([controller.signal, controller.signal]);
  });

  it("returns an empty list when there are no calls", async () => {
    await expect(executeToolCalls([weather], [])).resolves.toEqual([]);
  });

  it("does not execute any code strings from the model (data only)", async () => {
    const execute = vi.fn(() => "ran");
    const tool = Tool.define({
      name: "only_this",
      description: "x",
      parameters: z.object({}),
      execute,
    });
    // Model tries to invoke a tool by string name — dispatch only runs
    // registered tools with parsed arguments; nothing is eval'd.
    const messages = await executeToolCalls([tool], [call("c1", "only_this", {})]);
    expect(execute).toHaveBeenCalledTimes(1);
    expect(messages[0]!.content).toBe("ran"); // strings pass through, not eval'd
  });
});
