import { describe, expect, it } from "vitest";
import {
  AI,
  OutputValidationError,
  STRUCTURED_OUTPUT_MAX_ATTEMPTS,
  ValidationError,
  z,
} from "@murphychain/core";
import type { GenerateOptions } from "@murphychain/core";
import { FakeProvider, makeResponse } from "./helpers";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** A provider that replies with a fixed sequence of texts, one per call. */
function scriptedProvider(replies: string[]): FakeProvider {
  let index = 0;
  return new FakeProvider({
    generateImpl: async (options: GenerateOptions) => {
      const text = replies[Math.min(index, replies.length - 1)] ?? "";
      index += 1;
      return makeResponse({
        text,
        provider: "fake",
        model: options.model ?? "fake-1",
        message: { role: "assistant", content: text },
      });
    },
  });
}

const personSchema = z.object({ name: z.string(), age: z.number().int().positive() });

/** The last user message of a request (the JSON instruction or a repair). */
function lastUserMessage(options: GenerateOptions): string {
  const messages = options.messages.filter((message) => message.role === "user");
  return messages[messages.length - 1]?.content ?? "";
}

// ---------------------------------------------------------------------------
// Instruction injection
// ---------------------------------------------------------------------------

describe("structured output instruction", () => {
  it("appends a JSON-only instruction containing the JSON Schema", async () => {
    const provider = scriptedProvider(['{"name":"Ada","age":36}']);
    const ai = new AI({ provider, model: "fake-1" });
    await ai.generateObject("Describe Ada.", personSchema);

    const instruction = lastUserMessage(provider.generateCalls[0]!);
    expect(instruction).toContain("JSON value");
    expect(instruction).toContain('"name"');
    expect(instruction).toContain("no markdown code fences");
    // The original prompt still precedes the instruction.
    expect(provider.generateCalls[0]!.messages[0]).toMatchObject({
      role: "user",
      content: "Describe Ada.",
    });
  });

  it("counts the instruction toward input limits", async () => {
    const provider = scriptedProvider(['{"name":"Ada","age":36}']);
    const ai = new AI({ provider, model: "fake-1", maxInputChars: 100 });
    await expect(ai.generateObject("Describe Ada.", personSchema)).rejects.toThrow(
      ValidationError,
    );
  });

  it("works with message-array input", async () => {
    const provider = scriptedProvider(['{"name":"Ada","age":36}']);
    const ai = new AI({ provider, model: "fake-1" });
    const response = await ai.generateObject(
      [{ role: "user", content: "Describe Ada." }],
      personSchema,
    );
    expect(response.object).toEqual({ name: "Ada", age: 36 });
  });

  it("rejects a non-Zod schema eagerly", async () => {
    const provider = scriptedProvider([]);
    const ai = new AI({ provider, model: "fake-1" });
    await expect(ai.generateObject("x", { type: "object" } as never)).rejects.toThrow(
      ValidationError,
    );
  });

  it("rejects output combined with tools", async () => {
    const provider = scriptedProvider([]);
    const ai = new AI({ provider, model: "fake-1" });
    await expect(
      ai.generate({
        prompt: "x",
        output: personSchema,
        tools: [{ name: "t", description: "d", parameters: z.object({}) }],
      }),
    ).rejects.toThrow(ValidationError);
  });

  it("rejects passing the schema twice to generateObject", async () => {
    const provider = scriptedProvider([]);
    const ai = new AI({ provider, model: "fake-1" });
    await expect(
      ai.generateObject({ prompt: "x", output: personSchema }, personSchema),
    ).rejects.toThrow("not both");
  });
});

// ---------------------------------------------------------------------------
// Parsing & validation success
// ---------------------------------------------------------------------------

describe("structured output success", () => {
  it("parses a plain JSON reply and exposes response.object", async () => {
    const provider = scriptedProvider(['{"name":"Ada","age":36}']);
    const ai = new AI({ provider, model: "fake-1" });
    const response = await ai.generateObject("Describe Ada.", personSchema);
    expect(response.object).toEqual({ name: "Ada", age: 36 });
    expect(response.text).toContain("Ada");
  });

  it("tolerates markdown code fences", async () => {
    const provider = scriptedProvider(['```json\n{"name":"Ada","age":36}\n```']);
    const ai = new AI({ provider, model: "fake-1" });
    const response = await ai.generateObject("Describe Ada.", personSchema);
    expect(response.object).toEqual({ name: "Ada", age: 36 });
  });

  it("tolerates JSON wrapped in surrounding prose", async () => {
    const provider = scriptedProvider([
      'Sure! Here is the person you asked for: {"name":"Ada","age":36} — let me know if you need more.',
    ]);
    const ai = new AI({ provider, model: "fake-1" });
    const response = await ai.generateObject("Describe Ada.", personSchema);
    expect(response.object).toEqual({ name: "Ada", age: 36 });
  });

  it("applies schema transforms to the parsed value", async () => {
    const provider = scriptedProvider(['{"name":"Ada","age":36}']);
    const ai = new AI({ provider, model: "fake-1" });
    const schema = personSchema.extend({ name: z.string().transform((s) => s.toUpperCase()) });
    const response = await ai.generateObject("Describe Ada.", schema);
    expect(response.object).toMatchObject({ name: "ADA" });
  });

  it("works through generate({ output }) as well", async () => {
    const provider = scriptedProvider(['{"name":"Ada","age":36}']);
    const ai = new AI({ provider, model: "fake-1" });
    const response = await ai.generate({ prompt: "Describe Ada.", output: personSchema });
    expect(response.object).toEqual({ name: "Ada", age: 36 });
  });

  it("makes exactly one request when the first reply is valid", async () => {
    const provider = scriptedProvider(['{"name":"Ada","age":36}']);
    const ai = new AI({ provider, model: "fake-1" });
    await ai.generateObject("Describe Ada.", personSchema);
    expect(provider.generateCalls).toHaveLength(1);
  });
});

// ---------------------------------------------------------------------------
// Repair loop
// ---------------------------------------------------------------------------

describe("structured output repair loop", () => {
  it("re-asks with validation errors when the reply has the wrong shape", async () => {
    const provider = scriptedProvider(['{"name":"Ada","age":"thirty-six"}', '{"name":"Ada","age":36}']);
    const ai = new AI({ provider, model: "fake-1" });
    const response = await ai.generateObject("Describe Ada.", personSchema);
    expect(response.object).toEqual({ name: "Ada", age: 36 });
    expect(provider.generateCalls).toHaveLength(2);

    const repair = lastUserMessage(provider.generateCalls[1]!);
    expect(repair).toContain("age");
    expect(repair).toContain("Fix these problems");
    // The failed reply is echoed back as an assistant message before the repair.
    const messages = provider.generateCalls[1]!.messages;
    const assistantEcho = messages.filter((message) => message.role === "assistant").pop();
    expect(assistantEcho?.content).toContain("thirty-six");
  });

  it("re-asks when the reply is not JSON at all", async () => {
    const provider = scriptedProvider(["I am sorry, I cannot help with that.", '{"name":"Ada","age":36}']);
    const ai = new AI({ provider, model: "fake-1" });
    const response = await ai.generateObject("Describe Ada.", personSchema);
    expect(response.object).toEqual({ name: "Ada", age: 36 });
    const repair = lastUserMessage(provider.generateCalls[1]!);
    expect(repair).toContain("not valid JSON");
  });

  it("re-asks when the reply is empty", async () => {
    const provider = scriptedProvider(["", '{"name":"Ada","age":36}']);
    const ai = new AI({ provider, model: "fake-1" });
    const response = await ai.generateObject("Describe Ada.", personSchema);
    expect(response.object).toEqual({ name: "Ada", age: 36 });
  });

  it("accumulates usage and latency across attempts", async () => {
    const provider = scriptedProvider(['{"name":"Ada","age":"old"}', '{"name":"Ada","age":36}']);
    const ai = new AI({ provider, model: "fake-1" });
    const response = await ai.generateObject("Describe Ada.", personSchema);
    // makeResponse reports 3/2/5 per call; two calls must sum.
    expect(response.usage).toEqual({
      promptTokens: 6,
      completionTokens: 4,
      totalTokens: 10,
    });
    expect(response.latencyMs).toBeGreaterThanOrEqual(0);
    expect(ai.stats().requests).toBe(2);
    expect(ai.stats().promptTokens).toBe(6);
  });

  it("counts the instruction only once; repairs build on prior messages", async () => {
    const provider = scriptedProvider([
      'not json',
      'still not json',
      '{"name":"Ada","age":36}',
    ]);
    const ai = new AI({ provider, model: "fake-1" });
    await ai.generateObject("Describe Ada.", personSchema);
    expect(provider.generateCalls).toHaveLength(3);
    // Attempt 3 carries: prompt + instruction + 2×(assistant echo + repair).
    const third = provider.generateCalls[2]!;
    expect(third.messages.filter((m) => m.role === "assistant")).toHaveLength(2);
    // prompt + instruction + 2 repairs are all user messages.
    expect(third.messages.filter((m) => m.role === "user")).toHaveLength(4);
  });
});

// ---------------------------------------------------------------------------
// Exhaustion
// ---------------------------------------------------------------------------

describe("structured output exhaustion", () => {
  it("throws OutputValidationError after the bounded attempts", async () => {
    const provider = scriptedProvider(["no json here", "still no", "nope"]);
    const ai = new AI({ provider, model: "fake-1" });
    try {
      await ai.generateObject("Describe Ada.", personSchema);
      expect.unreachable("should have thrown");
    } catch (error) {
      expect(error).toBeInstanceOf(OutputValidationError);
      const outputError = error as OutputValidationError;
      expect(outputError.code).toBe("OUTPUT_VALIDATION");
      expect(outputError.retryable).toBe(false);
      expect(outputError.attempts).toBe(STRUCTURED_OUTPUT_MAX_ATTEMPTS);
      expect(outputError.issues.length).toBeGreaterThan(0);
      expect(outputError.message).toContain("failed validation");
    }
    expect(provider.generateCalls).toHaveLength(STRUCTURED_OUTPUT_MAX_ATTEMPTS);
  });

  it("reports schema issues from the final attempt", async () => {
    const provider = scriptedProvider(['{"name":"Ada","age":"old"}']);
    const ai = new AI({ provider, model: "fake-1" });
    try {
      await ai.generateObject("Describe Ada.", personSchema);
      expect.unreachable("should have thrown");
    } catch (error) {
      const outputError = error as OutputValidationError;
      expect(outputError.issues.some((issue) => issue.path.includes("age"))).toBe(true);
    }
  });

  it("redacts secrets in the last-reply preview", async () => {
    const provider = scriptedProvider(["key sk-LEAKED_SECRET_VALUE here, no json"]);
    const ai = new AI({ provider, model: "fake-1" });
    try {
      await ai.generateObject("Describe Ada.", personSchema);
      expect.unreachable("should have thrown");
    } catch (error) {
      const outputError = error as OutputValidationError;
      expect(JSON.stringify(outputError.details)).not.toContain("sk-LEAKED_SECRET_VALUE");
    }
  });

  it("counts errors and requests in stats", async () => {
    const provider = scriptedProvider(["no"]);
    const ai = new AI({ provider, model: "fake-1" });
    await expect(ai.generateObject("Describe Ada.", personSchema)).rejects.toThrow();
    expect(ai.stats().requests).toBe(STRUCTURED_OUTPUT_MAX_ATTEMPTS);
  });
});

// ---------------------------------------------------------------------------
// Incompatibilities
// ---------------------------------------------------------------------------

describe("structured output incompatibilities", () => {
  it("rejects stream() with output", async () => {
    const provider = scriptedProvider([]);
    const ai = new AI({ provider, model: "fake-1" });
    await expect(ai.stream({ prompt: "x", output: personSchema })).rejects.toThrow(
      ValidationError,
    );
    await expect(ai.stream({ prompt: "x", output: personSchema })).rejects.toThrow(
      "does not support",
    );
    expect(provider.streamCalls).toHaveLength(0);
  });

  it("propagates provider errors from a repair attempt", async () => {
    let calls = 0;
    const provider = new FakeProvider({
      generateImpl: async () => {
        calls += 1;
        if (calls === 1) return makeResponse({ text: "not json" });
        throw new Error("provider exploded");
      },
    });
    const ai = new AI({ provider, model: "fake-1", retries: 0 });
    await expect(ai.generateObject("Describe Ada.", personSchema)).rejects.toThrow(
      "provider exploded",
    );
  });

  it("respects cancellation between attempts", async () => {
    const controller = new AbortController();
    const provider = new FakeProvider({
      generateImpl: async () => {
        controller.abort();
        return makeResponse({ text: "not json" });
      },
    });
    const ai = new AI({ provider, model: "fake-1" });
    await expect(
      ai.generateObject("Describe Ada.", personSchema, { signal: controller.signal }),
    ).rejects.toThrow();
  });
});
