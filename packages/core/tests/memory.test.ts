import { describe, expect, it } from "vitest";
import { z } from "zod";
import { AI, BufferMemory, SummarizingMemory, Tool, ValidationError } from "@murphychain/core";
import type { Message } from "@murphychain/core";
import { FakeProvider, makeResponse } from "./helpers";

// ---------------------------------------------------------------------------
// Fixtures
// ---------------------------------------------------------------------------

function user(content: string): Message {
  return { role: "user", content };
}

function assistantMsg(content: string): Message {
  return { role: "assistant", content };
}

function makeAI(provider: FakeProvider = new FakeProvider()): AI {
  return new AI({ provider, model: "fake-1", retries: 0 });
}

/** An assistant message that requested one tool call (start of an atomic unit). */
function toolRequest(id: string, name = "calculator"): Message {
  return {
    role: "assistant",
    content: "",
    toolCalls: [{ id, name, arguments: { a: 1 } }],
  };
}

function toolResult(id: string, name = "calculator"): Message {
  return { role: "tool", name, toolCallId: id, content: "1" };
}

// ---------------------------------------------------------------------------
// BufferMemory
// ---------------------------------------------------------------------------

describe("BufferMemory", () => {
  it("starts empty", () => {
    expect(new BufferMemory().messages).toEqual([]);
  });

  it("remembers messages in order", async () => {
    const memory = new BufferMemory();
    await memory.remember([user("a"), assistantMsg("b")]);
    await memory.remember([user("c")]);
    expect(memory.messages.map((m) => m.content)).toEqual(["a", "b", "c"]);
  });

  it("ignores an empty remember()", async () => {
    const memory = new BufferMemory();
    await memory.remember([]);
    expect(memory.messages).toEqual([]);
  });

  it("evicts the oldest messages beyond maxMessages", async () => {
    const memory = new BufferMemory({ maxMessages: 3 });
    await memory.remember([user("1"), user("2"), user("3"), user("4"), user("5")]);
    expect(memory.messages.map((m) => m.content)).toEqual(["3", "4", "5"]);
  });

  it("evicts assistant tool-call units atomically (never orphans tool results)", async () => {
    const memory = new BufferMemory({ maxMessages: 3 });
    await memory.remember([toolRequest("c1"), toolResult("c1"), user("4"), user("5")]);
    // The (assistant+tool) unit is evicted as a whole.
    expect(memory.messages.map((m) => m.role)).toEqual(["user", "user"]);
    expect(memory.messages.map((m) => m.content)).toEqual(["4", "5"]);
  });

  it("keeps tool units whole even when the bound overshoots", async () => {
    const memory = new BufferMemory({ maxMessages: 4 });
    await memory.remember([
      user("q"),
      toolRequest("c1"),
      toolResult("c1"),
      user("x"),
      user("y"),
      user("z"),
    ]);
    // 6 > 4: evict q, then the whole (assistant+tool) unit → 3 remain.
    expect(memory.messages.map((m) => m.content)).toEqual(["x", "y", "z"]);
    expect(memory.messages.some((m) => m.role === "tool")).toBe(false);
  });

  it("clear() drops everything", async () => {
    const memory = new BufferMemory();
    await memory.remember([user("a")]);
    memory.clear();
    expect(memory.messages).toEqual([]);
  });

  it("rejects invalid maxMessages", () => {
    expect(() => new BufferMemory({ maxMessages: 0 })).toThrow(ValidationError);
    expect(() => new BufferMemory({ maxMessages: 2.5 })).toThrow(ValidationError);
    expect(() => new BufferMemory({ maxMessages: -1 })).toThrow(ValidationError);
  });

  it("stores clones, not caller references", async () => {
    const memory = new BufferMemory();
    const input = [user("original")];
    await memory.remember(input);
    input[0]!.content = "mutated";
    expect(memory.messages[0]!.content).toBe("original");
  });
});

// ---------------------------------------------------------------------------
// SummarizingMemory
// ---------------------------------------------------------------------------

describe("SummarizingMemory", () => {
  it("requires an AI instance", () => {
    expect(() => new SummarizingMemory({ ai: {} as unknown as AI })).toThrow(ValidationError);
  });

  it("requires keepRecent < maxMessages", () => {
    const ai = makeAI();
    expect(() => new SummarizingMemory({ ai, keepRecent: 10, maxMessages: 10 })).toThrow(
      ValidationError,
    );
    expect(() => new SummarizingMemory({ ai, keepRecent: 0 })).toThrow(ValidationError);
    expect(() => new SummarizingMemory({ ai, maxMessages: 1 })).toThrow(ValidationError);
  });

  it("keeps messages verbatim while under maxMessages (no summarize call)", async () => {
    const provider = new FakeProvider();
    const memory = new SummarizingMemory({
      ai: makeAI(provider),
      keepRecent: 2,
      maxMessages: 5,
    });

    await memory.remember([user("1"), user("2"), user("3")]);

    expect(memory.messages.map((m) => m.content)).toEqual(["1", "2", "3"]);
    expect(provider.generateCalls).toHaveLength(0);
  });

  it("summarizes old messages once the bound is exceeded", async () => {
    const provider = new FakeProvider({
      generateImpl: async () => makeResponse({ text: "  The user said 1-3.  " }),
    });
    const memory = new SummarizingMemory({
      ai: makeAI(provider),
      keepRecent: 2,
      maxMessages: 4,
    });

    await memory.remember([user("1"), user("2"), user("3"), user("4"), user("5")]);

    // One summarization call: system prompt + one user message with the transcript.
    expect(provider.generateCalls).toHaveLength(1);
    const call = provider.generateCalls[0]!;
    expect(call.messages[0]!.role).toBe("system");
    expect(call.messages[1]!.content).toContain("New messages to fold into the summary");
    expect(call.messages[1]!.content).toContain("3");

    // One summary message + keepRecent verbatim messages.
    expect(memory.messages).toHaveLength(3);
    expect(memory.messages[0]!.role).toBe("user");
    expect(memory.messages[0]!.content).toContain("[Conversation summary so far]");
    expect(memory.messages[0]!.content).toContain("The user said 1-3.");
    expect(memory.messages.slice(1).map((m) => m.content)).toEqual(["4", "5"]);
  });

  it("folds the previous summary into the next summarization", async () => {
    const provider = new FakeProvider({
      generateImpl: async () => makeResponse({ text: "summary v2" }),
    });
    const memory = new SummarizingMemory({
      ai: makeAI(provider),
      keepRecent: 1,
      maxMessages: 3,
    });

    await memory.remember([user("a"), user("b")]); // fits (bound 3)
    expect(provider.generateCalls).toHaveLength(0);

    await memory.remember([user("c"), user("d")]); // overflows → one summarize call
    expect(provider.generateCalls).toHaveLength(1);
    const transcript = provider.generateCalls[0]!.messages[1]!.content;
    expect(transcript).toContain("a");
    expect(transcript).toContain("b");
    expect(transcript).toContain("c");
  });

  it("keeps the newest messages verbatim after summarizing", async () => {
    const provider = new FakeProvider({
      generateImpl: async () => makeResponse({ text: "rolled up" }),
    });
    const memory = new SummarizingMemory({
      ai: makeAI(provider),
      keepRecent: 2,
      maxMessages: 4,
    });

    await memory.remember([
      user("old-1"),
      user("old-2"),
      user("old-3"),
      user("new-1"),
      user("new-2"),
    ]);

    const contents = memory.messages.map((m) => m.content);
    expect(contents).toHaveLength(3);
    expect(contents[1]).toBe("new-1");
    expect(contents[2]).toBe("new-2");
  });

  it("propagates an empty summary as a ValidationError", async () => {
    const provider = new FakeProvider({
      generateImpl: async () => makeResponse({ text: "   " }),
    });
    const memory = new SummarizingMemory({
      ai: makeAI(provider),
      keepRecent: 1,
      maxMessages: 3,
    });

    await expect(memory.remember([user("a"), user("b"), user("c"), user("d")])).rejects.toThrow(
      ValidationError,
    );
  });

  it("propagates provider failures honestly (no data loss, no partial state)", async () => {
    const provider = new FakeProvider({
      generateImpl: async () => {
        throw new Error("provider down");
      },
    });
    const memory = new SummarizingMemory({
      ai: makeAI(provider),
      keepRecent: 1,
      maxMessages: 3,
    });

    await expect(memory.remember([user("a"), user("b"), user("c"), user("d")])).rejects.toThrow(
      "provider down",
    );
    // Summarize failed before anything was discarded: all messages intact,
    // no summary message inserted.
    expect(memory.messages.map((m) => m.content)).toEqual(["a", "b", "c", "d"]);
  });

  it("clear() drops buffer and summary", async () => {
    const provider = new FakeProvider({
      generateImpl: async () => makeResponse({ text: "summary" }),
    });
    const memory = new SummarizingMemory({
      ai: makeAI(provider),
      keepRecent: 1,
      maxMessages: 3,
    });
    await memory.remember([user("a"), user("b"), user("c"), user("d")]);
    expect(memory.messages.length).toBeGreaterThan(0);

    memory.clear();
    expect(memory.messages).toEqual([]);
    // A fresh cycle starts without the old summary.
    await memory.remember([user("x")]);
    expect(memory.messages.map((m) => m.content)).toEqual(["x"]);
  });

  it("never leaves a tool result at the front without its assistant request", async () => {
    const provider = new FakeProvider({
      generateImpl: async () => makeResponse({ text: "sum" }),
    });
    const memory = new SummarizingMemory({
      ai: makeAI(provider),
      keepRecent: 3,
      maxMessages: 5,
    });

    await memory.remember([
      user("q"),
      toolRequest("c1"),
      toolResult("c1"),
      user("w"),
      user("e"),
      user("r"),
    ]);

    // The (assistant+tool) unit was folded together, not split.
    const roles = memory.messages.map((m) => m.role);
    expect(roles.some((r) => r === "tool")).toBe(false);
    // The summary is a single leading user message.
    expect(memory.messages[0]!.content).toContain("[Conversation summary so far]");
  });
});

// ---------------------------------------------------------------------------
// Extension contract
// ---------------------------------------------------------------------------

describe("Memory extension", () => {
  it("accepts a custom Memory subclass in the agent", async () => {
    const { Agent } = await import("@murphychain/core");
    const provider = new FakeProvider({
      generateImpl: async () => makeResponse({ text: "ok" }),
    });

    class LastOnlyMemory extends BufferMemory {
      override async remember(messages: readonly Message[]): Promise<void> {
        await super.remember(messages.slice(-1));
      }
    }

    const memory = new LastOnlyMemory({ maxMessages: 10 });
    const agent = new Agent({ ai: makeAI(provider), memory });

    await agent.run("first");
    await agent.run("second");

    // Each run persisted one message (its final assistant answer).
    expect(memory.messages.map((m) => m.role)).toEqual(["assistant", "assistant"]);
  });

  it("keeps tool JSON interop working for agents", async () => {
    const tool = Tool.define({
      name: "echo",
      description: "Echoes.",
      parameters: z.object({ text: z.string() }),
      execute: async ({ text }) => text,
    });
    const result = await tool.run({ text: "hi" });
    expect(result).toBe("hi");
  });
});
