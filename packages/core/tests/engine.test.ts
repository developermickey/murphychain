import { afterEach, describe, expect, it } from "vitest";
import {
  AI,
  ConfigurationError,
  ExecutionCancelledError,
  ExecutionTimeoutError,
  Model,
  Prompt,
  ProviderAuthError,
  ProviderError,
  ProviderServerError,
  UnsupportedCapabilityError,
  ValidationError,
  listProviders,
  registerProvider,
  unregisterProvider,
  z,
} from "@murphychain/core";
import type {
  AIChunk,
  AIResponse,
  ChunkEvent,
  ProviderFactoryConfig,
  RequestStartEvent,
} from "@murphychain/core";
import { FAST_RETRY, FakeProvider, defaultStream, makeResponse } from "./helpers";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** A provider whose generate() hangs forever and ignores its abort signal. */
function hangingProvider(): FakeProvider {
  return new FakeProvider({ generateImpl: () => new Promise<AIResponse>(() => undefined) });
}

/** A provider that fails `times` with a retryable error, then succeeds. */
function flakyProvider(times: number): FakeProvider {
  let calls = 0;
  return new FakeProvider({
    generateImpl: async () => {
      calls += 1;
      if (calls <= times) throw new ProviderServerError("fake", `failure ${calls}`);
      return makeResponse({ provider: "fake", model: "fake-1" });
    },
  });
}

function collect(stream: AsyncIterable<AIChunk>): Promise<AIChunk[]> {
  return (async () => {
    const chunks: AIChunk[] = [];
    for await (const chunk of stream) chunks.push(chunk);
    return chunks;
  })();
}

const registered: string[] = [];
function registerScratch(name: string, factory: (config: ProviderFactoryConfig) => unknown): void {
  registerProvider(name, factory as never);
  registered.push(name);
}

afterEach(() => {
  while (registered.length > 0) unregisterProvider(registered.pop() as string);
});

// ---------------------------------------------------------------------------
// Construction & configuration
// ---------------------------------------------------------------------------

describe("AI construction", () => {
  it("accepts a provider instance and exposes its name", () => {
    const ai = new AI({ provider: new FakeProvider(), model: "m" });
    expect(ai.providerName).toBe("fake");
  });

  it("normalizes a provider string name", () => {
    const ai = new AI({ provider: "  OpenAI  ", model: "m" });
    expect(ai.providerName).toBe("openai");
  });

  it("rejects a non-object config", () => {
    expect(() => new AI(undefined as never)).toThrowError(ConfigurationError);
    expect(() => new AI(null as never)).toThrowError(ConfigurationError);
  });

  it("rejects invalid configuration with structured issues", () => {
    expect(() => new AI({ provider: "" })).toThrowError(ConfigurationError);
    expect(() => new AI({ provider: 42 as never })).toThrowError(ConfigurationError);
    expect(() => new AI({ provider: "p", temperature: -1 })).toThrowError(ConfigurationError);
    expect(() => new AI({ provider: "p", temperature: Number.NaN })).toThrowError(
      ConfigurationError,
    );
    expect(() => new AI({ provider: "p", maxTokens: 0 })).toThrowError(ConfigurationError);
    expect(() => new AI({ provider: "p", topP: 2 })).toThrowError(ConfigurationError);
    expect(() => new AI({ provider: "p", timeoutMs: -1 })).toThrowError(ConfigurationError);
    expect(() => new AI({ provider: "p", retries: 21 })).toThrowError(ConfigurationError);
    expect(() => new AI({ provider: "p", unknownOption: true } as never)).toThrowError(
      ConfigurationError,
    );
  });

  it("rejects non-http(s) baseUrl", () => {
    expect(() => new AI({ provider: "p", baseUrl: "ftp://api.example.com" })).toThrowError(
      ConfigurationError,
    );
    expect(() => new AI({ provider: "p", baseUrl: "not a url" })).toThrowError(ConfigurationError);
  });

  it("accepts http(s) baseUrl", () => {
    expect(() => new AI({ provider: "p", baseUrl: "https://api.example.com/v1" })).not.toThrow();
  });
});

// ---------------------------------------------------------------------------
// Request resolution
// ---------------------------------------------------------------------------

describe("request resolution", () => {
  it("wraps a string prompt as a user message", async () => {
    const provider = new FakeProvider();
    const ai = new AI({ provider, model: "m" });
    await ai.generate("hello");
    expect(provider.generateCalls[0]!.messages).toEqual([{ role: "user", content: "hello" }]);
  });

  it("accepts a messages array", async () => {
    const provider = new FakeProvider();
    const ai = new AI({ provider, model: "m" });
    await ai.generate([
      { role: "user", content: "hi" },
      { role: "assistant", content: "yo" },
      { role: "user", content: "again" },
    ]);
    expect(provider.generateCalls[0]!.messages).toHaveLength(3);
  });

  it("prepends config systemPrompt and request system in order", async () => {
    const provider = new FakeProvider();
    const ai = new AI({ provider, model: "m", systemPrompt: "Config system" });
    await ai.generate({ prompt: "hi", system: "Request system" });
    expect(provider.generateCalls[0]!.messages).toEqual([
      { role: "system", content: "Config system" },
      { role: "system", content: "Request system" },
      { role: "user", content: "hi" },
    ]);
  });

  it("requires a model and reports how to fix it", async () => {
    const ai = new AI({ provider: new FakeProvider() });
    await expect(ai.generate("hi")).rejects.toThrowError(ConfigurationError);
    await expect(ai.generate("hi")).rejects.toThrow(/new AI\(\{ model/);
  });

  it("merges parameters: call > Model defaults > AI config", async () => {
    const provider = new FakeProvider();
    const model = new Model("m-model", { temperature: 0.3, topP: 0.8 });
    const ai = new AI({ provider, model, temperature: 0.9, maxTokens: 100 });

    await ai.generate({ prompt: "a", temperature: 0.1 });
    expect(provider.generateCalls[0]!.temperature).toBe(0.1);
    expect(provider.generateCalls[0]!.maxTokens).toBe(100); // config fallback
    expect(provider.generateCalls[0]!.topP).toBe(0.8); // Model default

    await ai.generate({ prompt: "b" });
    expect(provider.generateCalls[1]!.temperature).toBe(0.3); // Model default beats config
    expect(provider.generateCalls[1]!.maxTokens).toBe(100);

    const ai2 = new AI({ provider, model: "plain", temperature: 0.7 });
    await ai2.generate({ prompt: "c", maxTokens: 5 });
    expect(provider.generateCalls[2]!.maxTokens).toBe(5);
    expect(provider.generateCalls[2]!.temperature).toBe(0.7); // no Model defaults
  });

  it("resolves a per-request Model instance", async () => {
    const provider = new FakeProvider();
    const ai = new AI({ provider, model: "default-model" });
    await ai.generate({ prompt: "hi", model: new Model("other-model", { temperature: 0 }) });
    expect(provider.generateCalls[0]!.model).toBe("other-model");
    expect(provider.generateCalls[0]!.temperature).toBe(0);
  });

  it("passes stop sequences through", async () => {
    const provider = new FakeProvider();
    const ai = new AI({ provider, model: "m", stop: ["END"] });
    await ai.generate("hi");
    expect(provider.generateCalls[0]!.stop).toEqual(["END"]);
  });

  it("normalizes Zod tool definitions to JSON Schema exactly once", async () => {
    const provider = new FakeProvider();
    const ai = new AI({ provider, model: "m" });
    await ai.generate({
      prompt: "weather?",
      tools: [{ name: "get_weather", description: "d", parameters: z.object({ city: z.string() }) }],
      toolChoice: { name: "get_weather" },
    });
    const tools = provider.generateCalls[0]!.tools;
    expect(tools).toHaveLength(1);
    expect(tools?.[0]!.jsonSchema).toEqual({
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
    });
    expect(tools?.[0]!.jsonSchema.$schema).toBeUndefined();
    expect(provider.generateCalls[0]!.toolChoice).toEqual({ name: "get_weather" });
  });
});

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

describe("input validation", () => {
  const ai = new AI({ provider: new FakeProvider(), model: "m" });

  it("rejects an empty prompt", async () => {
    await expect(ai.generate("")).rejects.toThrowError(ValidationError);
  });

  it("rejects neither messages nor prompt", async () => {
    await expect(ai.generate({} as never)).rejects.toThrowError(ValidationError);
  });

  it("rejects both messages and prompt", async () => {
    await expect(
      ai.generate({ prompt: "a", messages: [{ role: "user", content: "b" }] }),
    ).rejects.toThrowError(ValidationError);
  });

  it("rejects unknown request keys", async () => {
    await expect(ai.generate({ prompt: "a", nope: 1 } as never)).rejects.toThrowError(
      ValidationError,
    );
  });

  it("rejects out-of-range parameters", async () => {
    await expect(ai.generate({ prompt: "a", temperature: -1 })).rejects.toThrowError(
      ValidationError,
    );
    await expect(ai.generate({ prompt: "a", maxTokens: 0 })).rejects.toThrowError(ValidationError);
    await expect(ai.generate({ prompt: "a", topP: 2 })).rejects.toThrowError(ValidationError);
  });

  it("rejects toolChoice without tools", async () => {
    await expect(ai.generate({ prompt: "a", toolChoice: "auto" })).rejects.toThrowError(
      ValidationError,
    );
  });

  it("rejects malformed tool messages", async () => {
    await expect(ai.generate([{ role: "tool", content: "x" }])).rejects.toThrowError(
      ValidationError,
    );
    await expect(
      ai.generate([
        { role: "tool", content: "x", name: "t", toolCallId: "c1" },
        { role: "user", content: "y", toolCalls: [{ id: "c", name: "n", arguments: {} }] },
      ]),
    ).rejects.toThrowError(ValidationError);
  });

  it("rejects non-object / non-array inputs", async () => {
    await expect(ai.generate(42 as never)).rejects.toThrowError(ValidationError);
    await expect(ai.generate(null as never)).rejects.toThrowError(ValidationError);
  });

  it("enforces maxMessages", async () => {
    const limited = new AI({ provider: new FakeProvider(), model: "m", maxMessages: 2 });
    const messages = [
      { role: "user" as const, content: "1" },
      { role: "user" as const, content: "2" },
      { role: "user" as const, content: "3" },
    ];
    await expect(limited.generate(messages)).rejects.toThrowError(ValidationError);
  });

  it("enforces maxInputChars", async () => {
    const limited = new AI({ provider: new FakeProvider(), model: "m", maxInputChars: 5 });
    await expect(limited.generate("0123456789")).rejects.toThrowError(ValidationError);
  });

  it("validates per-call options", async () => {
    await expect(ai.generate("hi", { timeoutMs: -1 })).rejects.toThrowError(ValidationError);
    await expect(ai.generate("hi", { signal: "nope" as never })).rejects.toThrowError(
      ValidationError,
    );
    await expect(ai.generate("hi", { includeRaw: "yes" as never })).rejects.toThrowError(
      ValidationError,
    );
    await expect(ai.generate("hi", { metadata: [] as never })).rejects.toThrowError(
      ValidationError,
    );
  });
});

// ---------------------------------------------------------------------------
// Capability gates
// ---------------------------------------------------------------------------

describe("capability gates", () => {
  it("rejects generate when the provider lacks it", async () => {
    const provider = new FakeProvider({ capabilities: { generate: false } });
    const ai = new AI({ provider, model: "m" });
    await expect(ai.generate("hi")).rejects.toThrowError(UnsupportedCapabilityError);
    expect(provider.generateCalls).toHaveLength(0);
  });

  it("rejects stream when the provider lacks streaming", async () => {
    const provider = new FakeProvider({ capabilities: { streaming: false } });
    const ai = new AI({ provider, model: "m" });
    await expect(ai.stream("hi")).rejects.toThrowError(UnsupportedCapabilityError);
    expect(provider.streamCalls).toHaveLength(0);
  });

  it("rejects tools when the provider lacks tool calling", async () => {
    const provider = new FakeProvider({ capabilities: { toolCalling: false } });
    const ai = new AI({ provider, model: "m" });
    await expect(
      ai.generate({ prompt: "x", tools: [{ name: "t", description: "d", parameters: {} }] }),
    ).rejects.toThrowError(UnsupportedCapabilityError);
  });

  it("rejects system messages when the provider lacks them", async () => {
    const provider = new FakeProvider({ capabilities: { systemMessage: false } });
    const ai = new AI({ provider, model: "m" });
    await expect(ai.generate({ prompt: "x", system: "s" })).rejects.toThrowError(
      UnsupportedCapabilityError,
    );
  });

  it("rejects embed when the provider lacks embeddings", async () => {
    const provider = new FakeProvider({ capabilities: { embeddings: false } });
    const ai = new AI({ provider, model: "m" });
    await expect(ai.embed("text")).rejects.toThrowError(UnsupportedCapabilityError);
    await expect(ai.embedMany(["text"])).rejects.toThrowError(UnsupportedCapabilityError);
  });
});

// ---------------------------------------------------------------------------
// Retry behaviour
// ---------------------------------------------------------------------------

describe("retry", () => {
  it("retries retryable failures until success and counts them", async () => {
    const provider = flakyProvider(2);
    const ai = new AI({ provider, model: "m", retries: FAST_RETRY });
    const response = await ai.generate("hi");
    expect(response.text).toBe("Hello!");
    expect(ai.stats().retries).toBe(2);
    expect(ai.stats().requests).toBe(1);
    expect(ai.stats().errors).toBe(0);
  });

  it("does not retry non-retryable errors", async () => {
    const provider = new FakeProvider({
      generateImpl: async () => {
        throw new ProviderAuthError("fake", "401");
      },
    });
    const ai = new AI({ provider, model: "m", retries: FAST_RETRY });
    await expect(ai.generate("hi")).rejects.toThrowError(ProviderAuthError);
    expect(provider.generateCalls).toHaveLength(1);
    expect(ai.stats().retries).toBe(0);
  });

  it("throws the last retryable error after exhausting retries", async () => {
    const provider = new FakeProvider({
      generateImpl: async () => {
        throw new ProviderServerError("fake", "always");
      },
    });
    const ai = new AI({ provider, model: "m", retries: { ...FAST_RETRY, retries: 1 } });
    await expect(ai.generate("hi")).rejects.toThrowError(ProviderServerError);
    expect(provider.generateCalls).toHaveLength(2);
  });

  it("emits a retry event per attempt with attempt numbers", async () => {
    const provider = flakyProvider(1);
    const ai = new AI({ provider, model: "m", retries: FAST_RETRY });
    const attempts: number[] = [];
    ai.on("retry", (event) => attempts.push(event.attempt));
    await ai.generate("hi");
    expect(attempts).toEqual([2]);
  });

  it("does not retry a timeout that fires during backoff", async () => {
    const provider = new FakeProvider({
      generateImpl: async () => {
        throw new ProviderServerError("fake", "fail then hang in backoff");
      },
    });
    const ai = new AI({
      provider,
      model: "m",
      timeoutMs: 30,
      retries: { retries: 5, initialDelayMs: 10_000, maxDelayMs: 10_000, backoffFactor: 1, jitter: false },
    });
    await expect(ai.generate("hi")).rejects.toThrowError(ExecutionTimeoutError);
  });
});

// ---------------------------------------------------------------------------
// Timeout & cancellation
// ---------------------------------------------------------------------------

describe("timeout and cancellation", () => {
  // raceAbort: even a provider that ignores the signal cannot outlive the
  // engine's deadline.
  it("enforces the deadline when the provider hangs and ignores its signal", async () => {
    const provider = hangingProvider();
    const ai = new AI({ provider, model: "m", timeoutMs: 50, retries: 0 });
    await expect(ai.generate("hi")).rejects.toThrowError(ExecutionTimeoutError);
    await expect(ai.generate("hi")).rejects.toThrow(/timed out after 50ms/);
    // The provider was handed the signal, and it is aborted.
    expect(provider.generateCalls[0]!.signal?.aborted).toBe(true);
  });

  it("honours a per-call timeoutMs override", async () => {
    const provider = hangingProvider();
    const ai = new AI({ provider, model: "m", timeoutMs: 60_000, retries: 0 });
    await expect(ai.generate("hi", { timeoutMs: 40 })).rejects.toThrow(/timed out after 40ms/);
  });

  it("emits an abort event with reason 'timeout'", async () => {
    const provider = hangingProvider();
    const ai = new AI({ provider, model: "m", timeoutMs: 30, retries: 0 });
    const reasons: string[] = [];
    ai.on("abort", (event) => reasons.push(event.reason));
    await expect(ai.generate("hi")).rejects.toThrowError(ExecutionTimeoutError);
    expect(reasons).toEqual(["timeout"]);
  });

  it("maps a user signal abort to ExecutionCancelledError", async () => {
    const provider = hangingProvider();
    const ai = new AI({ provider, model: "m", timeoutMs: 0, retries: 0 });
    const controller = new AbortController();
    const promise = ai.generate("hi", { signal: controller.signal });
    controller.abort();
    await expect(promise).rejects.toThrowError(ExecutionCancelledError);
  });

  it("emits an abort event with reason 'user' on cancellation", async () => {
    const provider = hangingProvider();
    const ai = new AI({ provider, model: "m", timeoutMs: 0, retries: 0 });
    const reasons: string[] = [];
    ai.on("abort", (event) => reasons.push(event.reason));
    const controller = new AbortController();
    const promise = ai.generate("hi", { signal: controller.signal });
    controller.abort();
    await expect(promise).rejects.toThrowError(ExecutionCancelledError);
    expect(reasons).toEqual(["user"]);
  });

  it("rejects immediately for an already-aborted signal", async () => {
    const provider = new FakeProvider();
    const ai = new AI({ provider, model: "m" });
    const controller = new AbortController();
    controller.abort();
    await expect(ai.generate("hi", { signal: controller.signal })).rejects.toThrowError(
      ExecutionCancelledError,
    );
    expect(provider.generateCalls).toHaveLength(0); // fails before the provider runs
  });
});

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

describe("streaming", () => {
  it("yields engine-stamped chunk indexes and aggregates usage", async () => {
    const provider = new FakeProvider();
    const ai = new AI({ provider, model: "m" });
    const chunks = await collect(await ai.stream("hi"));
    expect(chunks.map((chunk) => chunk.index)).toEqual([0, 1, 2]);
    expect(chunks.map((chunk) => chunk.text)).toEqual(["Hel", "lo", ""]);
    expect(chunks[2]!.finishReason).toBe("stop");
    expect(chunks[2]!.usage).toEqual({ promptTokens: 1, completionTokens: 1, totalTokens: 2 });
  });

  it("is single-use", async () => {
    const ai = new AI({ provider: new FakeProvider(), model: "m" });
    const stream = await ai.stream("hi");
    await collect(stream);
    expect(() => stream[Symbol.asyncIterator]()).toThrowError(ConfigurationError);
  });

  it("cancels the provider stream when the consumer breaks early", async () => {
    let providerFinished = false;
    const provider = new FakeProvider({
      streamImpl: () =>
        (async function* () {
          try {
            yield { text: "a", index: 0 };
            await new Promise((resolve) => setTimeout(resolve, 50));
            yield { text: "b", index: 1 };
          } finally {
            providerFinished = true;
          }
        })(),
    });
    const ai = new AI({ provider, model: "m" });
    for await (const chunk of await ai.stream("hi")) {
      expect(chunk.text).toBe("a");
      break;
    }
    await new Promise((resolve) => setTimeout(resolve, 10));
    expect(providerFinished).toBe(true);
  });

  it("retries only until the first chunk arrives", async () => {
    let calls = 0;
    const provider = new FakeProvider({
      streamImpl: () => {
        calls += 1;
        if (calls === 1) {
          return (async function* () {
            throw new ProviderServerError("fake", "open failure");
          })();
        }
        return defaultStream();
      },
    });
    const ai = new AI({ provider, model: "m", retries: FAST_RETRY });
    const chunks = await collect(await ai.stream("hi"));
    expect(calls).toBe(2);
    expect(ai.stats().retries).toBe(1);
    expect(chunks.map((chunk) => chunk.text)).toEqual(["Hel", "lo", ""]);
  });

  it("propagates failures after the first chunk without retrying", async () => {
    const provider = new FakeProvider({
      streamImpl: () =>
        (async function* () {
          yield { text: "first", index: 0 };
          throw new ProviderServerError("fake", "mid-stream failure");
        })(),
    });
    const ai = new AI({ provider, model: "m", retries: FAST_RETRY });
    const stream = await ai.stream("hi");
    const iterator = stream[Symbol.asyncIterator]();
    expect((await iterator.next()).value.text).toBe("first");
    await expect(iterator.next()).rejects.toThrowError(ProviderServerError);
    expect(ai.stats().retries).toBe(0);
  });

  it("clears the deadline after the first chunk (time-to-first-chunk only)", async () => {
    const provider = new FakeProvider({
      streamImpl: () =>
        (async function* () {
          yield { text: "a", index: 0 };
          await new Promise((resolve) => setTimeout(resolve, 60));
          yield { text: "b", index: 1 };
        })(),
    });
    const ai = new AI({ provider, model: "m", timeoutMs: 40, retries: 0 });
    const chunks = await collect(await ai.stream("hi"));
    expect(chunks.map((chunk) => chunk.text)).toEqual(["a", "b"]);
  });

  it("times out before the first chunk", async () => {
    const provider = new FakeProvider({
      streamImpl: () =>
        (async function* () {
          await new Promise((resolve) => setTimeout(resolve, 500));
          yield { text: "late", index: 0 };
        })(),
    });
    const ai = new AI({ provider, model: "m", timeoutMs: 40, retries: 0 });
    await expect(collect(await ai.stream("hi"))).rejects.toThrowError(ExecutionTimeoutError);
  });
});

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

describe("embeddings", () => {
  it("embeds a single text", async () => {
    const provider = new FakeProvider();
    const ai = new AI({ provider, model: "m" });
    await expect(ai.embed("hello")).resolves.toEqual([0.1, 0.2]);
    expect(provider.embedCalls[0]!.text).toBe("hello");
  });

  it("embeds a batch in order", async () => {
    const provider = new FakeProvider();
    const ai = new AI({ provider, model: "m" });
    await expect(ai.embedMany(["a", "b", "c"])).resolves.toEqual([
      [0.1, 0.2],
      [0.1, 0.2],
      [0.1, 0.2],
    ]);
    expect(provider.embedCalls.map((call) => call.text)).toEqual(["a", "b", "c"]);
  });

  it("validates inputs", async () => {
    const ai = new AI({ provider: new FakeProvider(), model: "m" });
    await expect(ai.embed("")).rejects.toThrowError(ValidationError);
    await expect(ai.embedMany([])).rejects.toThrowError(ValidationError);
    await expect(ai.embedMany(["ok", ""])).rejects.toThrowError(ValidationError);
    await expect(ai.embedMany([42 as never])).rejects.toThrowError(ValidationError);
    await expect(ai.embedMany(new Array(1_001).fill("x"))).rejects.toThrowError(ValidationError);
  });

  it("retries retryable embed failures", async () => {
    let calls = 0;
    const provider = new FakeProvider({
      embedImpl: async () => {
        calls += 1;
        if (calls === 1) throw new ProviderServerError("fake", "transient");
        return [1, 2];
      },
    });
    const ai = new AI({ provider, model: "m", retries: FAST_RETRY });
    await expect(ai.embed("x")).resolves.toEqual([1, 2]);
    expect(ai.stats().retries).toBe(1);
  });

  it("enforces maxInputChars on embed input", async () => {
    const ai = new AI({ provider: new FakeProvider(), model: "m", maxInputChars: 3 });
    await expect(ai.embed("abcd")).rejects.toThrowError(ValidationError);
  });

  it("rejects malformed embedding payloads without retrying", async () => {
    const malformed: unknown[] = ["nope", [], [1, "x"], [NaN], [Infinity], null];
    for (const value of malformed) {
      const provider = new FakeProvider({ embedImpl: async () => value as number[] });
      const ai = new AI({ provider, model: "m", retries: FAST_RETRY });
      const error: unknown = await ai.embed("x").catch((e: unknown) => e);
      expect(error).toBeInstanceOf(ProviderError);
      expect((error as ProviderError).retryable).toBe(false);
      // A malformed payload is a provider bug, never a transient failure:
      // exactly one request went out, no retry, error counted.
      expect(ai.stats().requests).toBe(1);
      expect(ai.stats().retries).toBe(0);
      expect(ai.stats().errors).toBe(1);
    }
  });

  it("rejects a batch whose length does not match the inputs", async () => {
    const provider = new FakeProvider();
    provider.embedMany = async () => [[1, 2]]; // one vector for two inputs
    const ai = new AI({ provider, model: "m", retries: FAST_RETRY });
    await expect(ai.embedMany(["a", "b"])).rejects.toThrow(
      /returned 1 embeddings for 2 inputs/,
    );
    expect(ai.stats().retries).toBe(0);
  });

  it("rejects a batch with inconsistent vector dimensions", async () => {
    const provider = new FakeProvider();
    provider.embedMany = async () => [
      [1, 2],
      [1, 2, 3],
    ];
    const ai = new AI({ provider, model: "m" });
    await expect(ai.embedMany(["a", "b"])).rejects.toThrow(
      /inconsistent embedding dimensions/,
    );
    // Single embeddings keep their own dimension freedom (checked per call).
    const single = new FakeProvider();
    await expect(new AI({ provider: single, model: "m" }).embed("x")).resolves.toEqual([
      0.1, 0.2,
    ]);
  });
});

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

describe("events", () => {
  it("emits the full lifecycle for a successful generate", async () => {
    const ai = new AI({ provider: new FakeProvider(), model: "m" });
    const sequence: string[] = [];
    ai.on("request:start", () => sequence.push("start"));
    ai.on("retry", () => sequence.push("retry"));
    ai.on("request:success", () => sequence.push("success"));
    ai.on("request:error", () => sequence.push("error"));
    await ai.generate("hi");
    expect(sequence).toEqual(["start", "success"]);
  });

  it("emits retry events between start and success", async () => {
    const ai = new AI({ provider: flakyProvider(1), model: "m", retries: FAST_RETRY });
    const sequence: string[] = [];
    ai.on("request:start", () => sequence.push("start"));
    ai.on("retry", () => sequence.push("retry"));
    ai.on("request:success", () => sequence.push("success"));
    await ai.generate("hi");
    expect(sequence).toEqual(["start", "retry", "success"]);
  });

  it("carries counts (never content) on request:start", async () => {
    const ai = new AI({
      provider: new FakeProvider(),
      model: "m",
      systemPrompt: "secret system",
      metadata: { env: "test" },
    });
    let start: RequestStartEvent | undefined;
    ai.on("request:start", (event) => {
      start = event;
    });
    await ai.generate({ prompt: "very secret prompt" }, { metadata: { traceId: "t1" } });
    expect(start).toBeDefined();
    expect(Object.keys(start as never)).toEqual([
      "requestId",
      "provider",
      "model",
      "messageCount",
      "totalChars",
      "metadata",
    ]);
    expect(start?.messageCount).toBe(2); // config system + user prompt
    expect(start?.totalChars).toBe("secret system".length + "very secret prompt".length);
    expect(start?.metadata).toEqual({ env: "test", traceId: "t1" });
    expect(JSON.stringify(start)).not.toContain("secret prompt");
  });

  it("carries only index and length on chunk events", async () => {
    const ai = new AI({ provider: new FakeProvider(), model: "m" });
    const events: ChunkEvent[] = [];
    ai.on("chunk", (event) => events.push(event));
    await collect(await ai.stream("hi"));
    expect(events.map((event) => event.index)).toEqual([0, 1, 2]);
    expect(events[0]!.textLength).toBe(3);
    expect(JSON.stringify(events)).not.toContain("Hel");
  });

  it("emits request:error with the mapped error", async () => {
    const ai = new AI({
      provider: new FakeProvider({
        generateImpl: async () => {
          throw new ProviderAuthError("fake", "bad key");
        },
      }),
      model: "m",
    });
    const errors: Array<{ error: ProviderError }> = [];
    ai.on("request:error", (event) => errors.push({ error: event.error as ProviderError }));
    await expect(ai.generate("hi")).rejects.toThrowError(ProviderAuthError);
    expect(errors).toHaveLength(1);
    expect(errors[0]!.error.code).toBe("PROVIDER_AUTH");
    expect(errors[0]!.error.provider).toBe("fake");
  });

  it("isolates listener exceptions from the request", async () => {
    const ai = new AI({ provider: new FakeProvider(), model: "m" });
    ai.on("request:start", () => {
      throw new Error("listener bug");
    });
    ai.on("request:success", () => {
      throw new Error("listener bug 2");
    });
    await expect(ai.generate("hi")).resolves.toBeTruthy();
  });

  it("supports once() and off()", async () => {
    const ai = new AI({ provider: new FakeProvider(), model: "m" });
    let onceCount = 0;
    let offCount = 0;
    const listener = (): void => {
      offCount += 1;
    };
    ai.once("request:start", () => {
      onceCount += 1;
    });
    ai.on("request:start", listener);
    await ai.generate("hi");
    ai.off("request:start", listener);
    await ai.generate("hi");
    expect(onceCount).toBe(1);
    expect(offCount).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Stats
// ---------------------------------------------------------------------------

describe("stats", () => {
  it("accumulates counters across calls and resets", async () => {
    const ai = new AI({ provider: new FakeProvider(), model: "m", retries: FAST_RETRY });
    await ai.generate("hi");
    await collect(await ai.stream("hi"));
    await ai.embed("x");

    const stats = ai.stats();
    expect(stats.requests).toBe(3);
    expect(stats.errors).toBe(0);
    expect(stats.chunks).toBe(3);
    expect(stats.promptTokens).toBe(3 + 1);
    expect(stats.completionTokens).toBe(2 + 1);
    expect(stats.totalTokens).toBe(5 + 2);
    expect(stats.totalLatencyMs).toBeGreaterThanOrEqual(0);
    expect(Object.isFrozen(stats)).toBe(true);

    ai.resetStats();
    expect(ai.stats()).toEqual({
      requests: 0,
      errors: 0,
      retries: 0,
      chunks: 0,
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
      totalLatencyMs: 0,
    });
  });

  it("counts errors", async () => {
    const ai = new AI({
      provider: new FakeProvider({
        generateImpl: async () => {
          throw new ProviderAuthError("fake", "401");
        },
      }),
      model: "m",
    });
    await expect(ai.generate("hi")).rejects.toThrow();
    expect(ai.stats().errors).toBe(1);
    expect(ai.stats().requests).toBe(1);
  });

  it("counts retries performed during a stream open", async () => {
    let calls = 0;
    const provider = new FakeProvider({
      streamImpl: () => {
        calls += 1;
        if (calls === 1) {
          return (async function* () {
            throw new ProviderServerError("fake", "open failure");
          })();
        }
        return defaultStream();
      },
    });
    const ai = new AI({ provider, model: "m", retries: FAST_RETRY });
    await collect(await ai.stream("hi"));
    expect(ai.stats().retries).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// includeRaw
// ---------------------------------------------------------------------------

describe("includeRaw", () => {
  it("attaches raw provider payloads by default", async () => {
    const ai = new AI({ provider: new FakeProvider(), model: "m" });
    const response = await ai.generate("hi");
    expect(response.raw).toBeDefined();
  });

  it("omits raw when disabled per call or per config", async () => {
    const ai = new AI({ provider: new FakeProvider(), model: "m", includeRaw: false });
    const response = await ai.generate("hi");
    expect(response.raw).toBeUndefined();

    const ai2 = new AI({ provider: new FakeProvider(), model: "m" });
    const response2 = await ai2.generate("hi", { includeRaw: false });
    expect(response2.raw).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Provider registry integration
// ---------------------------------------------------------------------------

describe("provider registry integration", () => {
  it("gives an actionable error for an unknown provider", async () => {
    const ai = new AI({ provider: "does-not-exist", model: "m" });
    await expect(ai.generate("hi")).rejects.toThrowError(ConfigurationError);
    await expect(ai.generate("hi")).rejects.toThrow(/Unknown provider "does-not-exist"/);
    await expect(ai.generate("hi")).rejects.toThrow(/registerProvider/);
  });

  it("lists registered providers in the error when some exist", () => {
    registerScratch("reg-engine-test", () => new FakeProvider());
    const error = (() => {
      try {
        // eslint-disable-next-line no-new
        new AI({ provider: "ghost", model: "m" });
        return undefined;
      } catch (e) {
        return e as ConfigurationError;
      }
    })();
    expect(error).toBeUndefined(); // config errors surface at request time
    expect(listProviders()).toContain("reg-engine-test");
  });

  it("loads a provider factory once and caches it", async () => {
    let factoryCalls = 0;
    registerScratch("cached-provider", () => {
      factoryCalls += 1;
      return new FakeProvider({ name: "cached-provider" });
    });
    const ai = new AI({ provider: "cached-provider", model: "m" });
    await ai.generate("hi");
    await ai.generate("hi");
    await ai.stream("hi");
    expect(factoryCalls).toBe(1);
  });

  it("passes the API key accessor (rotation) to the factory", async () => {
    let seen: ProviderFactoryConfig | undefined;
    registerScratch("rotating-provider", (config) => {
      seen = config;
      return new FakeProvider({ name: "rotating-provider" });
    });
    let keyCalls = 0;
    const ai = new AI({
      provider: "rotating-provider",
      model: "m",
      baseUrl: "https://api.example.com",
      apiKey: () => `key-${(keyCalls += 1)}`,
    });
    await ai.generate("hi");
    // The accessor is re-evaluated per request — credentials can rotate.
    expect(seen?.apiKey?.()).toBe("key-1");
    expect(seen?.apiKey?.()).toBe("key-2");
    expect(seen?.baseUrl).toBe("https://api.example.com");
    expect(seen?.model).toBe("m");
  });

  it("maps a missing adapter package to an install hint", async () => {
    registerScratch("not-installed", () => {
      const error = new Error("Cannot find package '@murphychain/not-installed'");
      (error as { code?: string }).code = "ERR_MODULE_NOT_FOUND";
      throw error;
    });
    const ai = new AI({ provider: "not-installed", model: "m" });
    await expect(ai.generate("hi")).rejects.toThrow(/pnpm add @murphychain\/not-installed/);
  });

  it("wraps unexpected factory failures", async () => {
    registerScratch("broken-provider", () => {
      throw new Error("factory exploded");
    });
    const ai = new AI({ provider: "broken-provider", model: "m" });
    await expect(ai.generate("hi")).rejects.toThrow(/factory exploded/);
  });
});

// ---------------------------------------------------------------------------
// DX smoke test (the intended developer experience)
// ---------------------------------------------------------------------------

describe("intended DX", () => {
  it("supports new AI(...) → generate → Prompt.template end to end", async () => {
    const provider = new FakeProvider();
    const ai = new AI({ provider, model: "gpt-4.1", apiKey: "k" });
    const prompt = Prompt.template("Say {greeting} to {name}.");
    const response = await ai.generate(prompt.format({ greeting: "hello", name: "Ada" }));
    expect(response.text).toBe("Hello!");
    expect(response.provider).toBe("fake");
    expect(response.model).toBe("gpt-4.1"); // provider echoes the model that served it
    expect(typeof response.latencyMs).toBe("number");
  });

  it("switches providers with a one-line change", async () => {
    const openaiLike = new FakeProvider({ name: "openai-like" });
    const geminiLike = new FakeProvider({ name: "gemini-like" });
    const first = await new AI({ provider: openaiLike, model: "m" }).generate("hi");
    const second = await new AI({ provider: geminiLike, model: "m" }).generate("hi");
    expect(first.provider).toBe("openai-like");
    expect(second.provider).toBe("gemini-like");
    // Identical response shape — application code does not change.
    expect(Object.keys(first)).toEqual(Object.keys(second));
  });
});
