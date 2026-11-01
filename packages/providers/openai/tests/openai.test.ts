import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { APIUserAbortError } from "openai";
import {
  ConfigurationError,
  ProviderAuthError,
  ProviderBadRequestError,
  ProviderContentFilterError,
  ProviderError,
  ProviderRateLimitError,
  ProviderServerError,
} from "@murphychain/core";
import type { Message, ModelProvider } from "@murphychain/core";
import { createOpenAIProvider, parseRetryAfterMs } from "@murphychain/openai";
import { startMockServer, type MockServer } from "./mockServer";
import type { ProviderFactoryConfig } from "@murphychain/core";

const USER: Message[] = [{ role: "user", content: "hi" }];

const COMPLETION = {
  id: "chatcmpl-1",
  object: "chat.completion",
  created: 1,
  model: "gpt-4.1",
  choices: [
    {
      index: 0,
      message: { role: "assistant", content: "Hello world" },
      finish_reason: "stop",
    },
  ],
  usage: { prompt_tokens: 10, completion_tokens: 5, total_tokens: 15 },
};

/** Builds one SSE `data:` payload shaped like a ChatCompletionChunk. */
function chunk(delta: unknown, finishReason: string | null = null, tail: Record<string, unknown> = {}): string {
  return JSON.stringify({
    id: "chatcmpl-s1",
    object: "chat.completion.chunk",
    created: 1,
    model: "gpt-4.1",
    choices: [{ index: 0, delta, finish_reason: finishReason }],
    ...tail,
  });
}

function makeProvider(server: MockServer, overrides: Partial<ProviderFactoryConfig> = {}): ModelProvider {
  return createOpenAIProvider({
    apiKey: () => "sk-test-key-12345",
    baseUrl: server.baseUrl,
    model: "gpt-4.1",
    ...overrides,
  });
}

let server: MockServer;
const spawned: MockServer[] = [];

/** Starts an additional mock server (tracked for cleanup) and makes it current. */
async function newServer(
  handler: Parameters<typeof startMockServer>[0],
): Promise<MockServer> {
  const instance = await startMockServer(handler);
  spawned.push(instance);
  server = instance;
  return instance;
}

beforeEach(async () => {
  spawned.length = 0;
  await newServer(() => ({ body: COMPLETION }));
});

afterEach(async () => {
  for (const instance of spawned) await instance.close();
});

// ---------------------------------------------------------------------------
// generate
// ---------------------------------------------------------------------------

describe("OpenAIProvider.generate", () => {
  it("sends a correctly shaped request and maps the response", async () => {
    const provider = makeProvider(server);
    const response = await provider.generate({
      messages: [
        { role: "system", content: "be nice" },
        { role: "user", content: "hi" },
      ],
      model: "gpt-4.1",
      temperature: 0.2,
      maxTokens: 100,
      topP: 0.9,
      stop: ["END"],
    });

    expect(response.text).toBe("Hello world");
    expect(response.provider).toBe("openai");
    expect(response.model).toBe("gpt-4.1");
    expect(response.finishReason).toBe("stop");
    expect(response.usage).toEqual({ promptTokens: 10, completionTokens: 5, totalTokens: 15 });
    expect(response.raw).toBeDefined();
    expect(response.message.role).toBe("assistant");

    const request = server.requests[0]!;
    expect(request.method).toBe("POST");
    expect(request.url).toBe("/chat/completions");
    expect(request.headers.authorization).toBe("Bearer sk-test-key-12345");
    expect(request.body).toMatchObject({
      model: "gpt-4.1",
      temperature: 0.2,
      max_completion_tokens: 100, // not the deprecated max_tokens
      top_p: 0.9,
      stop: ["END"],
      stream: false,
      messages: [
        { role: "system", content: "be nice" },
        { role: "user", content: "hi" },
      ],
    });
  });

  it("omits unset parameters instead of sending defaults", async () => {
    const provider = makeProvider(server);
    await provider.generate({ messages: USER, model: "gpt-4.1" });
    const body = server.requests[0]!.body as Record<string, unknown>;
    expect(body.temperature).toBeUndefined();
    expect(body.max_completion_tokens).toBeUndefined();
    expect(body.top_p).toBeUndefined();
    expect(body.stop).toBeUndefined();
    expect(body.tools).toBeUndefined();
    expect(body.stream_options).toBeUndefined();
  });

  it("omits raw when includeRaw is false", async () => {
    const provider = makeProvider(server);
    const response = await provider.generate({ messages: USER, model: "gpt-4.1", includeRaw: false });
    expect(response.raw).toBeUndefined();
  });

  it("maps assistant tool calls and tool results on the wire", async () => {
    const provider = makeProvider(server);
    await provider.generate({
      messages: [
        { role: "user", content: "weather?" },
        {
          role: "assistant",
          content: "",
          toolCalls: [{ id: "call_1", name: "get_weather", arguments: { city: "Paris" } }],
        },
        { role: "tool", content: "sunny", name: "get_weather", toolCallId: "call_1" },
      ],
      model: "gpt-4.1",
    });
    const body = server.requests[0]!.body as { messages: Array<Record<string, unknown>> };
    expect(body.messages[1]).toMatchObject({
      role: "assistant",
      tool_calls: [{ id: "call_1", type: "function", function: { name: "get_weather" } }],
    });
    expect(JSON.parse((body.messages[1]!.tool_calls as Array<{ function: { arguments: string } }>)[0]!.function.arguments)).toEqual({ city: "Paris" });
    expect(body.messages[2]).toEqual({
      role: "tool",
      tool_call_id: "call_1",
      content: "sunny",
    });
  });

  it("maps tools and named tool choice to the wire format", async () => {
    const provider = makeProvider(server);
    await provider.generate({
      messages: USER,
      model: "gpt-4.1",
      tools: [
        {
          name: "get_weather",
          description: "Get weather",
          jsonSchema: { type: "object", properties: { city: { type: "string" } } },
        },
      ],
      toolChoice: { name: "get_weather" },
    });
    const body = server.requests[0]!.body as Record<string, unknown>;
    expect(body.tools).toEqual([
      {
        type: "function",
        function: {
          name: "get_weather",
          description: "Get weather",
          parameters: { type: "object", properties: { city: { type: "string" } } },
        },
      },
    ]);
    expect(body.tool_choice).toEqual({ type: "function", function: { name: "get_weather" } });
  });

  it("passes string tool choices through unchanged", async () => {
    const provider = makeProvider(server);
    await provider.generate({
      messages: USER,
      model: "gpt-4.1",
      tools: [{ name: "t", description: "d", jsonSchema: { type: "object" } }],
      toolChoice: "required",
    });
    expect((server.requests[0]!.body as Record<string, unknown>).tool_choice).toBe("required");
  });

  it("parses tool calls from the response", async () => {
    await newServer(() => ({
      body: {
        ...COMPLETION,
        choices: [
          {
            index: 0,
            message: {
              role: "assistant",
              content: null,
              tool_calls: [
                {
                  id: "call_abc",
                  type: "function",
                  function: { name: "get_weather", arguments: '{"city":"Paris"}' },
                },
              ],
            },
            finish_reason: "tool_calls",
          },
        ],
      },
    }));
    const provider = makeProvider(server);
    const response = await provider.generate({ messages: USER, model: "gpt-4.1" });
    expect(response.text).toBe("");
    expect(response.finishReason).toBe("tool_calls");
    expect(response.message.toolCalls).toEqual([
      { id: "call_abc", name: "get_weather", arguments: { city: "Paris" } },
    ]);
  });

  it("rejects a completion with no choices as malformed", async () => {
    await newServer(() => ({ body: { ...COMPLETION, choices: [] } }));
    const provider = makeProvider(server);
    await expect(provider.generate({ messages: USER, model: "gpt-4.1" })).rejects.toSatisfy(
      (error: unknown) =>
        error instanceof ProviderError && error.code === "PROVIDER_MALFORMED_RESPONSE",
    );
  });

  it("requires a model", async () => {
    const provider = createOpenAIProvider({ apiKey: () => "k", baseUrl: server.baseUrl });
    await expect(provider.generate({ messages: USER })).rejects.toThrowError(ConfigurationError);
  });
});

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

describe("OpenAIProvider error mapping", () => {
  const errorBody = (message: string): string => JSON.stringify({ error: { message, type: "invalid" } });

  it("maps 401/403 to ProviderAuthError (never retried)", async () => {
    for (const status of [401, 403]) {
      await newServer(() => ({ status, raw: errorBody("bad credentials") }));
      const provider = makeProvider(server);
      const error = await provider.generate({ messages: USER, model: "gpt-4.1" }).catch((e: unknown) => e);
      expect(error).toBeInstanceOf(ProviderAuthError);
      expect((error as ProviderError).status).toBe(status);
      expect((error as ProviderError).retryable).toBe(false);
    }
  });

  it("maps 429 to ProviderRateLimitError and parses retry-after", async () => {
    await newServer(() => ({
      status: 429,
      raw: errorBody("slow down"),
      extraHeaders: { "retry-after": "2" },
    }));
    const provider = makeProvider(server);
    const error = (await provider
      .generate({ messages: USER, model: "gpt-4.1" })
      .catch((e: unknown) => e)) as ProviderRateLimitError;
    expect(error).toBeInstanceOf(ProviderRateLimitError);
    expect(error.retryable).toBe(true);
    expect(error.retryAfterMs).toBe(2_000);
  });

  it("maps 429 without retry-after to a retryable error with no hint", async () => {
    await newServer(() => ({ status: 429, raw: errorBody("slow down") }));
    const provider = makeProvider(server);
    const error = (await provider
      .generate({ messages: USER, model: "gpt-4.1" })
      .catch((e: unknown) => e)) as ProviderRateLimitError;
    expect(error).toBeInstanceOf(ProviderRateLimitError);
    expect(error.retryAfterMs).toBeUndefined();
  });

  it("maps 400/404/422 to ProviderBadRequestError", async () => {
    for (const status of [400, 404, 422]) {
      await newServer(() => ({ status, raw: errorBody("your fault") }));
      const provider = makeProvider(server);
      const error = await provider.generate({ messages: USER, model: "gpt-4.1" }).catch((e: unknown) => e);
      expect(error).toBeInstanceOf(ProviderBadRequestError);
      expect((error as ProviderError).retryable).toBe(false);
    }
  });

  it("maps 451 to ProviderContentFilterError", async () => {
    await newServer(() => ({ status: 451, raw: errorBody("blocked") }));
    const provider = makeProvider(server);
    await expect(provider.generate({ messages: USER, model: "gpt-4.1" })).rejects.toThrowError(
      ProviderContentFilterError,
    );
  });

  it("maps 5xx to ProviderServerError (retried by the engine)", async () => {
    await newServer(() => ({ status: 500, raw: errorBody("our fault") }));
    const provider = makeProvider(server);
    const error = (await provider
      .generate({ messages: USER, model: "gpt-4.1" })
      .catch((e: unknown) => e)) as ProviderError;
    expect(error).toBeInstanceOf(ProviderServerError);
    expect(error.retryable).toBe(true);
    expect(error.status).toBe(500);
  });

  it("redacts secrets that appear in provider error messages", async () => {
    await newServer(() => ({
      status: 401,
      raw: errorBody("Incorrect API key provided: sk-abcdef1234567890"),
    }));
    const provider = makeProvider(server);
    const error = (await provider
      .generate({ messages: USER, model: "gpt-4.1" })
      .catch((e: unknown) => e)) as Error;
    expect(error.message).not.toContain("sk-abcdef1234567890");
    expect(error.message).toContain("[REDACTED]");
  });

  it("rethrows the SDK abort error unchanged (engine maps abort reasons)", async () => {
    await newServer(() => ({ hang: true }));
    const provider = makeProvider(server);
    const controller = new AbortController();
    const promise = provider.generate({ messages: USER, model: "gpt-4.1", signal: controller.signal });
    setTimeout(() => controller.abort(), 20);
    const error = await promise.catch((e: unknown) => e);
    // The SDK's abort error keeps `name: "Error"` — assert its class.
    expect(error).toBeInstanceOf(APIUserAbortError);
  });
});

describe("parseRetryAfterMs", () => {
  it("parses seconds", () => {
    expect(parseRetryAfterMs(new Headers({ "retry-after": "3" }))).toBe(3_000);
    expect(parseRetryAfterMs(new Headers({ "retry-after": "0.5" }))).toBe(500);
  });

  it("parses HTTP dates", () => {
    const past = new Date(Date.now() - 10_000).toUTCString();
    expect(parseRetryAfterMs(new Headers({ "retry-after": past }))).toBe(0); // clamped
    const future = new Date(Date.now() + 60_000).toUTCString();
    const parsed = parseRetryAfterMs(new Headers({ "retry-after": future }));
    expect(parsed).toBeGreaterThan(0);
    expect(parsed).toBeLessThanOrEqual(60_000);
  });

  it("returns undefined for missing or garbage values", () => {
    expect(parseRetryAfterMs(undefined)).toBeUndefined();
    expect(parseRetryAfterMs(new Headers())).toBeUndefined();
    expect(parseRetryAfterMs(new Headers({ "retry-after": "soon" }))).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

describe("OpenAIProvider.stream", () => {
  it("streams text deltas and aggregates the terminal chunk", async () => {
    await newServer(() => ({
      sse: [
        chunk({ role: "assistant", content: "Hel" }),
        chunk({ content: "lo" }),
        chunk({}, "stop"),
        JSON.stringify({
          id: "chatcmpl-s1",
          object: "chat.completion.chunk",
          created: 1,
          model: "gpt-4.1",
          choices: [],
          usage: { prompt_tokens: 7, completion_tokens: 2, total_tokens: 9 },
        }),
        "[DONE]",
      ],
    }));
    const provider = makeProvider(server);
    const chunks = [];
    for await (const c of provider.stream({ messages: USER, model: "gpt-4.1" })) chunks.push(c);

    expect(chunks.map((c) => c.text)).toEqual(["Hel", "lo", ""]);
    const terminal = chunks.at(-1)!;
    expect(terminal.finishReason).toBe("stop");
    expect(terminal.usage).toEqual({ promptTokens: 7, completionTokens: 2, totalTokens: 9 });

    const body = server.requests[0]!.body as Record<string, unknown>;
    expect(body.stream).toBe(true);
    expect(body.stream_options).toEqual({ include_usage: true });
  });

  it("accumulates fragmented tool-call arguments across chunks", async () => {
    await newServer(() => ({
      sse: [
        chunk({
          tool_calls: [
            { index: 0, id: "call_9", type: "function", function: { name: "f", arguments: "" } },
          ],
        }),
        chunk({ tool_calls: [{ index: 0, function: { arguments: '{"a":' } }] }),
        chunk({ tool_calls: [{ index: 0, function: { arguments: "1}" } }] }, "tool_calls"),
        "[DONE]",
      ],
    }));
    const provider = makeProvider(server);
    const chunks = [];
    for await (const c of provider.stream({ messages: USER, model: "gpt-4.1" })) chunks.push(c);
    const terminal = chunks.at(-1)!;
    expect(terminal.toolCalls).toEqual([{ id: "call_9", name: "f", arguments: { a: 1 } }]);
    expect(terminal.finishReason).toBe("tool_calls");
  });

  it("reports malformed streamed tool-call JSON as empty arguments", async () => {
    await newServer(() => ({
      sse: [
        chunk({
          tool_calls: [
            { index: 0, id: "call_x", type: "function", function: { name: "f", arguments: "{oops" } },
          ],
        }, "tool_calls"),
        "[DONE]",
      ],
    }));
    const provider = makeProvider(server);
    const chunks = [];
    for await (const c of provider.stream({ messages: USER, model: "gpt-4.1" })) chunks.push(c);
    expect(chunks.at(-1)!.toolCalls).toEqual([{ id: "call_x", name: "f", arguments: {} }]);
  });

  it("maps HTTP errors raised while opening the stream", async () => {
    await newServer(() => ({ status: 401, raw: JSON.stringify({ error: { message: "no" } }) }));
    const provider = makeProvider(server);
    const iterator = provider.stream({ messages: USER, model: "gpt-4.1" })[Symbol.asyncIterator]();
    await expect(iterator.next()).rejects.toThrowError(ProviderAuthError);
  });
});

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

describe("OpenAIProvider embeddings", () => {
  it("embeds a single text", async () => {
    await newServer(() => ({
      body: {
        object: "list",
        model: "text-embedding-3-small",
        data: [{ object: "embedding", index: 0, embedding: [0.1, 0.2, 0.3] }],
        usage: { prompt_tokens: 2, total_tokens: 2 },
      },
    }));
    const provider = makeProvider(server);
    await expect(provider.embed("hello")).resolves.toEqual([0.1, 0.2, 0.3]);
    const request = server.requests[0]!;
    expect(request.url).toBe("/embeddings");
    // The SDK defaults to base64 encoding; the adapter decodes it transparently.
    expect(request.body).toMatchObject({
      model: "text-embedding-3-small",
      input: ["hello"],
      encoding_format: "base64",
    });
  });

  it("orders embedMany results by input index", async () => {
    await newServer((request) => {
      const input = (request.body as { input: string[] }).input;
      return {
        body: {
          object: "list",
          model: "text-embedding-3-small",
          // Deliberately out of order: the adapter must sort by index.
          data: input
            .map((text, index) => ({ object: "embedding", index, embedding: [text.length] }))
            .reverse(),
          usage: { prompt_tokens: 2, total_tokens: 2 },
        },
      };
    });
    const provider = makeProvider(server);
    await expect(provider.embedMany(["ab", "abcd"])).resolves.toEqual([[2], [4]]);
    expect((server.requests[0]!.body as { input: string[] }).input).toEqual(["ab", "abcd"]);
  });

  it("uses a configured embedding model", async () => {
    await newServer(() => ({
      body: {
        object: "list",
        data: [{ object: "embedding", index: 0, embedding: [1] }],
        usage: { prompt_tokens: 1, total_tokens: 1 },
      },
    }));
    const provider = createOpenAIProvider({
      apiKey: () => "k",
      baseUrl: server.baseUrl,
      options: { embeddingModel: "text-embedding-3-large" },
    });
    await provider.embed("x");
    expect((server.requests[0]!.body as Record<string, unknown>).model).toBe("text-embedding-3-large");
  });

  it("rejects count mismatches as malformed responses", async () => {
    await newServer(() => ({
      body: {
        object: "list",
        data: [{ object: "embedding", index: 0, embedding: [1] }],
        usage: { prompt_tokens: 1, total_tokens: 1 },
      },
    }));
    const provider = makeProvider(server);
    await expect(provider.embedMany(["a", "b"])).rejects.toSatisfy(
      (error: unknown) =>
        error instanceof ProviderError && error.code === "PROVIDER_MALFORMED_RESPONSE",
    );
  });
});

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

describe("OpenAIProvider configuration", () => {
  it("fails loudly on unknown providerOptions", () => {
    expect(() =>
      createOpenAIProvider({ apiKey: () => "k", options: { typoOption: true } }),
    ).toThrowError(ConfigurationError);
  });

  it("rejects non-http(s) baseURL in providerOptions", () => {
    expect(() =>
      createOpenAIProvider({ apiKey: () => "k", options: { baseURL: "ftp://x" } }),
    ).toThrowError(ConfigurationError);
  });

  it("requires an API key (accessor, option, or OPENAI_API_KEY)", async () => {
    const previous = process.env.OPENAI_API_KEY;
    delete process.env.OPENAI_API_KEY;
    try {
      const provider = createOpenAIProvider({ baseUrl: server.baseUrl });
      const error = await provider
        .generate({ messages: USER, model: "gpt-4.1" })
        .catch((e: unknown) => e);
      expect(error).toBeInstanceOf(ConfigurationError);
      expect((error as Error).message).toContain("OPENAI_API_KEY");
    } finally {
      if (previous !== undefined) process.env.OPENAI_API_KEY = previous;
    }
  });

  it("falls back to the OPENAI_API_KEY environment variable", async () => {
    const previous = process.env.OPENAI_API_KEY;
    process.env.OPENAI_API_KEY = "sk-env-key-99999";
    try {
      const provider = createOpenAIProvider({ baseUrl: server.baseUrl, model: "gpt-4.1" });
      await provider.generate({ messages: USER });
      expect(server.requests[0]!.headers.authorization).toBe("Bearer sk-env-key-99999");
    } finally {
      if (previous === undefined) delete process.env.OPENAI_API_KEY;
      else process.env.OPENAI_API_KEY = previous;
    }
  });

  it("supports a rotating key accessor", async () => {
    let counter = 0;
    const provider = createOpenAIProvider({
      apiKey: () => `sk-rotating-${(counter += 1)}`,
      baseUrl: server.baseUrl,
      model: "gpt-4.1",
    });
    await provider.generate({ messages: USER });
    await provider.generate({ messages: USER });
    expect(server.requests[0]!.headers.authorization).toBe("Bearer sk-rotating-1");
    expect(server.requests[1]!.headers.authorization).toBe("Bearer sk-rotating-2");
  });
});
