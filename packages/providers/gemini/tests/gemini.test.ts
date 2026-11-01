import { afterEach, beforeEach, describe, expect, it } from "vitest";
import {
  ConfigurationError,
  ProviderAuthError,
  ProviderBadRequestError,
  ProviderContentFilterError,
  ProviderError,
  ProviderRateLimitError,
  ProviderServerError,
} from "@murphychain/core";
import type { Message, ModelProvider, ProviderFactoryConfig } from "@murphychain/core";
import { createGeminiProvider } from "@murphychain/gemini";
import { startMockServer, type MockServer } from "./mockServer";

const USER: Message[] = [{ role: "user", content: "hi" }];

const GENERATION = {
  candidates: [
    {
      content: { role: "model", parts: [{ text: "Hello from Gemini" }] },
      finishReason: "STOP",
      index: 0,
    },
  ],
  usageMetadata: { promptTokenCount: 8, candidatesTokenCount: 4, totalTokenCount: 12 },
  modelVersion: "gemini-2.5-flash",
};

/** Builds one SSE `data:` payload shaped like a GenerateContentResponse. */
function sseChunk(payload: unknown): string {
  return JSON.stringify(payload);
}

function makeProvider(server: MockServer, overrides: Partial<ProviderFactoryConfig> = {}): ModelProvider {
  return createGeminiProvider({
    apiKey: () => "AIza-test-key-123",
    baseUrl: server.baseUrl,
    model: "gemini-2.5-flash",
    ...overrides,
  });
}

const GEMINI_ENV = ["GEMINI_API_KEY", "GOOGLE_API_KEY"] as const;

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

/** Temporarily clears the Gemini key environment variables. */
async function withoutEnvKeys<T>(run: () => Promise<T> | T): Promise<T> {
  const saved = GEMINI_ENV.map((name) => [name, process.env[name]] as const);
  for (const [name] of saved) delete process.env[name];
  try {
    return await run();
  } finally {
    for (const [name, value] of saved) {
      if (value !== undefined) process.env[name] = value;
    }
  }
}

beforeEach(async () => {
  spawned.length = 0;
  await newServer(() => ({ body: GENERATION }));
});

afterEach(async () => {
  for (const instance of spawned) await instance.close();
});

// ---------------------------------------------------------------------------
// generate
// ---------------------------------------------------------------------------

describe("GeminiProvider.generate", () => {
  it("sends a correctly shaped request and maps the response", async () => {
    const provider = makeProvider(server);
    const response = await provider.generate({
      messages: [
        { role: "system", content: "be nice" },
        { role: "user", content: "hi" },
      ],
      model: "gemini-2.5-flash",
      temperature: 0.2,
      maxTokens: 100,
      topP: 0.9,
      stop: ["END"],
    });

    expect(response.text).toBe("Hello from Gemini");
    expect(response.provider).toBe("gemini");
    expect(response.model).toBe("gemini-2.5-flash");
    expect(response.finishReason).toBe("stop");
    expect(response.usage).toEqual({ promptTokens: 8, completionTokens: 4, totalTokens: 12 });
    expect(response.raw).toBeDefined();

    const request = server.requests[0]!;
    expect(request.method).toBe("POST");
    expect(request.url).toBe("/v1beta/models/gemini-2.5-flash:generateContent");
    expect(request.headers["x-goog-api-key"]).toBe("AIza-test-key-123");

    const body = request.body as Record<string, unknown>;
    expect(body.contents).toEqual([{ role: "user", parts: [{ text: "hi" }] }]);
    // System messages move to systemInstruction, never into contents.
    expect((body.systemInstruction as { parts: Array<{ text: string }> }).parts[0]!.text).toBe(
      "be nice",
    );
    expect(body.generationConfig).toEqual({
      temperature: 0.2,
      maxOutputTokens: 100,
      topP: 0.9,
      stopSequences: ["END"],
    });
  });

  it("joins multiple system messages with blank lines", async () => {
    const provider = makeProvider(server);
    await provider.generate({
      messages: [
        { role: "system", content: "Rule A" },
        { role: "system", content: "Rule B" },
        { role: "user", content: "hi" },
      ],
      model: "gemini-2.5-flash",
    });
    const body = server.requests[0]!.body as { systemInstruction: { parts: Array<{ text: string }> } };
    expect(body.systemInstruction.parts[0]!.text).toBe("Rule A\n\nRule B");
  });

  it("maps assistant tool calls and groups consecutive tool results", async () => {
    const provider = makeProvider(server);
    await provider.generate({
      messages: [
        { role: "user", content: "lookup x and y" },
        {
          role: "assistant",
          content: "",
          toolCalls: [
            { id: "c1", name: "lookup", arguments: { q: "x" } },
            { id: "c2", name: "lookup", arguments: { q: "y" } },
          ],
        },
        { role: "tool", content: '{"result":1}', name: "lookup", toolCallId: "c1" },
        { role: "tool", content: "42", name: "lookup", toolCallId: "c2" },
      ],
      model: "gemini-2.5-flash",
    });
    const body = server.requests[0]!.body as {
      contents: Array<{ role: string; parts: Array<Record<string, unknown>> }>;
    };
    expect(body.contents).toHaveLength(3); // user turn, model turn, grouped results turn
    expect(body.contents[0]).toEqual({ role: "user", parts: [{ text: "lookup x and y" }] });
    expect(body.contents[1]!.role).toBe("model");
    expect(body.contents[1]!.parts).toHaveLength(2);
    expect(body.contents[1]!.parts[0]!.functionCall).toMatchObject({ name: "lookup", args: { q: "x" } });
    // Consecutive tool results share one user turn with two functionResponse parts.
    expect(body.contents[2]!.role).toBe("user");
    expect(body.contents[2]!.parts).toHaveLength(2);
    const responses = body.contents[2]!.parts.map(
      (part) => part.functionResponse as { name: string; response: Record<string, unknown> },
    );
    expect(responses[0]).toMatchObject({ name: "lookup", response: { result: 1 } });
    // Non-object JSON results follow Gemini's documented `{ output }` wrapping.
    expect(responses[1]).toMatchObject({ name: "lookup", response: { output: 42 } });
  });

  it("maps tools with JSON-schema declarations and named tool choice", async () => {
    const provider = makeProvider(server);
    await provider.generate({
      messages: USER,
      model: "gemini-2.5-flash",
      tools: [
        {
          name: "get_weather",
          description: "Get weather",
          jsonSchema: { type: "object", properties: { city: { type: "string" } } },
        },
      ],
      toolChoice: { name: "get_weather" },
    });
    const body = server.requests[0]!.body as {
      tools: Array<{ functionDeclarations: Array<Record<string, unknown>> }>;
      toolConfig: { functionCallingConfig: { mode: string; allowedFunctionNames: string[] } };
    };
    expect(body.tools[0]!.functionDeclarations[0]).toEqual({
      name: "get_weather",
      description: "Get weather",
      parametersJsonSchema: { type: "object", properties: { city: { type: "string" } } },
    });
    expect(body.toolConfig.functionCallingConfig).toEqual({
      mode: "ANY",
      allowedFunctionNames: ["get_weather"],
    });
  });

  it("maps string tool choices to function calling modes", async () => {
    for (const [choice, mode] of [
      ["auto", "AUTO"],
      ["none", "NONE"],
      ["required", "ANY"],
    ] as const) {
      await newServer(() => ({ body: GENERATION }));
      const provider = makeProvider(server);
      await provider.generate({
        messages: USER,
        model: "gemini-2.5-flash",
        tools: [{ name: "t", description: "d", jsonSchema: { type: "object" } }],
        toolChoice: choice,
      });
      const body = server.requests[0]!.body as {
        toolConfig: { functionCallingConfig: { mode: string } };
      };
      expect(body.toolConfig.functionCallingConfig.mode).toBe(mode);
    }
  });

  it("parses function calls and synthesizes ids when absent", async () => {
    await newServer(() => ({
      body: {
        candidates: [
          {
            content: {
              role: "model",
              parts: [
                { functionCall: { name: "get_weather", args: { city: "Paris" } } },
                { functionCall: { id: "gemini_id", name: "lookup", args: { q: "x" } } },
              ],
            },
            finishReason: "STOP",
            index: 0,
          },
        ],
        usageMetadata: { promptTokenCount: 5, candidatesTokenCount: 3, totalTokenCount: 8 },
      },
    }));
    const provider = makeProvider(server);
    const response = await provider.generate({ messages: USER, model: "gemini-2.5-flash" });
    expect(response.text).toBe("");
    expect(response.message.toolCalls).toEqual([
      { id: "gemini_call_0", name: "get_weather", arguments: { city: "Paris" } },
      { id: "gemini_id", name: "lookup", arguments: { q: "x" } },
    ]);
  });

  it("excludes thought parts from the text", async () => {
    await newServer(() => ({
      body: {
        candidates: [
          {
            content: {
              role: "model",
              parts: [
                { text: "thinking about it...", thought: true },
                { text: "The answer is 4." },
              ],
            },
            finishReason: "STOP",
            index: 0,
          },
        ],
        usageMetadata: { promptTokenCount: 1, candidatesTokenCount: 1, totalTokenCount: 2 },
      },
    }));
    const provider = makeProvider(server);
    const response = await provider.generate({ messages: USER, model: "gemini-2.5-flash" });
    expect(response.text).toBe("The answer is 4.");
  });

  it("maps finish reasons onto the unified vocabulary", async () => {
    const cases: Array<[string, string]> = [
      ["STOP", "stop"],
      ["MAX_TOKENS", "length"],
      ["SAFETY", "content_filter"],
      ["RECITATION", "content_filter"],
      ["MALFORMED_FUNCTION_CALL", "error"],
      ["OTHER", "unknown"],
    ];
    for (const [wire, expected] of cases) {
      await newServer(() => ({
        body: {
          candidates: [
            {
              content: { role: "model", parts: [{ text: "x" }] },
              finishReason: wire,
              index: 0,
            },
          ],
          usageMetadata: { promptTokenCount: 1, candidatesTokenCount: 1, totalTokenCount: 2 },
        },
      }));
      const provider = makeProvider(server);
      const response = await provider.generate({ messages: USER, model: "gemini-2.5-flash" });
      expect(response.finishReason).toBe(expected);
    }
  });

  it("rejects safety-blocked prompts as content filter errors", async () => {
    await newServer(() => ({ body: { promptFeedback: { blockReason: "SAFETY" } } }));
    const provider = makeProvider(server);
    const error = await provider
      .generate({ messages: USER, model: "gemini-2.5-flash" })
      .catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ProviderContentFilterError);
    expect((error as Error).message).toContain("SAFETY");
  });

  it("counts thinking tokens into completion usage", async () => {
    await newServer(() => ({
      body: {
        ...GENERATION,
        usageMetadata: {
          promptTokenCount: 8,
          candidatesTokenCount: 4,
          thoughtsTokenCount: 6,
          totalTokenCount: 18,
        },
      },
    }));
    const provider = makeProvider(server);
    const response = await provider.generate({ messages: USER, model: "gemini-2.5-flash" });
    expect(response.usage).toEqual({ promptTokens: 8, completionTokens: 10, totalTokens: 18 });
  });

  it("requires a model", async () => {
    const provider = createGeminiProvider({ apiKey: () => "k", baseUrl: server.baseUrl });
    await expect(provider.generate({ messages: USER })).rejects.toThrowError(ConfigurationError);
  });
});

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

describe("GeminiProvider error mapping", () => {
  const errorBody = (message: string): string =>
    JSON.stringify({ error: { code: 403, message, status: "PERMISSION_DENIED" } });

  it("maps 401/403 to ProviderAuthError (never retried)", async () => {
    for (const status of [401, 403]) {
      await newServer(() => ({ status, raw: errorBody("permission denied") }));
      const provider = makeProvider(server);
      const error = await provider
        .generate({ messages: USER, model: "gemini-2.5-flash" })
        .catch((e: unknown) => e);
      expect(error).toBeInstanceOf(ProviderAuthError);
      expect((error as ProviderError).status).toBe(status);
      expect((error as ProviderError).retryable).toBe(false);
    }
  });

  it("maps 429 to ProviderRateLimitError (retried by the engine)", async () => {
    await newServer(() => ({ status: 429, raw: errorBody("quota exceeded") }));
    const provider = makeProvider(server);
    const error = (await provider
      .generate({ messages: USER, model: "gemini-2.5-flash" })
      .catch((e: unknown) => e)) as ProviderError;
    expect(error).toBeInstanceOf(ProviderRateLimitError);
    expect(error.retryable).toBe(true);
  });

  it("maps 400/404/422 to ProviderBadRequestError", async () => {
    for (const status of [400, 404, 422]) {
      await newServer(() => ({ status, raw: errorBody("bad request") }));
      const provider = makeProvider(server);
      const error = await provider
        .generate({ messages: USER, model: "gemini-2.5-flash" })
        .catch((e: unknown) => e);
      expect(error).toBeInstanceOf(ProviderBadRequestError);
      expect((error as ProviderError).retryable).toBe(false);
    }
  });

  it("maps 451 to ProviderContentFilterError", async () => {
    await newServer(() => ({ status: 451, raw: errorBody("blocked by legal") }));
    const provider = makeProvider(server);
    await expect(provider.generate({ messages: USER, model: "gemini-2.5-flash" })).rejects.toThrowError(
      ProviderContentFilterError,
    );
  });

  it("maps 5xx to ProviderServerError (retried by the engine)", async () => {
    await newServer(() => ({ status: 503, raw: errorBody("backend unavailable") }));
    const provider = makeProvider(server);
    const error = (await provider
      .generate({ messages: USER, model: "gemini-2.5-flash" })
      .catch((e: unknown) => e)) as ProviderError;
    expect(error).toBeInstanceOf(ProviderServerError);
    expect(error.retryable).toBe(true);
    expect(error.status).toBe(503);
  });

  it("redacts secrets that appear in provider error messages", async () => {
    await newServer(() => ({
      status: 400,
      raw: errorBody("API key not valid: AIzaSyFakeKey12345"),
    }));
    const provider = makeProvider(server);
    const error = (await provider
      .generate({ messages: USER, model: "gemini-2.5-flash" })
      .catch((e: unknown) => e)) as Error;
    expect(error.message).not.toContain("AIzaSyFakeKey12345");
    expect(error.message).toContain("[REDACTED]");
  });
});

// ---------------------------------------------------------------------------
// Streaming
// ---------------------------------------------------------------------------

describe("GeminiProvider.stream", () => {
  it("streams text chunks and aggregates the terminal chunk", async () => {
    await newServer(() => ({
      sse: [
        sseChunk({
          candidates: [{ content: { role: "model", parts: [{ text: "Gem" }] }, index: 0 }],
        }),
        sseChunk({
          candidates: [
            {
              content: { role: "model", parts: [{ text: "ini" }] },
              finishReason: "STOP",
              index: 0,
            },
          ],
          usageMetadata: { promptTokenCount: 3, candidatesTokenCount: 2, totalTokenCount: 5 },
        }),
      ],
    }));
    const provider = makeProvider(server);
    const chunks = [];
    for await (const c of provider.stream({ messages: USER, model: "gemini-2.5-flash" })) {
      chunks.push(c);
    }
    expect(chunks.map((c) => c.text)).toEqual(["Gem", "ini", ""]);
    const terminal = chunks.at(-1)!;
    expect(terminal.finishReason).toBe("stop");
    expect(terminal.usage).toEqual({ promptTokens: 3, completionTokens: 2, totalTokens: 5 });

    expect(server.requests[0]!.url).toBe(
      "/v1beta/models/gemini-2.5-flash:streamGenerateContent?alt=sse",
    );
  });

  it("emits tool calls carried in streamed parts", async () => {
    await newServer(() => ({
      sse: [
        sseChunk({
          candidates: [
            {
              content: {
                role: "model",
                parts: [{ functionCall: { name: "lookup", args: { q: "x" } } }],
              },
              finishReason: "STOP",
              index: 0,
            },
          ],
        }),
      ],
    }));
    const provider = makeProvider(server);
    const chunks = [];
    for await (const c of provider.stream({ messages: USER, model: "gemini-2.5-flash" })) {
      chunks.push(c);
    }
    const toolChunk = chunks.find((c) => c.toolCalls !== undefined);
    // Streamed ids embed the chunk index, keeping them unique across chunks.
    expect(toolChunk?.toolCalls).toEqual([
      { id: "gemini_call_0_0", name: "lookup", arguments: { q: "x" } },
    ]);
    expect(chunks.at(-1)!.finishReason).toBe("stop");
  });

  it("rejects a safety-blocked stream", async () => {
    // Safety blocks arrive as a first SSE event with promptFeedback and no
    // candidates — the adapter must fail instead of yielding empty text.
    await newServer(() => ({ sse: [sseChunk({ promptFeedback: { blockReason: "SAFETY" } })] }));
    const provider = makeProvider(server);
    const iterator = provider.stream({ messages: USER, model: "gemini-2.5-flash" })[Symbol.asyncIterator]();
    await expect(iterator.next()).rejects.toThrowError(ProviderContentFilterError);
  });
});

// ---------------------------------------------------------------------------
// Embeddings (the SDK batches through :batchEmbedContents in API-key mode)
// ---------------------------------------------------------------------------

describe("GeminiProvider embeddings", () => {
  it("embeds a single text through batchEmbedContents", async () => {
    await newServer(() => ({ body: { embeddings: [{ values: [0.1, 0.2, 0.3] }] } }));
    const provider = makeProvider(server);
    await expect(provider.embed("hello")).resolves.toEqual([0.1, 0.2, 0.3]);

    const request = server.requests[0]!;
    expect(request.url).toBe("/v1beta/models/text-embedding-004:batchEmbedContents");
    expect(request.headers["x-goog-api-key"]).toBe("AIza-test-key-123");
    const body = request.body as {
      requests: Array<{ model: string; content: { parts: Array<{ text: string }> } }>;
    };
    expect(body.requests).toHaveLength(1);
    expect(body.requests[0]!.model).toBe("models/text-embedding-004");
    expect(body.requests[0]!.content.parts[0]!.text).toBe("hello");
  });

  it("embeds batches concurrently while preserving input order", async () => {
    await newServer((request) => {
      const text = (
        request.body as { requests: Array<{ content: { parts: Array<{ text: string }> } }> }
      ).requests[0]!.content.parts[0]!.text;
      return { body: { embeddings: [{ values: [text.charCodeAt(0)] }] } };
    });
    const provider = makeProvider(server);
    await expect(provider.embedMany(["a", "b", "c"])).resolves.toEqual([[97], [98], [99]]);
    expect(server.requests).toHaveLength(3);
  });

  it("uses a configured embedding model", async () => {
    await newServer((request) => {
      const model = (request.body as { requests: Array<{ model: string }> }).requests[0]!.model;
      return { body: { embeddings: [{ values: [model.length] }] } };
    });
    const provider = createGeminiProvider({
      apiKey: () => "k",
      baseUrl: server.baseUrl,
      options: { embeddingModel: "gemini-embedding-001" },
    });
    await provider.embed("x");
    const body = server.requests[0]!.body as { requests: Array<{ model: string }> };
    expect(body.requests[0]!.model).toBe("models/gemini-embedding-001");
  });

  it("rejects empty embeddings as malformed responses", async () => {
    await newServer(() => ({ body: { embeddings: [{}] } }));
    const provider = makeProvider(server);
    await expect(provider.embed("x")).rejects.toSatisfy(
      (error: unknown) =>
        error instanceof ProviderError && error.code === "PROVIDER_MALFORMED_RESPONSE",
    );
  });
});

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

describe("GeminiProvider configuration", () => {
  it("fails loudly on unknown providerOptions", () => {
    expect(() =>
      createGeminiProvider({ apiKey: () => "k", options: { typoOption: true } }),
    ).toThrowError(ConfigurationError);
  });

  it("rejects non-http(s) baseUrl in providerOptions", () => {
    expect(() =>
      createGeminiProvider({ apiKey: () => "k", options: { baseUrl: "ftp://x" } }),
    ).toThrowError(ConfigurationError);
  });

  it("requires an API key (accessor, option, or GEMINI_API_KEY/GOOGLE_API_KEY)", async () => {
    await withoutEnvKeys(async () => {
      const provider = createGeminiProvider({ baseUrl: server.baseUrl });
      await expect(
        provider.generate({ messages: USER, model: "gemini-2.5-flash" }),
      ).rejects.toThrowError(ConfigurationError);
      await expect(
        provider.generate({ messages: USER, model: "gemini-2.5-flash" }),
      ).rejects.toThrow(/GEMINI_API_KEY/);
    });
  });

  it("falls back to the GEMINI_API_KEY environment variable", async () => {
    const previous = process.env.GEMINI_API_KEY;
    process.env.GEMINI_API_KEY = "AIza-env-key-999";
    try {
      const provider = createGeminiProvider({ baseUrl: server.baseUrl, model: "gemini-2.5-flash" });
      await provider.generate({ messages: USER });
      expect(server.requests[0]!.headers["x-goog-api-key"]).toBe("AIza-env-key-999");
    } finally {
      if (previous === undefined) delete process.env.GEMINI_API_KEY;
      else process.env.GEMINI_API_KEY = previous;
    }
  });

  it("supports a rotating key accessor", async () => {
    let counter = 0;
    const provider = createGeminiProvider({
      apiKey: () => `AIza-rotating-${(counter += 1)}`,
      baseUrl: server.baseUrl,
      model: "gemini-2.5-flash",
    });
    await provider.generate({ messages: USER });
    await provider.generate({ messages: USER });
    expect(server.requests[0]!.headers["x-goog-api-key"]).toBe("AIza-rotating-1");
    expect(server.requests[1]!.headers["x-goog-api-key"]).toBe("AIza-rotating-2");
  });

  it("honours a custom apiVersion", async () => {
    const provider = makeProvider(server, { options: { apiVersion: "v1" } });
    await provider.generate({ messages: USER, model: "gemini-2.5-flash" });
    expect(server.requests[0]!.url).toBe("/v1/models/gemini-2.5-flash:generateContent");
  });
});
