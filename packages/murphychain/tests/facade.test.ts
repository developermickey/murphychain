import { afterEach, describe, expect, it } from "vitest";
import { AI as CoreAI, listProviders, Prompt as CorePrompt } from "@murphychain/core";
import {
  AI,
  ConfigurationError,
  Model,
  Prompt,
  PromptTemplateError,
  builtinProviders,
  redactSecrets,
  registerProvider,
} from "murphychain";
import { startMockServer, type MockServer } from "./mockServer";

const servers: MockServer[] = [];

async function openAIServer(): Promise<MockServer> {
  const instance = await startMockServer(() => ({
    body: {
      id: "chatcmpl-e2e",
      object: "chat.completion",
      created: 1,
      model: "gpt-4.1",
      choices: [
        { index: 0, message: { role: "assistant", content: "OpenAI says hi" }, finish_reason: "stop" },
      ],
      usage: { prompt_tokens: 4, completion_tokens: 3, total_tokens: 7 },
    },
  }));
  servers.push(instance);
  return instance;
}

async function geminiServer(): Promise<MockServer> {
  const instance = await startMockServer(() => ({
    body: {
      candidates: [
        {
          content: { role: "model", parts: [{ text: "Gemini says hi" }] },
          finishReason: "STOP",
          index: 0,
        },
      ],
      usageMetadata: { promptTokenCount: 4, candidatesTokenCount: 3, totalTokenCount: 7 },
      modelVersion: "gemini-2.5-flash",
    },
  }));
  servers.push(instance);
  return instance;
}

afterEach(async () => {
  while (servers.length > 0) await servers.pop()?.close();
});

describe("facade surface", () => {
  it("re-exports the core engine (same class objects, not copies)", () => {
    expect(AI).toBe(CoreAI);
    expect(Prompt).toBe(CorePrompt);
  });

  it("exports the expected public API", () => {
    expect(typeof AI).toBe("function");
    expect(typeof Prompt.template).toBe("function");
    expect(typeof Model).toBe("function");
    expect(typeof redactSecrets).toBe("function");
    expect(typeof registerProvider).toBe("function");
    expect(typeof ConfigurationError).toBe("function");
  });

  it("registers the built-in providers lazily on import", () => {
    expect(listProviders()).toContain("openai");
    expect(listProviders()).toContain("gemini");
    expect(builtinProviders).toEqual(["gemini", "openai"]);
    expect(Object.isFrozen(builtinProviders)).toBe(true);
  });
});

describe("end-to-end through the registry (real adapters, mock HTTP)", () => {
  it("runs new AI({ provider: 'openai' }) → generate against a mock server", async () => {
    const server = await openAIServer();
    const ai = new AI({
      provider: "openai",
      model: "gpt-4.1",
      apiKey: "sk-facade-key",
      baseUrl: server.baseUrl,
    });
    const response = await ai.generate("Hello");
    expect(response.text).toBe("OpenAI says hi");
    expect(response.provider).toBe("openai");
    expect(response.usage.totalTokens).toBe(7);
    expect(server.requests[0]!.headers.authorization).toBe("Bearer sk-facade-key");
    expect(ai.stats().requests).toBe(1);
  });

  it("switches providers with a one-line change and keeps the response shape", async () => {
    const openai = await openAIServer();
    const gemini = await geminiServer();

    const withOpenAI = await new AI({
      provider: "openai",
      model: "gpt-4.1",
      apiKey: "sk-facade-key",
      baseUrl: openai.baseUrl,
    }).generate("Hello");

    const withGemini = await new AI({
      provider: "gemini",
      model: "gemini-2.5-flash",
      apiKey: "AIza-facade-key",
      baseUrl: gemini.baseUrl,
    }).generate("Hello");

    expect(withOpenAI.text).toBe("OpenAI says hi");
    expect(withGemini.text).toBe("Gemini says hi");
    // Application code sees an identical response contract.
    expect(Object.keys(withOpenAI).sort()).toEqual(Object.keys(withGemini).sort());
    expect(withOpenAI.usage).toEqual(withGemini.usage);
    expect(withOpenAI.finishReason).toBe(withGemini.finishReason);
    expect(gemini.requests[0]!.headers["x-goog-api-key"]).toBe("AIza-facade-key");
  });

  it("streams through the facade end to end", async () => {
    const server = await startMockServer(() => ({
      sse: [
        JSON.stringify({
          id: "c1",
          object: "chat.completion.chunk",
          created: 1,
          model: "gpt-4.1",
          choices: [{ index: 0, delta: { role: "assistant", content: "chunky" }, finish_reason: null }],
        }),
        JSON.stringify({
          id: "c2",
          object: "chat.completion.chunk",
          created: 1,
          model: "gpt-4.1",
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
          usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 },
        }),
        "[DONE]",
      ],
    }));
    servers.push(server);
    const ai = new AI({
      provider: "openai",
      model: "gpt-4.1",
      apiKey: "sk-facade-key",
      baseUrl: server.baseUrl,
    });
    const chunks = [];
    for await (const chunk of await ai.stream("Hello")) chunks.push(chunk);
    expect(chunks.map((chunk) => chunk.text)).toEqual(["chunky", ""]);
    expect(chunks[1]!.finishReason).toBe("stop");
  });

  it("reports unknown providers with actionable guidance", async () => {
    const ai = new AI({ provider: "not-a-real-provider", model: "m" });
    await expect(ai.generate("hi")).rejects.toThrowError(ConfigurationError);
    await expect(ai.generate("hi")).rejects.toThrow(/Registered providers: gemini, openai/);
  });
});

describe("facade DX primitives", () => {
  it("Prompt templates compose with generate()", async () => {
    const server = await openAIServer();
    const ai = new AI({ provider: "openai", model: "gpt-4.1", apiKey: "sk-k", baseUrl: server.baseUrl });
    const template = Prompt.template("Summarize {topic} in {words} words.");
    const response = await ai.generate(template.format({ topic: "gravity", words: 20 }));
    expect(response.text).toBe("OpenAI says hi");
    expect(template.variables).toEqual(["topic", "words"]);
    expect(() => template.format({ topic: "gravity" })).toThrowError(PromptTemplateError);
  });

  it("Model defaults flow into requests", async () => {
    const server = await openAIServer();
    const ai = new AI({ provider: "openai", model: "gpt-4.1", apiKey: "sk-k", baseUrl: server.baseUrl });
    await ai.generate({ prompt: "hi", model: new Model("gpt-4.1-mini", { temperature: 0 }) });
    const body = server.requests[0]!.body as Record<string, unknown>;
    expect(body.model).toBe("gpt-4.1-mini");
    expect(body.temperature).toBe(0);
  });
});
