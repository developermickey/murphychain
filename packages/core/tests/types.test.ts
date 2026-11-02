import { describe, expectTypeOf, it } from "vitest";
import { z } from "zod";
import {
  AI,
  EventBus,
  Model,
  Prompt,
  registerProvider,
  type AIChunk,
  type AIConfig,
  type AIEventMap,
  type AIResponse,
  type AIStats,
  type FinishReason,
  type Message,
  type ModelProvider,
  type MurphyChainError,
  type ProviderCapabilities,
  type RetryPolicy,
  type TokenUsage,
} from "@murphychain/core";

/**
 * Compile-time contract tests for the public type surface.
 *
 * Every assertion lives in `typeContracts`, which is never called: the file
 * has zero runtime side effects (no clients constructed, no network, no
 * module-scope executions). `tsc --noEmit` typechecks the whole file,
 * so the assertions below are enforced by the typecheck script.
 */
function typeContracts(): void {
  // -- AI method signatures ------------------------------------------------
  const ai = new AI({ provider: "openai", model: "gpt-4.1", apiKey: "k" });
  expectTypeOf(ai.generate("hi")).toExtend<Promise<AIResponse>>();
  expectTypeOf(ai.generate([{ role: "user", content: "hi" }])).toExtend<Promise<AIResponse>>();
  expectTypeOf(
    ai.generate({ prompt: "hi", temperature: 0.5, toolChoice: "auto" }),
  ).toExtend<Promise<AIResponse>>();
  expectTypeOf(ai.stream("hi")).toExtend<Promise<AsyncIterable<AIChunk>>>();
  expectTypeOf(ai.embed("text")).toExtend<Promise<number[]>>();
  expectTypeOf(ai.embedMany(["a", "b"])).toExtend<Promise<number[][]>>();
  expectTypeOf(ai.stats()).toExtend<AIStats>();

  // -- Call options --------------------------------------------------------
  const controller = new AbortController();
  void ai.generate("hi", {
    signal: controller.signal,
    timeoutMs: 1_000,
    metadata: { traceId: "t" },
    includeRaw: false,
  });

  // -- Strict config rejects unknown keys (compile-time) -------------------
  // @ts-expect-error -- strict config must reject unknown keys
  void new AI({ provider: "p", modle: "typo" });
  // @ts-expect-error -- provider is required
  void new AI({ model: "m" });

  // -- Event typing --------------------------------------------------------
  ai.on("request:start", (event) => {
    expectTypeOf(event.messageCount).toExtend<number>();
    expectTypeOf(event.metadata).toExtend<Readonly<Record<string, unknown>>>();
  });
  ai.on("chunk", (event) => {
    expectTypeOf(event.index).toExtend<number>();
  });
  ai.on("request:success", (event) => {
    expectTypeOf(event.usage).toExtend<TokenUsage>();
    expectTypeOf(event.finishReason).toExtend<FinishReason>();
  });
  ai.on("request:error", (event) => {
    expectTypeOf(event.error).toExtend<MurphyChainError>();
  });
  ai.on("abort", (event) => {
    expectTypeOf(event.reason).toExtend<"timeout" | "user">();
  });
  // @ts-expect-error -- "nope" is not an engine event
  ai.on("nope", () => undefined);

  const bus = new EventBus<AIEventMap>();
  void bus;

  // -- Generalized event bus (custom maps) --------------------------------
  const customBus = new EventBus<{ ping: { n: number }; pong: { ok: boolean } }>();
  customBus.on("ping", (event) => {
    expectTypeOf(event.n).toExtend<number>();
  });
  customBus.once("pong", (event) => {
    expectTypeOf(event.ok).toExtend<boolean>();
  });
  customBus.on(
    "ping",
    // @ts-expect-error -- "ping" payloads have no "ok" property
    (event) => expectTypeOf(event.ok).toExtend<boolean>(),
  );
  void customBus;

  // -- Response and message shapes ----------------------------------------
  expectTypeOf<AIResponse["text"]>().toExtend<string>();
  expectTypeOf<AIResponse["provider"]>().toExtend<string>();
  expectTypeOf<AIResponse["message"]>().toExtend<Message>();
  expectTypeOf<AIResponse["usage"]>().toExtend<TokenUsage>();
  expectTypeOf<AIResponse["finishReason"]>().toExtend<FinishReason>();
  expectTypeOf<AIResponse["latencyMs"]>().toExtend<number>();
  expectTypeOf<AIResponse["raw"]>().toExtend<unknown | undefined>();

  expectTypeOf<AIChunk["index"]>().toExtend<number>();
  expectTypeOf<AIChunk["text"]>().toExtend<string>();

  // -- Prompt and Model ----------------------------------------------------
  const template = Prompt.template("{a} {b}");
  expectTypeOf(template.format({ a: 1, b: true })).toExtend<string>();
  expectTypeOf(template.variables).toExtend<readonly string[]>();
  // @ts-expect-error -- values must be string | number | boolean
  void template.format({ a: { nested: true }, b: "x" });

  const model = new Model("m", { temperature: 0.5 });
  expectTypeOf(model.name).toExtend<string>();
  expectTypeOf(model.defaults).toExtend<Readonly<{ temperature?: number | undefined }>>();
  expectTypeOf(model.withDefaults({ maxTokens: 10 })).toExtend<Model>();

  // -- Provider contract ---------------------------------------------------
  const capabilities: ProviderCapabilities = {
    generate: true,
    streaming: false,
    toolCalling: false,
    embeddings: false,
    systemMessage: true,
    structuredOutput: false,
  };
  const provider: ModelProvider = {
    name: "typed",
    capabilities,
    generate: async () => ({
      text: "",
      provider: "typed",
      model: "m",
      message: { role: "assistant", content: "" },
      finishReason: "stop",
      usage: { promptTokens: 0, completionTokens: 0, totalTokens: 0 },
      latencyMs: 0,
    }),
    stream: () => (async function* () {})(),
    embed: async () => [0.1],
    embedMany: async () => [[0.1]],
  };
  registerProvider("typed-sync", () => provider);
  registerProvider("typed-async", async () => provider);

  // -- Config optionality (exactOptionalPropertyTypes) ---------------------
  expectTypeOf<AIConfig["temperature"]>().toExtend<number | undefined>();
  expectTypeOf<AIConfig["providerOptions"]>().toExtend<Record<string, unknown> | undefined>();

  // -- Zod interop ---------------------------------------------------------
  const tool = { name: "t", description: "d", parameters: z.object({ q: z.string() }) };
  expectTypeOf(tool.parameters).toExtend<z.ZodType>();

  // -- Retry policy --------------------------------------------------------
  expectTypeOf<RetryPolicy["retries"]>().toExtend<number>();
  expectTypeOf<RetryPolicy["jitter"]>().toExtend<boolean>();
}

describe("public type surface", () => {
  it("is compile-time only (never executed)", () => {
    // The contract above is enforced by `tsc --noEmit`, not at runtime.
    expectTypeOf(typeContracts).toExtend<() => void>();
  });
});

// Ensure the compiler keeps `typeContracts` analyzed without calling it.
void typeContracts;
