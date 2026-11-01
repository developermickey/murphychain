/**
 * Workflows — tools, chains, and structured output. Runs fully offline:
 * no API key, no network.
 *
 *   pnpm example:workflows
 *
 * Demonstrates: `Tool.define` with Zod-validated arguments, `executeToolCalls`
 * dispatching model-requested calls (failures serialized for self-recovery),
 * `Chain` sequencing steps over shared state with per-step timing, and
 * `generateObject` returning schema-validated structured output (with a
 * repair loop when the model's first reply is invalid).
 */
import {
  AI,
  Chain,
  ChainStepError,
  OutputValidationError,
  Tool,
  executeToolCalls,
  registerProvider,
  type AIResponse,
  type GenerateOptions,
  type ModelProvider,
  type ProviderCapabilities,
  type ToolCall,
  z,
} from "murphychain";

// ---------------------------------------------------------------------------
// 1. Tools: named, schema-validated functions the model can request.
// ---------------------------------------------------------------------------

const getWeather = Tool.define({
  name: "get_weather",
  description: "Get the current weather for a city",
  parameters: z.object({ city: z.string().min(1) }),
  execute: async ({ city }) => ({ city, tempC: 21, conditions: "clear" }),
});

const convertUnits = Tool.define({
  name: "convert_units",
  description: "Convert a temperature from Celsius to Fahrenheit",
  parameters: z.object({ celsius: z.number() }),
  execute: ({ celsius }) => ({ fahrenheit: celsius * 1.8 + 32 }),
});

console.log("── Tools ──────────────────────────────────────────────");
console.log("get_weather JSON Schema:", JSON.stringify(getWeather.normalized.jsonSchema));

// Model-requested calls (in a real app these come from response.message.toolCalls).
const requestedCalls: ToolCall[] = [
  { id: "call_1", name: "get_weather", arguments: { city: "Oslo" } },
  // A call with invalid arguments — the engine serializes the failure so the
  // model can see it and correct itself on the next turn.
  { id: "call_2", name: "convert_units", arguments: { celsius: "hot" } },
];

const toolResults = await executeToolCalls([getWeather, convertUnits], requestedCalls);
for (const result of toolResults) {
  console.log(`tool result [${result.toolCallId ?? "?"}] ${result.name}: ${result.content}`);
}

// ---------------------------------------------------------------------------
// 2. Chains: explicit, typed steps over shared state.
// ---------------------------------------------------------------------------

interface ReportState {
  city: string;
  weather?: { city: string; tempC: number; conditions: string } | undefined;
  fahrenheit?: number | undefined;
  summary?: string | undefined;
}

const chain = new Chain<ReportState>({ name: "weather-report" })
  .step("fetch", async (state) => ({
    weather: await getWeather.run({ city: state.city }),
  }))
  .step("convert", (state) => ({
    fahrenheit: state.weather === undefined ? undefined : state.weather.tempC * 1.8 + 32,
  }))
  .step("summarize", (state) => ({
    summary:
      state.weather === undefined
        ? "no data"
        : `${state.weather.city}: ${state.weather.tempC}°C / ${Math.round(state.fahrenheit ?? 0)}°F, ${state.weather.conditions}`,
  }));

console.log("\n── Chain ──────────────────────────────────────────────");
const { state, steps } = await chain.run({ city: "Oslo" });
console.log("summary:", state.summary);
console.log("timings:", steps.map((step) => `${step.name}=${step.durationMs}ms`).join(" "));

// A failing chain surfaces the failing step by name.
const failing = new Chain<ReportState>({ name: "broken" }).step("boom", () => {
  throw new Error("step exploded");
});
try {
  await failing.run({ city: "Oslo" });
} catch (error) {
  if (error instanceof ChainStepError) {
    console.log(`chain error surfaced: step="${error.step}" code=${error.code}`);
  }
}

// ---------------------------------------------------------------------------
// 3. Structured output: generateObject validates the model's JSON reply
//    against a Zod schema and repairs it if the first reply is invalid.
// ---------------------------------------------------------------------------

const personSchema = z.object({ name: z.string(), age: z.number().int().positive(), role: z.string() });

/** A model that replies with invalid JSON once, then valid JSON. */
function makeFlakyModel(): ModelProvider {
  const capabilities: ProviderCapabilities = Object.freeze({
    generate: true,
    streaming: false,
    toolCalling: false,
    embeddings: false,
    systemMessage: true,
    structuredOutput: false, // engine-level: works without provider support
  });
  let calls = 0;
  const provider: ModelProvider = {
    name: "offline-model",
    capabilities,
    async generate(_options: GenerateOptions): Promise<AIResponse> {
      calls += 1;
      const text =
        calls === 1
          ? '{"name": "Ada Lovelace", "age": "thirty-six"}' // wrong type on age
          : '{"name": "Ada Lovelace", "age": 36, "role": "mathematician"}';
      return {
        text,
        provider: provider.name,
        model: "offline-1",
        message: { role: "assistant", content: text },
        finishReason: "stop",
        usage: { promptTokens: 12, completionTokens: 9, totalTokens: 21 },
        latencyMs: 1,
      };
    },
    stream(): AsyncIterable<never> {
      throw new Error("streaming not supported by this example provider");
    },
    async embed(): Promise<number[]> {
      throw new Error("embeddings not supported by this example provider");
    },
    async embedMany(): Promise<number[][]> {
      throw new Error("embeddings not supported by this example provider");
    },
  };
  return provider;
}

console.log("\n── Structured output ───────────────────────────────────");
registerProvider("offline-model", () => makeFlakyModel());
const ai = new AI({ provider: "offline-model", model: "offline-1" });

const response = await ai.generateObject(
  "Extract the person from this sentence: Ada Lovelace was a mathematician who died at 36.",
  personSchema,
);
console.log("validated object:", JSON.stringify(response.object));
console.log("attempts:", ai.stats().requests, "(first reply failed validation and was repaired)");

// Exhausting the repair loop surfaces a typed error with the concrete issues.
const impossible = z.object({ answer: z.number().int().min(10_000) });
const stubborn = new AI({
  provider: {
    ...makeFlakyModel(),
    async generate(options: GenerateOptions): Promise<AIResponse> {
      const text = '{"answer": 1}';
      return {
        text,
        provider: "stubborn",
        model: options.model ?? "stubborn-1",
        message: { role: "assistant", content: text },
        finishReason: "stop",
        usage: { promptTokens: 1, completionTokens: 1, totalTokens: 2 },
        latencyMs: 1,
      };
    },
  },
  model: "stubborn-1",
});
try {
  await stubborn.generateObject("What is the answer?", impossible);
} catch (error) {
  if (error instanceof OutputValidationError) {
    console.log(`repair loop exhausted after ${error.attempts} attempts; issues:`);
    for (const issue of error.issues.slice(0, 2)) {
      console.log(`  - ${issue.path || "(root)"}: ${issue.message}`);
    }
  }
}

console.log("\nDone — tools, chains, and structured output, no network needed.");
