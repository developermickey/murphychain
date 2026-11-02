/**
 * Agent runtime — a bounded tool-calling loop with memory. Runs fully
 * offline: no API key, no network.
 *
 *   pnpm example:agent
 *
 * Demonstrates: `Agent` executing model-requested tool calls until the model
 * answers (or the step budget is spent), content-free `step`/`tool`/`finish`
 * events, `BufferMemory` carrying conversation history across runs, and the
 * guarantees that a final step's tool calls are never executed and the system
 * prompt is never stored in memory.
 */
import {
  AI,
  Agent,
  BufferMemory,
  Tool,
  type AIResponse,
  type GenerateOptions,
  type Message,
  type ModelProvider,
  type ProviderCapabilities,
  z,
} from "murphychain";

// ---------------------------------------------------------------------------
// 1. An offline model: scripted replies that drive the agent loop.
// ---------------------------------------------------------------------------

const CAPABILITIES: ProviderCapabilities = Object.freeze({
  generate: true,
  streaming: false,
  toolCalling: true,
  embeddings: false,
  systemMessage: true,
  structuredOutput: false,
});

function reply(text: string): AIResponse {
  return {
    text,
    provider: "offline-agent",
    model: "offline-1",
    message: { role: "assistant", content: text },
    finishReason: "stop",
    usage: { promptTokens: 20, completionTokens: 12, totalTokens: 32 },
    latencyMs: 1,
  };
}

function requestTool(
  name: string,
  args: Record<string, unknown>,
  id: string,
): AIResponse {
  return {
    text: "",
    provider: "offline-agent",
    model: "offline-1",
    message: {
      role: "assistant",
      content: "",
      toolCalls: [{ id, name, arguments: args }],
    },
    finishReason: "tool_calls",
    usage: { promptTokens: 20, completionTokens: 8, totalTokens: 28 },
    latencyMs: 1,
  };
}

interface OfflineModel {
  provider: ModelProvider;
  /** Messages of every generate call — lets the example verify what was sent. */
  calls: Message[][];
}

/** Wraps a scripted `(callIndex) => response` function as a ModelProvider. */
function offlineModel(script: (callIndex: number) => AIResponse): OfflineModel {
  const calls: Message[][] = [];
  const provider: ModelProvider = {
    name: "offline-agent",
    capabilities: CAPABILITIES,
    async generate(options: GenerateOptions): Promise<AIResponse> {
      const response = script(calls.length);
      calls.push(options.messages);
      return response;
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
  return { provider, calls };
}

const getWeather = Tool.define({
  name: "get_weather",
  description: "Get the current weather for a city",
  parameters: z.object({ city: z.string().min(1) }),
  execute: async ({ city }) => ({ city, tempC: 21, conditions: "clear" }),
});

// ---------------------------------------------------------------------------
// 2. The loop: model asks for a tool, agent executes it, model answers.
// ---------------------------------------------------------------------------

// Run 1: step 0 requests the tool, step 1 answers with the result.
// Run 2 (callIndex 2): a plain follow-up answer.
const model = offlineModel((callIndex) => {
  if (callIndex === 0) return requestTool("get_weather", { city: "Oslo" }, "call_1");
  if (callIndex === 1) return reply("It is 21°C and clear in Oslo.");
  return reply("Still 21°C — you asked about Oslo earlier.");
});

const memory = new BufferMemory({ maxMessages: 20 });
const agent = new Agent({
  name: "weather-agent",
  system: "You answer weather questions using the tools provided.",
  ai: new AI({ provider: model.provider, model: "offline-1" }),
  tools: [getWeather],
  memory,
});

// Events are content-free: names, counts, timings — never message text.
agent.on("step", (event) =>
  console.log(`step ${event.index}: toolCalls=${event.toolCalls} latency=${event.latencyMs}ms`),
);
agent.on("tool", (event) =>
  console.log(`tool ${event.name}: ok=${event.ok} in ${event.durationMs}ms`),
);
agent.on("finish", (event) =>
  console.log(
    `finish: steps=${event.steps} stopped=${event.stopped} tokens=${event.usage.totalTokens}`,
  ),
);

console.log("── Agent loop ─────────────────────────────────────────");
const first = await agent.run("What is the weather in Oslo?");
console.log("answer:", first.response.text);
console.log("tool result:", first.steps[0]!.toolMessages[0]!.content);
console.log("stopped:", first.stopped, "| total tokens:", first.usage.totalTokens);

if (first.stopped !== "completed" || first.steps.length !== 2) {
  throw new Error("agent loop contract violated: expected 2 steps, completed");
}

// ---------------------------------------------------------------------------
// 3. Memory: the second run replays the executed conversation.
// ---------------------------------------------------------------------------

console.log("\n── Memory ─────────────────────────────────────────────");
console.log(
  "remembered after run 1:",
  memory.messages.map((m) => m.role).join(" → "),
);

const second = await agent.run("And what did I ask about before?");
console.log("second answer:", second.response.text);
console.log(
  "second run sent:",
  model.calls[2]!.map((m) => m.role).join(" → "),
);

// The system prompt is re-seeded each run, never stored in memory; the
// remembered history sits between it and the new input.
const remembered = memory.messages.map((m) => m.role);
if (remembered.includes("system")) {
  throw new Error("memory contract violated: system prompt was persisted");
}
const replayed = model.calls[2]!;
if (replayed[0]!.role !== "system" || replayed[replayed.length - 1]!.role !== "user") {
  throw new Error("seed contract violated: expected system … user");
}

// ---------------------------------------------------------------------------
// 4. The bound: a model that never answers exhausts maxSteps — and the final
//    step's tool calls are NOT executed (nothing would consume them).
// ---------------------------------------------------------------------------

console.log("\n── Bounded loop ───────────────────────────────────────");
let executed = 0;
const loopTool = Tool.define({
  name: "tick",
  description: "Never returns an answer.",
  parameters: z.object({}),
  execute: async () => {
    executed += 1;
    return "tick";
  },
});

const looping = offlineModel(() => requestTool("tick", {}, `call_${executed + 1}`));
const looper = new Agent({
  name: "looper",
  ai: new AI({ provider: looping.provider, model: "offline-1" }),
  tools: [loopTool],
  maxSteps: 3,
});

const exhausted = await looper.run("keep going");
console.log(
  `stopped=${exhausted.stopped} steps=${exhausted.steps.length} toolExecutions=${executed}`,
);

if (exhausted.stopped !== "max_steps" || executed !== 2) {
  throw new Error(
    `bound contract violated: expected max_steps with 2 executions, got ${exhausted.stopped} with ${executed}`,
  );
}

console.log("\nDone — agent loop, memory, and bounded execution, no network needed.");
