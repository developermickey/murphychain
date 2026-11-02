/**
 * MCP integration — connect to a Model Context Protocol server and use its
 * tools in MurphyChain. Runs fully offline: no API key, no network. The MCP
 * server is a local subprocess spawned over stdio.
 *
 *   pnpm example:mcp
 *
 * Demonstrates: `connectStdio` spawning a server, `mcp.tools()` adapting
 * remote tools into ordinary MurphyChain `Tool`s, direct execution, the error
 * path (server errors are redacted by the engine before they reach logs or
 * the model), and an `Agent` loop whose model requests an MCP tool.
 */
import { fileURLToPath } from "node:url";
import {
  AI,
  Agent,
  Tool,
  executeToolCalls,
  z,
  type AIResponse,
  type GenerateOptions,
  type Message,
  type ModelProvider,
  type ProviderCapabilities,
} from "murphychain";
import { connectStdio } from "@murphychain/mcp";

// ---------------------------------------------------------------------------
// 1. Connect: spawn the local MCP server over stdio.
// ---------------------------------------------------------------------------

const serverPath = fileURLToPath(new URL("./server.ts", import.meta.url));

const mcp = await connectStdio({
  command: process.execPath,
  args: ["--import", "tsx", serverPath],
});

console.log(`connected: ${mcp.serverInfo?.name} v${mcp.serverInfo?.version}`);

// ---------------------------------------------------------------------------
// 2. Discover: server tools become ordinary MurphyChain tools.
// ---------------------------------------------------------------------------

const tools = await mcp.tools({ namePrefix: "mcp_" });
console.log(
  `tools: ${tools.map((tool) => tool.name).join(", ")}  (from ${mcp.serverInfo?.name})`,
);

// ---------------------------------------------------------------------------
// 3. Execute directly — no model involved.
// ---------------------------------------------------------------------------

const add = tools.find((tool) => tool.name === "mcp_add")!;
const sum = await add.run({ a: 2, b: 3 });
console.log(`mcp_add(2, 3) = ${sum}`);
if (sum !== "5") throw new Error(`expected "5", got ${JSON.stringify(sum)}`);

const getTime = tools.find((tool) => tool.name === "mcp_get_time")!;
const now = await getTime.run({});
if (Number.isNaN(Date.parse(now))) {
  throw new Error(`expected an ISO-8601 timestamp, got ${JSON.stringify(now)}`);
}
console.log(`mcp_get_time() = ${now}`);

// ---------------------------------------------------------------------------
// 4. Errors: a server failure flows through engine dispatch and gets
//    redacted — the credential the server echoed never reaches the tool
//    message the model (or your logs) will see.
// ---------------------------------------------------------------------------

const [failure] = await executeToolCalls(tools, [
  { id: "c1", name: "mcp_fail", arguments: {} },
]);
if (failure === undefined || failure.role !== "tool") {
  throw new Error("expected a tool message for the failing call");
}
if (failure.content.includes("EXAMPLE12345678")) {
  throw new Error("secret leaked into the tool message");
}
console.log(`mcp_fail() → redacted tool message: ${failure.content}`);

// ---------------------------------------------------------------------------
// 5. Agents: an offline scripted model requests an MCP tool; the agent runs
//    it against the server and feeds the result back — same loop as local
//    tools, no code changes on the agent side.
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
    provider: "offline-mcp",
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
    provider: "offline-mcp",
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

const calls: Message[][] = [];
const provider: ModelProvider = {
  name: "offline-mcp",
  capabilities: CAPABILITIES,
  async generate(options: GenerateOptions): Promise<AIResponse> {
    const response =
      calls.length === 0
        ? requestTool("mcp_shout", { text: "hello world" }, "call_1")
        : reply(`The server shouted: ${lastToolResult(options.messages)}`);
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

/** The most recent tool message in a conversation — the MCP server's answer. */
function lastToolResult(messages: Message[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const message = messages[i]!;
    if (message.role === "tool") return message.content;
  }
  return "";
}

const agent = new Agent({
  name: "mcp-agent",
  system: "You answer using the tools provided.",
  ai: new AI({ provider, model: "offline-1" }),
  tools,
});

agent.on("tool", (event) =>
  console.log(`agent tool ${event.name}: ok=${event.ok} in ${event.durationMs}ms`),
);

const answer = await agent.run("Shout 'hello world' for me.");
console.log(`agent answer: ${answer.response.text}`);

// Verify the loop really used the MCP server: the second model call must
// carry the MCP tool's result ("HELLO WORLD") as a tool message.
const toolMessages = (calls[1] ?? []).filter((message) => message.role === "tool");
const shouted = toolMessages.at(-1)?.content ?? "";
if (shouted !== "HELLO WORLD") {
  throw new Error(`expected MCP tool result "HELLO WORLD", got ${JSON.stringify(shouted)}`);
}

// Local tools still mix in unchanged:
const double = Tool.define({
  name: "double",
  description: "Multiply a number by two.",
  parameters: z.object({ n: z.number() }),
  execute: async ({ n }) => String(n * 2),
});
console.log(`local tool double(21) = ${await double.run({ n: 21 })}`);

// ---------------------------------------------------------------------------
// 6. Close — kills the server process. Idempotent.
// ---------------------------------------------------------------------------

await mcp.close();
await mcp.close();
console.log(`closed: isClosed=${mcp.isClosed}`);
