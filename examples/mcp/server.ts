/**
 * A tiny MCP server used by the offline example — spawned by `index.ts` over
 * stdio. Implements three ordinary tools plus one that always fails, so the
 * example can show the error path.
 */
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";

const server = new Server(
  { name: "murphychain-example-server", version: "1.0.0" },
  { capabilities: { tools: {} } },
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
  tools: [
    {
      name: "add",
      description: "Add two integers.",
      inputSchema: {
        type: "object",
        properties: { a: { type: "number" }, b: { type: "number" } },
        required: ["a", "b"],
      },
    },
    {
      name: "shout",
      description: "Convert text to uppercase.",
      inputSchema: {
        type: "object",
        properties: { text: { type: "string" } },
        required: ["text"],
      },
    },
    {
      name: "get_time",
      description: "Get the current time as an ISO-8601 string.",
      inputSchema: { type: "object" },
    },
    {
      name: "fail",
      description: "Always fails — used to demonstrate the error path.",
      inputSchema: { type: "object" },
    },
  ],
}));

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const args = request.params.arguments;
  switch (request.params.name) {
    case "add": {
      const a = Number(args?.["a"]);
      const b = Number(args?.["b"]);
      return { content: [{ type: "text", text: String(a + b) }] };
    }
    case "shout":
      return {
        content: [{ type: "text", text: String(args?.["text"] ?? "").toUpperCase() }],
      };
    case "get_time":
      return { content: [{ type: "text", text: new Date().toISOString() }] };
    case "fail":
      // A realistic accident: a server echoing a credential in its error.
      return {
        content: [{ type: "text", text: "upstream rejected key sk-EXAMPLE12345678" }],
        isError: true,
      };
    default:
      throw new Error(`unknown tool: ${request.params.name}`);
  }
});

await server.connect(new StdioServerTransport());
