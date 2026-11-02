/**
 * A minimal MCP server spoken to over stdio. Spawned by the stdio tests
 * (`node --import tsx fixtures/stdio-server.ts`) — runs fully offline.
 */
import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema } from "@modelcontextprotocol/sdk/types.js";

const server = new Server(
  { name: "stdio-fixture", version: "1.0.0" },
  { capabilities: { tools: {} } },
);

server.setRequestHandler(ListToolsRequestSchema, async () => ({
    tools: [
      {
        name: "echo",
        description: "Echoes the message back.",
        inputSchema: {
          type: "object" as const,
          properties: { message: { type: "string" } },
          required: ["message"],
        },
      },
      {
        name: "env_probe",
        description: "Reports the process environment.",
        inputSchema: { type: "object" as const },
      },
      {
        name: "fail",
        description: "Always fails.",
        inputSchema: { type: "object" as const },
      },
    ],
  }),
);

server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const name = request.params.name;
  const args = request.params.arguments;
  switch (name) {
    case "echo":
      return {
        content: [{ type: "text" as const, text: String(args?.["message"] ?? "") }],
      };
    case "env_probe":
      return {
        content: [
          {
            type: "text" as const,
            text: JSON.stringify({
              flag: process.env["MCP_TEST_FLAG"] ?? "unset",
              hasPath: typeof process.env["PATH"] === "string",
            }),
          },
        ],
      };
    case "fail":
      return {
        content: [{ type: "text" as const, text: "server exploded sk-SECRET12345678" }],
        isError: true,
      };
    default:
      throw new Error(`unknown tool: ${name}`);
  }
});

await server.connect(new StdioServerTransport());
