import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
  type CallToolResult,
  type Tool as McpToolDef,
} from "@modelcontextprotocol/sdk/types.js";
import type { Transport } from "@modelcontextprotocol/sdk/shared/transport.js";

// ---------------------------------------------------------------------------
// Fixture server: a raw SDK Server driven by explicit specs, so tests control
// every detail of the protocol (pagination, errors, slow responses, content).
// ---------------------------------------------------------------------------

export interface FixtureToolSpec {
  name: string;
  description?: string;
  /** JSON Schema for arguments. Default: `{ type: "object" }`. */
  inputSchema?: McpToolDef["inputSchema"];
  handler: (args: Record<string, unknown> | undefined) => CallToolResult | Promise<CallToolResult>;
}

export interface FixtureServerOptions {
  /** One array per `tools/list` page (page 2+ served via cursors). */
  pages: FixtureToolSpec[][];
}

export function makeFixtureServer(options: FixtureServerOptions): Server {
  const byName = new Map<string, FixtureToolSpec>();
  for (const page of options.pages) {
    for (const spec of page) byName.set(spec.name, spec);
  }

  const server = new Server(
    { name: "fixture-server", version: "2.0.0" },
    { capabilities: { tools: {} } },
  );

  server.setRequestHandler(ListToolsRequestSchema, async (request) => {
    const cursor = request.params?.cursor;
    const pageIndex = cursor === undefined ? 0 : Number(cursor);
    const page = options.pages[pageIndex] ?? [];
    const tools: McpToolDef[] = page.map((spec) => ({
      name: spec.name,
      ...(spec.description !== undefined ? { description: spec.description } : {}),
      inputSchema: spec.inputSchema ?? { type: "object" },
    }));
    const result: { tools: McpToolDef[]; nextCursor?: string } = { tools };
    if (pageIndex + 1 < options.pages.length) result.nextCursor = String(pageIndex + 1);
    return result;
  });

  server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const spec = byName.get(request.params.name);
    if (spec === undefined) throw new Error(`unknown tool: ${request.params.name}`);
    return spec.handler(request.params.arguments);
  });

  return server;
}

/** Starts a fixture server on one end of a linked in-memory transport pair. */
export async function startFixture(
  server: Server,
): Promise<{ transport: Transport; stop: () => Promise<void> }> {
  const [clientTransport, serverTransport] = InMemoryTransport.createLinkedPair();
  await server.connect(serverTransport);
  return {
    transport: clientTransport,
    stop: async () => {
      await serverTransport.close().catch(() => undefined);
      await clientTransport.close().catch(() => undefined);
    },
  };
}

// ---------------------------------------------------------------------------
// Shared tool specs
// ---------------------------------------------------------------------------

/** Echoes `args.message` as text. */
export const echoTool: FixtureToolSpec = {
  name: "echo",
  description: "Echoes the message back.",
  inputSchema: {
    type: "object",
    properties: { message: { type: "string" } },
    required: ["message"],
  },
  handler: (args) => ({
    content: [{ type: "text", text: String(args?.["message"] ?? "") }],
  }),
};

/** Returns an image block — binary content that must never be inlined. */
export const imageTool: FixtureToolSpec = {
  name: "draw",
  description: "Returns a tiny image.",
  handler: () => ({
    content: [{ type: "image", data: "AAAA", mimeType: "image/png" }],
  }),
};

/** Fails with a secret-looking string — must be redacted on the way back. */
export const failTool: FixtureToolSpec = {
  name: "fail",
  description: "Always fails.",
  handler: () => ({
    content: [{ type: "text", text: "boom sk-SECRET12345678" }],
    isError: true,
  }),
};

/** Sleeps 500 ms before answering — for timeout and cancellation tests. */
export const slowTool: FixtureToolSpec = {
  name: "slow",
  description: "Answers late.",
  handler: async () => {
    await new Promise((resolve) => setTimeout(resolve, 500));
    return { content: [{ type: "text", text: "late" }] };
  },
};

/** Structured content without text blocks — JSON fallback path. */
export const structuredTool: FixtureToolSpec = {
  name: "structured",
  description: "Returns structured content only.",
  handler: () => ({
    content: [],
    structuredContent: { answer: 42 },
  }),
};

/** A name the MurphyChain tool contract rejects. */
export const badNameTool: FixtureToolSpec = {
  name: "bad.name",
  description: "Invalid tool name.",
  handler: () => ({ content: [{ type: "text", text: "unreachable" }] }),
};
