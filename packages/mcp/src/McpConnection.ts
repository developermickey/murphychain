import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { StdioClientTransport, getDefaultEnvironment } from "@modelcontextprotocol/sdk/client/stdio.js";
import type { Transport } from "@modelcontextprotocol/sdk/shared/transport.js";
import type { Tool as McpServerTool } from "@modelcontextprotocol/sdk/types.js";
import { Tool, ValidationError } from "@murphychain/core";

/**
 * MCP (Model Context Protocol) integration: connect to an MCP server and use
 * its tools in MurphyChain agents.
 *
 * The protocol itself is handled by the official `@modelcontextprotocol/sdk`
 * — MurphyChain owns adaptation only: turning remote tools into ordinary
 * {@link Tool} instances whose results flow through the engine's normal
 * dispatch, redaction, and event paths.
 *
 * ```ts
 * const mcp = await connectStdio({ command: "npx", args: ["-y", "@some/mcp-server"] });
 * const tools = await mcp.tools({ namePrefix: "mcp_" });   // → Tool[]
 * const agent = new Agent({ ai, tools: [...localTools, ...tools] });
 * await mcp.close();
 * ```
 *
 * Only launch MCP servers you trust: their tool definitions and results reach
 * your model. Server-provided content is treated as unvalidated input —
 * shapes are checked by the SDK, tool names by this adapter, errors are
 * redacted by the engine, and binary payloads are never inlined into the
 * model context.
 */

/** Identification sent to the server during the MCP handshake. */
const DEFAULT_CLIENT_NAME = "murphychain";
const DEFAULT_CLIENT_VERSION = "1.0.0";

/**
 * Safety bound on `tools/list` pagination: a broken or hostile server that
 * never stops handing out cursors cannot loop this client forever.
 */
const MAX_TOOL_PAGES = 100;

const TOOL_NAME_PATTERN = /^[A-Za-z0-9_-]{1,64}$/;

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/** Options for {@link connectMcp}. */
export interface McpConnectOptions {
  /** Client name reported to the server. Default: `"murphychain"`. */
  clientName?: string | undefined;
  /**
   * Client version reported to the server. Independent of the npm package
   * version (informational only, servers may log it). Default: `"1.0.0"`.
   */
  clientVersion?: string | undefined;
  /** Timeout for each MCP request in milliseconds. Default: 60 000. */
  requestTimeoutMs?: number | undefined;
}

/** Configuration for {@link connectStdio} — spawns a server subprocess. */
export interface McpStdioConfig {
  /** Executable that starts the server, e.g. `"npx"` or `"node"`. */
  command: string;
  /** Command-line arguments, e.g. `["-y", "@modelcontextprotocol/server-everything"]`. */
  args?: string[] | undefined;
  /**
   * Environment variables for the server process. Merged over the SDK's
   * sanitized default environment (PATH, HOME, …) — never the full parent
   * environment — so spawned servers keep a working environment while
   * receiving only what you add.
   */
  env?: Record<string, string> | undefined;
  /** Working directory for the server process. Default: inherited. */
  cwd?: string | undefined;
  /** How the server's stderr is handled. Default: `"inherit"`. */
  stderr?: "inherit" | "ignore" | undefined;
  /** Client name reported to the server. Default: `"murphychain"`. */
  clientName?: string | undefined;
  /** Client version reported to the server. Default: `"1.0.0"`. */
  clientVersion?: string | undefined;
  /** Timeout for each MCP request in milliseconds. Default: 60 000. */
  requestTimeoutMs?: number | undefined;
}

/** Options for {@link McpConnection.tools}. */
export interface McpToolsOptions {
  /**
   * Prefix added to every adapted tool name (e.g. `"mcp_"`), useful to avoid
   * collisions with locally defined tools. Must match `[A-Za-z0-9_-]*`.
   */
  namePrefix?: string | undefined;
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

function assertConnectOptions(options: McpConnectOptions): void {
  if (typeof options !== "object" || options === null) {
    throw new ValidationError("MCP connect options must be an object.");
  }
  if (options.clientName !== undefined && (typeof options.clientName !== "string" || options.clientName.length === 0)) {
    throw new ValidationError('"clientName" must be a non-empty string.', { field: "clientName" });
  }
  if (
    options.clientVersion !== undefined &&
    (typeof options.clientVersion !== "string" || options.clientVersion.length === 0)
  ) {
    throw new ValidationError('"clientVersion" must be a non-empty string.', { field: "clientVersion" });
  }
  if (
    options.requestTimeoutMs !== undefined &&
    (typeof options.requestTimeoutMs !== "number" ||
      !Number.isFinite(options.requestTimeoutMs) ||
      options.requestTimeoutMs <= 0)
  ) {
    throw new ValidationError('"requestTimeoutMs" must be a finite number > 0.', {
      field: "requestTimeoutMs",
    });
  }
}

function assertStdioConfig(config: McpStdioConfig): void {
  if (typeof config !== "object" || config === null) {
    throw new ValidationError("connectStdio() requires a configuration object.");
  }
  if (typeof config.command !== "string" || config.command.trim().length === 0) {
    throw new ValidationError('"command" must be a non-empty string.', { field: "command" });
  }
  if (config.args !== undefined) {
    if (!Array.isArray(config.args) || config.args.some((arg) => typeof arg !== "string")) {
      throw new ValidationError('"args" must be an array of strings.', { field: "args" });
    }
  }
  if (config.env !== undefined) {
    if (
      typeof config.env !== "object" ||
      config.env === null ||
      Array.isArray(config.env) ||
      Object.values(config.env).some((value) => typeof value !== "string")
    ) {
      throw new ValidationError('"env" must be an object of string values.', { field: "env" });
    }
  }
  if (config.cwd !== undefined && (typeof config.cwd !== "string" || config.cwd.length === 0)) {
    throw new ValidationError('"cwd" must be a non-empty string.', { field: "cwd" });
  }
  if (config.stderr !== undefined && config.stderr !== "inherit" && config.stderr !== "ignore") {
    throw new ValidationError('"stderr" must be "inherit" or "ignore".', { field: "stderr" });
  }
  assertConnectOptions(config);
}

// ---------------------------------------------------------------------------
// Content extraction
// ---------------------------------------------------------------------------

/** Structural subset of an MCP CallToolResult the extractor reads. */
interface ToolResultContent {
  content: ReadonlyArray<{ type: string } & Record<string, unknown>>;
  structuredContent?: Record<string, unknown> | undefined;
}

/**
 * Compact, bounded description of a non-text content block. Binary payloads
 * (base64 images/audio) are described, never inlined — they would bloat the
 * conversation and carry no meaning to the model.
 */
function describeBlock(block: { type: string } & Record<string, unknown>): string {
  switch (block.type) {
    case "text":
      return typeof block.text === "string" ? block.text : "";
    case "image":
    case "audio": {
      const mimeType = typeof block.mimeType === "string" ? block.mimeType : "unknown";
      const size = typeof block.data === "string" ? block.data.length : 0;
      return `[${block.type}: ${mimeType}, ${size} base64 chars]`;
    }
    case "resource": {
      const resource = block["resource"];
      if (typeof resource === "object" && resource !== null) {
        const r = resource as Record<string, unknown>;
        const uri = typeof r["uri"] === "string" ? r["uri"] : "unknown uri";
        const mimeType = typeof r["mimeType"] === "string" ? r["mimeType"] : "unknown type";
        const size =
          typeof r["text"] === "string"
            ? r["text"].length
            : typeof r["blob"] === "string"
              ? r["blob"].length
              : 0;
        return `[resource: ${uri}, ${mimeType}, ${size} chars]`;
      }
      return "[resource]";
    }
    default:
      return `[${block.type}]`;
  }
}

/**
 * Extracts what the model should see from a tool result: joined text blocks
 * first, then structured content as JSON, then bounded descriptors of any
 * remaining blocks.
 */
function extractToolText(result: ToolResultContent): string {
  const texts: string[] = [];
  for (const block of result.content) {
    if (block.type === "text" && typeof block.text === "string" && block.text.length > 0) {
      texts.push(block.text);
    }
  }
  const text = texts.join("\n");
  if (text.trim().length > 0) return text;
  const structured = result.structuredContent;
  if (structured !== undefined && structured !== null) {
    try {
      return JSON.stringify(structured) ?? "";
    } catch {
      // Unserializable structured content falls through to block descriptors.
    }
  }
  return result.content.map(describeBlock).join("\n");
}

// ---------------------------------------------------------------------------
// Connection
// ---------------------------------------------------------------------------

/**
 * A live MCP connection. Created by {@link connectMcp} or
 * {@link connectStdio}, or by wrapping an already-connected SDK
 * {@link Client} (e.g. a transport with custom auth).
 */
export class McpConnection {
  /** The underlying SDK client — escape hatch for resources, prompts, … */
  readonly client: Client;

  private readonly requestTimeoutMs: number | undefined;
  private closed = false;

  constructor(client: Client, options: McpConnectOptions = {}) {
    assertConnectOptions(options);
    this.client = client;
    this.requestTimeoutMs = options.requestTimeoutMs;
  }

  /** Information the server reported during the handshake, if connected. */
  get serverInfo(): { name: string; version?: string } | undefined {
    const info = this.client.getServerVersion();
    if (info === undefined) return undefined;
    return info.version !== undefined
      ? { name: info.name, version: info.version }
      : { name: info.name };
  }

  /** True once {@link close} has been called. */
  get isClosed(): boolean {
    return this.closed;
  }

  /**
   * Lists the server's tools and adapts each one into a MurphyChain
   * {@link Tool}. Every call re-queries the server (`tools/list`, following
   * pagination), so the result reflects server-side changes.
   *
   * The MCP input schema is passed to the engine as JSON Schema (the tool's
   * documented schema-passthrough path): the engine enforces the structural
   * object invariant, and the MCP server validates arguments against its own
   * schema when the tool is called.
   */
  async tools(options: McpToolsOptions = {}): Promise<Tool<Record<string, unknown>, string>[]> {
    if (this.closed) {
      throw new ValidationError("MCP connection is closed.");
    }
    const prefix = options.namePrefix ?? "";
    if (typeof prefix !== "string" || !/^[A-Za-z0-9_-]*$/.test(prefix)) {
      throw new ValidationError('"namePrefix" must match [A-Za-z0-9_-]*.', { field: "namePrefix" });
    }
    const listed = await this.listAllTools();
    return listed.map((mcpTool) => this.adaptTool(mcpTool, prefix));
  }

  /**
   * Closes the connection (and the server process for stdio transports).
   * Idempotent: subsequent calls are no-ops.
   */
  async close(): Promise<void> {
    if (this.closed) return;
    this.closed = true;
    await this.client.close();
  }

  // -------------------------------------------------------------------------
  // Internals
  // -------------------------------------------------------------------------

  /** Per-request options: configured timeout and optional cancellation. */
  private requestOptions(signal?: AbortSignal): { timeout?: number; signal?: AbortSignal } {
    const options: { timeout?: number; signal?: AbortSignal } = {};
    if (this.requestTimeoutMs !== undefined) options.timeout = this.requestTimeoutMs;
    if (signal !== undefined) options.signal = signal;
    return options;
  }

  /** Follows `tools/list` pagination up to {@link MAX_TOOL_PAGES} pages. */
  private async listAllTools(): Promise<McpServerTool[]> {
    const tools: McpServerTool[] = [];
    let cursor: string | undefined;
    for (let page = 0; page < MAX_TOOL_PAGES; page++) {
      const result = await this.client.listTools(
        cursor !== undefined ? { cursor } : {},
        this.requestOptions(),
      );
      tools.push(...result.tools);
      cursor = result.nextCursor;
      if (cursor === undefined) return tools;
    }
    throw new ValidationError(
      `MCP server returned more than ${MAX_TOOL_PAGES} pages of tools; refusing to continue.`,
    );
  }

  /** Adapts one MCP tool into a MurphyChain Tool bound to this connection. */
  private adaptTool(
    mcpTool: McpServerTool,
    prefix: string,
  ): Tool<Record<string, unknown>, string> {
    const name = prefix + mcpTool.name;
    if (!TOOL_NAME_PATTERN.test(name)) {
      throw new ValidationError(
        `MCP tool "${mcpTool.name}" cannot be adapted: the name "${name}" must match [A-Za-z0-9_-]{1,64}. Rename it on the server or adjust "namePrefix".`,
        { tool: mcpTool.name },
      );
    }
    try {
      return Tool.define({
        name,
        description: mcpTool.description ?? "",
        parameters: mcpTool.inputSchema,
        execute: async (args: Record<string, unknown>, executeOptions) =>
          this.callTool(mcpTool.name, args, executeOptions.signal),
      });
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new ValidationError(`MCP tool "${mcpTool.name}" cannot be adapted: ${message}`, {
        tool: mcpTool.name,
      });
    }
  }

  /** One `tools/call` round trip; failures throw with the server's text. */
  private async callTool(
    name: string,
    args: Record<string, unknown>,
    signal?: AbortSignal,
  ): Promise<string> {
    const result = await this.client.callTool(
      { name, arguments: args },
      undefined,
      this.requestOptions(signal),
    );
    const text = extractToolText(result as unknown as ToolResultContent);
    if (result.isError === true) {
      throw new Error(text.length > 0 ? text : `MCP tool "${name}" failed.`);
    }
    return text;
  }
}

// ---------------------------------------------------------------------------
// Connect helpers
// ---------------------------------------------------------------------------

/**
 * Connects an already-constructed MCP transport (stdio, in-memory, or any
 * other transport the SDK offers). Prefer {@link connectStdio} for the common
 * subprocess case; use this to bring your own transport (e.g. one with
 * authentication headers).
 */
export async function connectMcp(
  transport: Transport,
  options: McpConnectOptions = {},
): Promise<McpConnection> {
  assertConnectOptions(options);
  if (typeof transport !== "object" || transport === null || typeof transport.start !== "function") {
    throw new ValidationError('"transport" must be an MCP Transport.', { field: "transport" });
  }
  const client = new Client(
    {
      name: options.clientName ?? DEFAULT_CLIENT_NAME,
      version: options.clientVersion ?? DEFAULT_CLIENT_VERSION,
    },
    { capabilities: {} },
  );
  await client.connect(transport);
  return new McpConnection(client, options);
}

/**
 * Spawns an MCP server over stdio and connects to it.
 *
 * ```ts
 * const mcp = await connectStdio({ command: "npx", args: ["-y", "@some/mcp-server"] });
 * ```
 */
export async function connectStdio(config: McpStdioConfig): Promise<McpConnection> {
  assertStdioConfig(config);
  const env = { ...getDefaultEnvironment(), ...config.env };
  const transport = new StdioClientTransport({
    command: config.command,
    ...(config.args !== undefined ? { args: config.args } : {}),
    env,
    ...(config.cwd !== undefined ? { cwd: config.cwd } : {}),
    ...(config.stderr !== undefined ? { stderr: config.stderr } : {}),
  });
  const options: McpConnectOptions = {};
  if (config.clientName !== undefined) options.clientName = config.clientName;
  if (config.clientVersion !== undefined) options.clientVersion = config.clientVersion;
  if (config.requestTimeoutMs !== undefined) options.requestTimeoutMs = config.requestTimeoutMs;
  return connectMcp(transport, options);
}
