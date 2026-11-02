/**
 * MurphyChain MCP integration — connect to Model Context Protocol servers and
 * use their tools in agents.
 *
 * ```ts
 * import { connectStdio } from "@murphychain/mcp";
 *
 * const mcp = await connectStdio({ command: "npx", args: ["-y", "@some/mcp-server"] });
 * const tools = await mcp.tools({ namePrefix: "mcp_" });   // → Tool[]
 * await mcp.close();
 * ```
 *
 * The protocol is handled by the official `@modelcontextprotocol/sdk`;
 * MurphyChain adapts remote tools into ordinary {@link Tool} instances.
 */

export { McpConnection, connectMcp, connectStdio } from "./McpConnection";
export type { McpConnectOptions, McpStdioConfig, McpToolsOptions } from "./McpConnection";
