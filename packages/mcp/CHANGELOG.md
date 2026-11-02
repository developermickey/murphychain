# @murphychain/mcp

## 0.5.0

### Minor Changes

- 133090f: MCP (Model Context Protocol) integration: connect to MCP servers and use
  their tools in MurphyChain agents.
  
  - `connectStdio(config)` spawns a server subprocess over stdio;
    `connectMcp(transport, options)` accepts any SDK transport (in-memory,
    SSE, custom auth).
  - `McpConnection.tools({ namePrefix })` re-lists tools on every call and
    adapts each one into an ordinary `Tool` — MCP `inputSchema` passes through
    as JSON Schema, so agents and chains consume them unchanged. `close()` is
    idempotent; `serverInfo` and a public `client` escape hatch are available.
  - Bounded and safe by construction: `tools/list` pagination capped at 100
    pages, per-request timeout (60 s default) with forwarded `AbortSignal`,
    tool names validated against `[A-Za-z0-9_-]{1,64}`, sanitized default
    environment for spawned servers (never the full parent env), binary result
    content described instead of inlined, and failures flowing through
    `redactSecrets()` like any tool error.
  
  Powered by the official `@modelcontextprotocol/sdk` (MIT). No changes to
  `@murphychain/core` or the `murphychain` facade.
