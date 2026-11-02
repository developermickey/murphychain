import { afterEach, describe, expect, it } from "vitest";
import { connectMcp, connectStdio, type McpConnection } from "@murphychain/mcp";
import { ValidationError, executeToolCalls } from "@murphychain/core";
import {
  badNameTool,
  echoTool,
  failTool,
  imageTool,
  makeFixtureServer,
  slowTool,
  startFixture,
  structuredTool,
  type FixtureToolSpec,
} from "./helpers";

// ---------------------------------------------------------------------------
// Harness: every test gets its own fixture server + connection, cleaned up
// after the test regardless of outcome.
// ---------------------------------------------------------------------------

const cleanups: Array<() => Promise<void>> = [];

afterEach(async () => {
  await Promise.all(
    cleanups.splice(0).map((fn) => fn().catch(() => undefined)),
  );
});

async function connectFixture(
  pages: FixtureToolSpec[][],
  options: Parameters<typeof connectMcp>[1] = {},
): Promise<McpConnection> {
  const fixture = await startFixture(makeFixtureServer({ pages }));
  cleanups.push(fixture.stop);
  const connection = await connectMcp(fixture.transport, options);
  cleanups.push(() => connection.close());
  return connection;
}

// ---------------------------------------------------------------------------
// Tool adaptation
// ---------------------------------------------------------------------------

describe("MCP tool adaptation", () => {
  it("adapts server tools into MurphyChain tools", async () => {
    const connection = await connectFixture([[echoTool, imageTool]]);
    const tools = await connection.tools();

    expect(tools).toHaveLength(2);
    const echo = tools.find((tool) => tool.name === "echo");
    expect(echo).toBeDefined();
    expect(echo!.description).toBe("Echoes the message back.");
    // The MCP input schema flows through as JSON Schema (documented path).
    expect(echo!.parameters).toEqual({
      type: "object",
      properties: { message: { type: "string" } },
      required: ["message"],
    });
    expect(echo!.normalized.jsonSchema).toEqual(echo!.parameters);
    expect(tools.map((tool) => tool.name)).toEqual(["echo", "draw"]);
  });

  it("applies a name prefix", async () => {
    const connection = await connectFixture([[echoTool]]);
    const tools = await connection.tools({ namePrefix: "mcp_" });
    expect(tools.map((tool) => tool.name)).toEqual(["mcp_echo"]);
    // The adapted name is what dispatch sees; the server still gets its own.
    const messages = await executeToolCalls(tools, [
      { id: "c1", name: "mcp_echo", arguments: { message: "hi" } },
    ]);
    expect(messages[0]!.content).toBe("hi");
  });

  it("reports server info from the handshake", async () => {
    const connection = await connectFixture([[echoTool]]);
    expect(connection.serverInfo).toEqual({ name: "fixture-server", version: "2.0.0" });
    expect(connection.isClosed).toBe(false);
  });

  it("follows pagination across tools/list pages", async () => {
    const connection = await connectFixture([[echoTool], [structuredTool], [imageTool]]);
    const tools = await connection.tools();
    expect(tools.map((tool) => tool.name)).toEqual(["echo", "structured", "draw"]);
  });

  it("returns an empty list for a server with no tools", async () => {
    const connection = await connectFixture([[]]);
    await expect(connection.tools()).resolves.toEqual([]);
  });

  it("refuses to loop forever on endless pagination", async () => {
    const pages: FixtureToolSpec[][] = Array.from({ length: 101 }, (_, i) => [
      { name: `tool_${i}`, handler: () => ({ content: [{ type: "text" as const, text: "x" }] }) },
    ]);
    const connection = await connectFixture(pages);
    await expect(connection.tools()).rejects.toThrow(/pages of tools/);
  });

  it("rejects a tool name the contract cannot express", async () => {
    const connection = await connectFixture([[badNameTool]]);
    await expect(connection.tools()).rejects.toThrow(/bad\.name.*cannot be adapted/);
  });

  it("validates namePrefix", async () => {
    const connection = await connectFixture([[echoTool]]);
    await expect(connection.tools({ namePrefix: "bad prefix!" })).rejects.toThrow(
      /namePrefix/,
    );
    await expect(connection.tools({ namePrefix: "x".repeat(61) })).rejects.toThrow(
      /must match/,
    );
  });
});

// ---------------------------------------------------------------------------
// Execution
// ---------------------------------------------------------------------------

describe("MCP tool execution", () => {
  it("executes a tool and returns its text", async () => {
    const connection = await connectFixture([[echoTool]]);
    const echo = (await connection.tools()).find((tool) => tool.name === "echo")!;
    await expect(echo.run({ message: "hello" })).resolves.toBe("hello");
  });

  it("describes binary content instead of inlining base64", async () => {
    const connection = await connectFixture([[imageTool]]);
    const draw = (await connection.tools()).find((tool) => tool.name === "draw")!;
    const result = await draw.run({});
    expect(result).toBe("[image: image/png, 4 base64 chars]");
    expect(result).not.toContain("AAAA");
  });

  it("falls back to structured content when there is no text", async () => {
    const connection = await connectFixture([[structuredTool]]);
    const structured = (await connection.tools()).find((tool) => tool.name === "structured")!;
    await expect(structured.run({})).resolves.toBe('{"answer":42}');
  });

  it("throws the server's error text when isError is set", async () => {
    const connection = await connectFixture([[failTool]]);
    const fail = (await connection.tools()).find((tool) => tool.name === "fail")!;
    await expect(fail.run({})).rejects.toThrow(/boom/);
  });

  it("redacts secrets when failures flow through engine dispatch", async () => {
    const connection = await connectFixture([[failTool]]);
    const tools = await connection.tools();
    const [message] = await executeToolCalls(tools, [
      { id: "c1", name: "fail", arguments: {} },
    ]);
    expect(message!.role).toBe("tool");
    expect(message!.content).toContain('"error"');
    expect(message!.content).toContain("boom");
    expect(message!.content).not.toContain("SECRET12345678");
  });

  it("serializes successful results for the model", async () => {
    const connection = await connectFixture([[echoTool]]);
    const tools = await connection.tools();
    const [message] = await executeToolCalls(tools, [
      { id: "c1", name: "echo", arguments: { message: "through dispatch" } },
    ]);
    expect(message!.content).toBe("through dispatch");
  });
});

// ---------------------------------------------------------------------------
// Timeouts and cancellation
// ---------------------------------------------------------------------------

describe("MCP timeouts and cancellation", () => {
  it("enforces the per-request timeout", async () => {
    const connection = await connectFixture([[slowTool]], { requestTimeoutMs: 50 });
    const slow = (await connection.tools()).find((tool) => tool.name === "slow")!;
    await expect(slow.run({})).rejects.toThrow(/timeout|timed out/i);
  });

  it("forwards the abort signal to the in-flight request", async () => {
    const connection = await connectFixture([[slowTool]]);
    const slow = (await connection.tools()).find((tool) => tool.name === "slow")!;
    const controller = new AbortController();
    setTimeout(() => controller.abort(), 50);
    await expect(slow.run({}, { signal: controller.signal })).rejects.toThrow(/abort/i);
  });
});

// ---------------------------------------------------------------------------
// Lifecycle and validation
// ---------------------------------------------------------------------------

describe("MCP lifecycle", () => {
  it("closes idempotently and rejects use afterwards", async () => {
    const connection = await connectFixture([[echoTool]]);
    await connection.close();
    expect(connection.isClosed).toBe(true);
    await connection.close(); // no throw
    await expect(connection.tools()).rejects.toThrow(/closed/);
  });

  it("rejects invalid connectMcp options before connecting", async () => {
    await expect(connectMcp({} as never)).rejects.toThrow(ValidationError);
    await expect(connectMcp({} as never)).rejects.toThrow(/Transport/);
    await expect(connectMcp({} as never, { clientName: "" })).rejects.toThrow(/clientName/);
    await expect(connectMcp({} as never, { clientVersion: "" })).rejects.toThrow(
      /clientVersion/,
    );
    for (const requestTimeoutMs of [0, -1, Number.NaN, Number.POSITIVE_INFINITY]) {
      await expect(connectMcp({} as never, { requestTimeoutMs })).rejects.toThrow(
        /requestTimeoutMs/,
      );
    }
  });

  it("rejects invalid connectStdio configs before spawning", async () => {
    const badConfigs: Array<unknown> = [
      null,
      {},
      { command: "   " },
      { command: "node", args: "not-an-array" },
      { command: "node", args: [1, 2] },
      { command: "node", env: "not-an-object" },
      { command: "node", env: { KEY: 1 } },
      { command: "node", cwd: "" },
      { command: "node", stderr: "pipe" },
      { command: "node", stderr: "loud" },
      { command: "node", clientName: "" },
      { command: "node", clientVersion: "" },
      { command: "node", requestTimeoutMs: 0 },
      { command: "node", requestTimeoutMs: -5 },
    ];
    for (const config of badConfigs) {
      await expect(connectStdio(config as never)).rejects.toThrow(ValidationError);
    }
  });
});
