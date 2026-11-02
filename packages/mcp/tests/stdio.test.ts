import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it } from "vitest";
import { connectStdio, type McpConnection } from "@murphychain/mcp";

// ---------------------------------------------------------------------------
// Real-subprocess coverage for the shipped transport: the fixture server is
// spawned with `node --import tsx` and spoken to over stdio. Fully offline.
// ---------------------------------------------------------------------------

const fixturePath = fileURLToPath(new URL("./fixtures/stdio-server.ts", import.meta.url));

const cleanups: Array<() => Promise<void>> = [];

afterEach(async () => {
  await Promise.all(
    cleanups.splice(0).map((fn) => fn().catch(() => undefined)),
  );
});

async function connectFixtureServer(
  env?: Record<string, string>,
): Promise<McpConnection> {
  const connection = await connectStdio({
    command: process.execPath,
    args: ["--import", "tsx", fixturePath],
    stderr: "ignore",
    ...(env !== undefined ? { env } : {}),
  });
  cleanups.push(() => connection.close());
  return connection;
}

describe("MCP stdio transport", () => {
  it("spawns a server process, lists tools, and executes them", async () => {
    const connection = await connectFixtureServer();

    expect(connection.serverInfo).toEqual({ name: "stdio-fixture", version: "1.0.0" });

    const tools = await connection.tools();
    expect(tools.map((tool) => tool.name)).toEqual(["echo", "env_probe", "fail"]);

    const echo = tools.find((tool) => tool.name === "echo")!;
    await expect(echo.run({ message: "over stdio" })).resolves.toBe("over stdio");
  });

  it("merges explicit env over the sanitized default environment", async () => {
    const connection = await connectFixtureServer({ MCP_TEST_FLAG: "merged" });
    const probe = (await connection.tools()).find((tool) => tool.name === "env_probe")!;
    const report = JSON.parse(await probe.run({})) as { flag: string; hasPath: boolean };
    // The override arrived…
    expect(report.flag).toBe("merged");
    // …and the server still inherited a working default environment (PATH),
    // which a plain env replacement would have dropped.
    expect(report.hasPath).toBe(true);
  });

  it("surfaces spawn failures as connection errors", async () => {
    await expect(
      connectStdio({ command: "/definitely/not/a/real/executable" }),
    ).rejects.toThrow();
  });
});
