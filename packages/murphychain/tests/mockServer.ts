import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import type { ServerResponse } from "node:http";

/** A request captured by the mock server. */
export interface RecordedRequest {
  method: string;
  url: string;
  headers: Record<string, string | string[] | undefined>;
  /** Parsed JSON body (or the raw string when not JSON). */
  body: unknown;
}

/** What the mock server should answer for one request. */
export interface MockRoute {
  status?: number;
  extraHeaders?: Record<string, string>;
  /** JSON body (serialized), ignored when `raw` or `sse` is set. */
  body?: unknown;
  /** Raw response body (e.g. a JSON error string). */
  raw?: string;
  /** Server-sent-event `data:` payloads; `[DONE]` is appended automatically for OpenAI. */
  sse?: string[];
  /** Keep the connection open without responding (for abort/timeout tests). */
  hang?: boolean;
  /** Destroy the socket immediately (transport-level failure). */
  destroySocket?: boolean;
}

export interface MockServer {
  baseUrl: string;
  requests: RecordedRequest[];
  close(): Promise<void>;
}

/**
 * Loopback HTTP server used to exercise the real provider SDKs without
 * network access: adapters point their `baseUrl`/`baseURL` here, so SDK
 * request building, SSE parsing, and error handling all run for real.
 */
export async function startMockServer(
  handler: (request: RecordedRequest, response: ServerResponse) => MockRoute | Promise<MockRoute>,
): Promise<MockServer> {
  const requests: RecordedRequest[] = [];
  const sockets = new Set<import("node:net").Socket>();
  const server: Server = createServer((req, res) => {
    const chunks: Buffer[] = [];
    req.on("data", (chunk: Buffer) => chunks.push(chunk));
    req.on("end", () => {
      const raw = Buffer.concat(chunks).toString("utf8");
      let body: unknown = raw;
      try {
        body = raw.length > 0 ? JSON.parse(raw) : undefined;
      } catch {
        // keep the raw string
      }
      const recorded: RecordedRequest = {
        method: req.method ?? "",
        url: req.url ?? "",
        headers: { ...req.headers },
        body,
      };
      requests.push(recorded);
      void Promise.resolve(handler(recorded, res)).then((route) => {
        if (route.hang === true) return; // leave the socket open on purpose
        if (route.destroySocket === true) {
          req.socket.destroy();
          return;
        }
        res.writeHead(route.status ?? 200, {
          "content-type": route.sse !== undefined ? "text/event-stream" : "application/json",
          ...route.extraHeaders,
        });
        if (route.sse !== undefined) {
          for (const data of route.sse) res.write(`data: ${data}\n\n`);
          res.end();
          return;
        }
        res.end(route.raw ?? JSON.stringify(route.body ?? {}));
      });
    });
  });
  server.on("connection", (socket) => {
    sockets.add(socket);
    socket.on("close", () => sockets.delete(socket));
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const { port } = server.address() as AddressInfo;
  return {
    baseUrl: `http://127.0.0.1:${port}`,
    requests,
    async close() {
      for (const socket of sockets) socket.destroy();
      await new Promise<void>((resolve) => server.close(() => resolve()));
    },
  };
}
