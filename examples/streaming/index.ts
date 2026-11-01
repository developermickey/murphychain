/**
 * Streaming — consume output as it arrives.
 *
 * Requires an API key:
 *   OPENAI_API_KEY=sk-...  pnpm example:streaming
 *   GEMINI_API_KEY=AIza... pnpm example:streaming
 *
 * Demonstrates: `stream()` returning a single-use async iterable, per-chunk
 * indexing, the terminal chunk (finish reason + usage), early `break`
 * cancellation, and time-to-first-chunk timeout semantics.
 */
import { AI, MurphyChainError } from "murphychain";

function resolveProvider(): { provider: "openai" | "gemini"; apiKey: string } | undefined {
  const openaiKey = process.env.OPENAI_API_KEY;
  if (openaiKey !== undefined && openaiKey.length > 0) {
    return { provider: "openai", apiKey: openaiKey };
  }
  const geminiKey = process.env.GEMINI_API_KEY ?? process.env.GOOGLE_API_KEY;
  if (geminiKey !== undefined && geminiKey.length > 0) {
    return { provider: "gemini", apiKey: geminiKey };
  }
  return undefined;
}

const selected = resolveProvider();

if (selected === undefined) {
  console.log("No API key found — set one to run this example:");
  console.log("  OPENAI_API_KEY=sk-...   pnpm example:streaming");
  console.log("  GEMINI_API_KEY=AIza...  pnpm example:streaming");
  process.exit(0);
}

const ai = new AI({
  provider: selected.provider,
  model: selected.provider === "openai" ? "gpt-4.1-mini" : "gemini-2.5-flash",
  apiKey: selected.apiKey,
  temperature: 0.6,
  // For streams this is a time-to-FIRST-chunk deadline (not a total deadline):
  // the clock stops as soon as the first delta arrives.
  timeoutMs: 15_000,
});

// Content-free chunk events: index + length only, safe to log.
ai.on("chunk", (event) => {
  process.stdout.write(`[chunk ${event.index} len=${event.textLength}] `);
});

try {
  const stream = await ai.stream("Write a short four-line poem about distributed systems.");

  console.log("--- streamed output ---\n");

  let totalChars = 0;
  for await (const chunk of stream) {
    // Chunks are single-use: the same stream cannot be iterated twice.
    process.stdout.write(chunk.text);
    totalChars += chunk.text.length;

    // The terminal chunk carries `finishReason` / `usage` and empty `text`.
    if (chunk.finishReason !== undefined) {
      process.stdout.write("\n");
      console.log(
        `--- terminal: finish=${chunk.finishReason} tokens=${chunk.usage?.totalTokens ?? "n/a"} ---`,
      );
    }
  }

  console.log(`chars streamed: ${totalChars}`);
  console.log(`stats: ${JSON.stringify(ai.stats())}`);

  // -----------------------------------------------------------------------
  // Early break cancels the underlying request — the engine aborts the
  // provider call instead of draining the rest of the stream.
  // -----------------------------------------------------------------------
  const partial = await ai.stream("Count slowly from one to one hundred.");
  let seen = 0;
  for await (const chunk of partial) {
    process.stdout.write(chunk.text);
    seen += 1;
    if (seen >= 3) {
      process.stdout.write("\n");
      console.log("--- broke out early; upstream request was cancelled ---");
      break;
    }
  }

  // Streams are single-use by design: a second iteration fails loudly
  // instead of silently replaying (or double-charging for) a request.
  try {
    for await (const _chunk of partial) {
      // never reached
    }
  } catch (error) {
    console.log(`second iteration rejected: ${(error as Error).message}`);
  }
} catch (error) {
  if (error instanceof MurphyChainError) {
    console.error(`\n[${error.code}] ${error.message}`);
    if (error.code === "PROVIDER_TIMEOUT") {
      console.error("hint: no chunk arrived within timeoutMs (time-to-first-chunk).");
    }
  } else {
    throw error;
  }
}
