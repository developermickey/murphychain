/**
 * Basic chat — the core MurphyChain DX in one file.
 *
 * Requires an API key:
 *   OPENAI_API_KEY=sk-...  pnpm example:basic-chat
 *   GEMINI_API_KEY=AIza... pnpm example:basic-chat
 *
 * Demonstrates: constructor config, generate(), prompt templates, engine
 * events (content-free by design), and usage/stat accounting.
 */
import { AI, MurphyChainError, Prompt } from "murphychain";

// ---------------------------------------------------------------------------
// Pick a provider from whichever key is available (no key = friendly exit).
// ---------------------------------------------------------------------------

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
  console.log("  OPENAI_API_KEY=sk-...   pnpm example:basic-chat");
  console.log("  GEMINI_API_KEY=AIza...  pnpm example:basic-chat");
  process.exit(0);
}

// ---------------------------------------------------------------------------
// One AI instance per provider+model; the engine owns retries, timeouts,
// validation, and events.
// ---------------------------------------------------------------------------

const ai = new AI({
  provider: selected.provider, // switch providers by changing this one value
  model: selected.provider === "openai" ? "gpt-4.1-mini" : "gemini-2.5-flash",
  apiKey: selected.apiKey,
  systemPrompt: "You are a concise assistant. Answer in at most two sentences.",
  temperature: 0.4,
});

// Events carry counts and lengths — never prompt or response content —
// so they are safe to log straight to your observability pipeline.
ai.on("request:start", (event) => {
  console.log(
    `[event] request:start  provider=${event.provider} messages=${event.messageCount} chars=${event.totalChars}`,
  );
});
ai.on("request:success", (event) => {
  console.log(
    `[event] request:success latency=${event.latencyMs}ms tokens=${event.usage.totalTokens} finish=${event.finishReason}`,
  );
});
ai.on("request:error", (event) => {
  console.log(`[event] request:error code=${event.error.code} retryable=${event.error.retryable}`);
});
ai.on("retry", (event) => {
  console.log(`[event] retry attempt=${event.attempt} delay=${event.delayMs}ms`);
});

// ---------------------------------------------------------------------------
// Prompt templates: format once, validate variable sets at the edges.
// ---------------------------------------------------------------------------

const question = Prompt.template("Explain {concept} to a {audience} in one paragraph.");

const prompt = question.format({ concept: "prompt engineering", audience: "backend developer" });

try {
  // Fails fast with a structured error when a variable is missing.
  question.format({ concept: "only-one-variable" });
} catch (error) {
  console.log(`[template] missing variables caught: ${(error as Error).message}`);
}

// ---------------------------------------------------------------------------
// Generate.
// ---------------------------------------------------------------------------

try {
  const response = await ai.generate(prompt);

  console.log(`\nprovider : ${response.provider}`);
  console.log(`model    : ${response.model}`);
  console.log(`latency  : ${response.latencyMs}ms`);
  console.log(
    `usage    : ${response.usage.promptTokens} prompt + ${response.usage.completionTokens} completion = ${response.usage.totalTokens} tokens`,
  );
  console.log(`finish   : ${response.finishReason}\n`);
  console.log(response.text);

  // Per-instance counters: requests, errors, retries, tokens, latency.
  console.log(`\nstats    : ${JSON.stringify(ai.stats())}`);
} catch (error) {
  // Every failure is a MurphyChainError with a stable `code` and `retryable`.
  if (error instanceof MurphyChainError) {
    console.error(`\n[${error.code}] ${error.message}`);
    console.error(`retryable: ${error.retryable}`);
    if (error.code === "PROVIDER_RATE_LIMIT") {
      console.error("hint: the engine retried automatically; slow your request rate.");
    }
  } else {
    throw error;
  }
}
