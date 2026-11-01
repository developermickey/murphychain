/**
 * Provider switching — the same application code against different backends.
 *
 * Requires API keys (one is enough; both lets you see the switch side by side):
 *   OPENAI_API_KEY=sk-...   pnpm example:provider-switch
 *   GEMINI_API_KEY=AIza...  pnpm example:provider-switch
 *
 * The ONLY thing that changes between runs is the `provider` value passed to
 * the constructor. Application code — prompt, response handling, events,
 * error handling — is identical, because MurphyChain normalizes both providers
 * onto one contract.
 */
import { AI, MurphyChainError, type AIResponse } from "murphychain";

interface Target {
  provider: "openai" | "gemini";
  model: string;
  apiKey: string;
}

/** Builds the list of providers we have credentials for. */
function availableTargets(): Target[] {
  const targets: Target[] = [];
  const openaiKey = process.env.OPENAI_API_KEY;
  if (openaiKey !== undefined && openaiKey.length > 0) {
    targets.push({ provider: "openai", model: "gpt-4.1-mini", apiKey: openaiKey });
  }
  const geminiKey = process.env.GEMINI_API_KEY ?? process.env.GOOGLE_API_KEY;
  if (geminiKey !== undefined && geminiKey.length > 0) {
    targets.push({ provider: "gemini", model: "gemini-2.5-flash", apiKey: geminiKey });
  }
  return targets;
}

const targets = availableTargets();

if (targets.length === 0) {
  console.log("No API key found — set one or both to run this example:");
  console.log("  OPENAI_API_KEY=sk-...   pnpm example:provider-switch");
  console.log("  GEMINI_API_KEY=AIza...  pnpm example:provider-switch");
  process.exit(0);
}

// ---------------------------------------------------------------------------
// This is the entire provider-specific surface of the application.
// ---------------------------------------------------------------------------

async function ask(target: Target): Promise<AIResponse> {
  const ai = new AI({
    provider: target.provider, // <-- the one-line switch
    model: target.model,
    apiKey: target.apiKey,
    temperature: 0.3,
  });

  ai.on("request:success", (event) => {
    console.log(
      `  [${target.provider}] ${event.latencyMs}ms, ${event.usage.totalTokens} tokens, finish=${event.finishReason}`,
    );
  });

  return ai.generate("In one sentence: what is a vector database?");
}

// ---------------------------------------------------------------------------
// Run the identical code path against every configured provider and verify
// the response contract is the same shape regardless of backend.
// ---------------------------------------------------------------------------

const CONTRACT_KEYS = [
  "text",
  "provider",
  "model",
  "message",
  "finishReason",
  "usage",
  "latencyMs",
] as const;

for (const target of targets) {
  console.log(`\n=== provider: ${target.provider} (model: ${target.model}) ===`);
  try {
    const response = await ask(target);

    // Same fields, same types, same semantics — whichever provider served it.
    for (const key of CONTRACT_KEYS) {
      if (!(key in response)) {
        throw new Error(`response contract violated: missing "${key}"`);
      }
    }

    console.log(`  provider field : ${response.provider}`);
    console.log(`  model field    : ${response.model}`);
    console.log(`  finishReason   : ${response.finishReason}`);
    console.log(`  text           : ${response.text}`);
  } catch (error) {
    // Errors are normalized too: one taxonomy, regardless of backend.
    if (error instanceof MurphyChainError) {
      console.error(`  [${error.code}] ${error.message} (retryable=${error.retryable})`);
    } else {
      throw error;
    }
  }
}

if (targets.length > 1) {
  console.log(
    "\nBoth runs used the same application code — only the constructor's `provider` value differed.",
  );
} else {
  console.log(
    `\nOnly one key found. Set the other key to see both providers run identical code.`,
  );
}
