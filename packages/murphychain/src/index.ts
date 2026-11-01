/**
 * MurphyChain — a modern, fast, secure, modular AI framework for JavaScript
 * and TypeScript. "Build AI. Connect Intelligence."
 *
 * This is the main entry point. It re-exports the engine API from
 * `@murphychain/core` and registers the built-in provider adapters as *lazy
 * loaders*, so provider SDKs are never evaluated until first use:
 *
 * ```ts
 * import { AI } from "murphychain";
 *
 * const ai = new AI({ provider: "openai", model: "gpt-4.1", apiKey: process.env.OPENAI_API_KEY });
 * const response = await ai.generate("Explain artificial intelligence.");
 * console.log(response.text);
 * ```
 *
 * Provider adapters are optional peer dependencies. Install the ones you use:
 * `pnpm add murphychain @murphychain/openai`.
 */

import { registerProvider, type ProviderFactory } from "@murphychain/core";

// ---------------------------------------------------------------------------
// Core engine API
// ---------------------------------------------------------------------------

export * from "@murphychain/core";

// ---------------------------------------------------------------------------
// Lazy provider registration
// ---------------------------------------------------------------------------

/**
 * Registers a lazy loader for a built-in provider. The adapter package (and
 * its official SDK) is imported on first use; until then it costs nothing.
 *
 * Re-registration is tolerated (idempotent import / hot reload): if a factory
 * is already registered under this name, the existing one is kept.
 */
function registerBuiltinProvider(name: string, loader: () => Promise<ProviderFactory>): void {
  try {
    registerProvider(name, async (config) => (await loader())(config));
  } catch (error) {
    const message = error instanceof Error ? error.message : "";
    if (!message.includes("already registered")) throw error;
  }
}

registerBuiltinProvider("openai", async () => {
  const module = await import("@murphychain/openai");
  return module.createOpenAIProvider;
});

registerBuiltinProvider("gemini", async () => {
  const module = await import("@murphychain/gemini");
  return module.createGeminiProvider;
});

/** Names of the built-in providers shipped with MurphyChain. */
export const builtinProviders: readonly string[] = Object.freeze(["gemini", "openai"]);
