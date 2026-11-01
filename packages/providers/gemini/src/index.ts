import {
  ConfigurationError,
  type ModelProvider,
  type ProviderFactoryConfig,
} from "@murphychain/core";
import {
  geminiProviderOptionsSchema,
  type GeminiProviderConfig,
  type GeminiProviderOptions,
} from "./config";
import { GeminiProvider } from "./provider";

export { GeminiProvider } from "./provider";
export { geminiProviderOptionsSchema } from "./config";
export type { GeminiProviderConfig, GeminiProviderOptions } from "./config";
export { mapGeminiError } from "./errors";

/**
 * Factory used by the `murphychain` facade's lazy provider registration.
 * Validates `providerOptions` and constructs {@link GeminiProvider}.
 */
export function createGeminiProvider(config: ProviderFactoryConfig): ModelProvider {
  const parsed = geminiProviderOptionsSchema.safeParse(config.options ?? {});
  if (!parsed.success) {
    throw new ConfigurationError("Invalid @murphychain/gemini providerOptions.", {
      provider: "gemini",
      issues: parsed.error.issues.map((issue) => ({
        path: issue.path.join("."),
        message: issue.message,
      })),
    });
  }
  const options: GeminiProviderOptions = parsed.data;
  const resolved: GeminiProviderConfig = {
    ...(config.apiKey !== undefined ? { apiKey: config.apiKey } : {}),
    ...(options.apiKey !== undefined ? { fallbackApiKey: options.apiKey } : {}),
    ...(config.baseUrl !== undefined ? { baseUrl: config.baseUrl } : {}),
    ...(config.model !== undefined ? { defaultModel: config.model } : {}),
    embeddingModel: options.embeddingModel,
    ...(options.apiVersion !== undefined ? { apiVersion: options.apiVersion } : {}),
    ...(options.headers !== undefined ? { headers: options.headers } : {}),
    ...(options.fetch !== undefined ? { fetch: options.fetch } : {}),
  };
  return new GeminiProvider(resolved);
}
