import { ConfigurationError, type ModelProvider, type ProviderFactoryConfig } from "@murphychain/core";
import {
  openAIProviderOptionsSchema,
  type OpenAIProviderConfig,
  type OpenAIProviderOptions,
} from "./config";
import { OpenAIProvider } from "./provider";

export { OpenAIProvider } from "./provider";
export { openAIProviderOptionsSchema } from "./config";
export type { OpenAIProviderConfig, OpenAIProviderOptions } from "./config";
export { mapOpenAIError, parseRetryAfterMs } from "./errors";

/**
 * Factory used by the `murphychain` facade's lazy provider registration.
 * Validates `providerOptions` and constructs {@link OpenAIProvider}.
 */
export function createOpenAIProvider(config: ProviderFactoryConfig): ModelProvider {
  const parsed = openAIProviderOptionsSchema.safeParse(config.options ?? {});
  if (!parsed.success) {
    throw new ConfigurationError("Invalid @murphychain/openai providerOptions.", {
      provider: "openai",
      issues: parsed.error.issues.map((issue) => ({
        path: issue.path.join("."),
        message: issue.message,
      })),
    });
  }
  const options: OpenAIProviderOptions = parsed.data;
  const resolved: OpenAIProviderConfig = {
    ...(config.apiKey !== undefined ? { apiKey: config.apiKey } : {}),
    ...(options.apiKey !== undefined ? { fallbackApiKey: options.apiKey } : {}),
    ...(config.baseUrl !== undefined ? { baseUrl: config.baseUrl } : {}),
    ...(config.model !== undefined ? { defaultModel: config.model } : {}),
    embeddingModel: options.embeddingModel,
    ...(options.organization !== undefined ? { organization: options.organization } : {}),
    ...(options.project !== undefined ? { project: options.project } : {}),
    ...(options.defaultHeaders !== undefined ? { defaultHeaders: options.defaultHeaders } : {}),
    ...(options.fetch !== undefined ? { fetch: options.fetch } : {}),
  };
  return new OpenAIProvider(resolved);
}
