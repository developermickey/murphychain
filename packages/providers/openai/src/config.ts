import { z } from "zod";

/** Default model used by `ai.embed()` when none is configured. */
export const DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small";

/**
 * Provider-specific options, passed via `new AI({ providerOptions: { ... } })`.
 * Unknown keys are rejected (`.strict()`) so typos fail loudly.
 */
export const openAIProviderOptionsSchema = z
  .object({
    /** Static API key. Prefer `apiKey` in the AI config (supports rotation). */
    apiKey: z.string().min(1).optional(),
    /** OpenAI organization id. */
    organization: z.string().min(1).optional(),
    /** OpenAI project id. */
    project: z.string().min(1).optional(),
    /** API endpoint override (must be http/https). Tested against mocks. */
    baseURL: z
      .url()
      .refine((value) => value.startsWith("http://") || value.startsWith("https://"), {
        message: "baseURL must use http(s)",
      })
      .optional(),
    /** Extra headers sent with every request (e.g. proxies). */
    defaultHeaders: z.record(z.string(), z.string()).optional(),
    /** Custom fetch implementation (proxies, testing, instrumentation). */
    fetch: z
      .custom<typeof globalThis.fetch>((value) => typeof value === "function", {
        message: "fetch must be a function",
      })
      .optional(),
    /** Model used by embed()/embedMany(). Default: text-embedding-3-small. */
    embeddingModel: z.string().min(1).default(DEFAULT_EMBEDDING_MODEL),
  })
  .strict();

export type OpenAIProviderOptions = z.infer<typeof openAIProviderOptionsSchema>;

/** Fully resolved provider configuration used by {@link OpenAIProvider}. */
export interface OpenAIProviderConfig {
  /** Per-request key accessor (credential rotation). */
  apiKey?: (() => string | undefined) | undefined;
  /** Static key fallback (from `providerOptions.apiKey`). */
  fallbackApiKey?: string | undefined;
  /** Endpoint override (validated by the engine before reaching us). */
  baseUrl?: string | undefined;
  /** Default model from `new AI({ model })`. */
  defaultModel?: string | undefined;
  /** Embedding model (validated, defaults applied by the schema). */
  embeddingModel: string;
  organization?: string | undefined;
  project?: string | undefined;
  defaultHeaders?: Record<string, string> | undefined;
  fetch?: typeof globalThis.fetch | undefined;
}
