import { z } from "zod";

/** Default model used by `ai.embed()` when none is configured. */
export const DEFAULT_GEMINI_EMBEDDING_MODEL = "text-embedding-004";

/**
 * Provider-specific options, passed via `new AI({ providerOptions: { ... } })`.
 * Unknown keys are rejected (`.strict()`) so typos fail loudly.
 */
export const geminiProviderOptionsSchema = z
  .object({
    /** Static API key. Prefer `apiKey` in the AI config (supports rotation). */
    apiKey: z.string().min(1).optional(),
    /** API endpoint override (must be http/https). Tested against mocks. */
    baseUrl: z
      .url()
      .refine((value) => value.startsWith("http://") || value.startsWith("https://"), {
        message: "baseUrl must use http(s)",
      })
      .optional(),
    /** Gemini API version. Default: v1beta. */
    apiVersion: z
      .string()
      .regex(/^[a-z0-9]+$/, "apiVersion must be lowercase alphanumerics (e.g. v1beta)")
      .optional(),
    /** Extra headers sent with every request (e.g. proxies). */
    headers: z.record(z.string(), z.string()).optional(),
    /** Custom fetch implementation (proxies, testing, instrumentation). */
    fetch: z
      .custom<typeof globalThis.fetch>((value) => typeof value === "function", {
        message: "fetch must be a function",
      })
      .optional(),
    /** Model used by embed()/embedMany(). Default: text-embedding-004. */
    embeddingModel: z.string().min(1).default(DEFAULT_GEMINI_EMBEDDING_MODEL),
  })
  .strict();

export type GeminiProviderOptions = z.infer<typeof geminiProviderOptionsSchema>;

/** Fully resolved provider configuration used by {@link GeminiProvider}. */
export interface GeminiProviderConfig {
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
  apiVersion?: string | undefined;
  headers?: Record<string, string> | undefined;
  fetch?: typeof globalThis.fetch | undefined;
}
