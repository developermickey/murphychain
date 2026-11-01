import { ConfigurationError } from "./Errors";
import type { ProviderFactory } from "./provider";

/**
 * Global provider registry.
 *
 * The `murphychain` facade registers built-in providers as *lazy loaders*
 * (`() => import("@murphychain/openai")`), so provider code — including the
 * official SDKs — is never evaluated until first use. Custom providers can
 * register sync factories or be passed to `new AI({ provider: instance })`.
 */

const registry = new Map<string, ProviderFactory>();

const NAME_PATTERN = /^[a-z][a-z0-9-]*$/;

export interface RegisterProviderOptions {
  /** Replace an existing registration instead of throwing. */
  override?: boolean | undefined;
}

/** Validates and normalizes a provider name (lowercase). */
function normalizeName(name: unknown): string {
  if (typeof name !== "string" || name.length === 0) {
    throw new ConfigurationError("Provider name must be a non-empty string.");
  }
  const key = name.trim().toLowerCase();
  if (!NAME_PATTERN.test(key)) {
    throw new ConfigurationError(
      `Invalid provider name "${name}". Use lowercase letters, digits, and dashes (e.g. "openai", "my-llm").`,
      { provider: name },
    );
  }
  return key;
}

/** Registers a provider factory under `name`. */
export function registerProvider(
  name: string,
  factory: ProviderFactory,
  options: RegisterProviderOptions = {},
): void {
  const key = normalizeName(name);
  if (typeof factory !== "function") {
    throw new ConfigurationError(`Provider "${key}" factory must be a function.`);
  }
  if (registry.has(key) && options.override !== true) {
    throw new ConfigurationError(
      `Provider "${key}" is already registered. Pass { override: true } to replace it.`,
      { provider: key },
    );
  }
  registry.set(key, factory);
}

/** Returns the factory registered under `name`, if any. */
export function resolveProviderFactory(name: string): ProviderFactory | undefined {
  return registry.get(normalizeName(name));
}

/** Lists all registered provider names, sorted. */
export function listProviders(): string[] {
  return [...registry.keys()].sort();
}

/** Removes a registration. Returns `true` when something was removed. */
export function unregisterProvider(name: string): boolean {
  return registry.delete(normalizeName(name));
}
