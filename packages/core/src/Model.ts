import { z } from "zod";
import { ValidationError } from "./Errors";

/** Default generation parameters attached to a model. */
export interface ModelDefaults {
  temperature?: number | undefined;
  maxTokens?: number | undefined;
  topP?: number | undefined;
  stop?: string[] | undefined;
}

const defaultsSchema = z.object({
  temperature: z.number().finite().min(0).optional(),
  maxTokens: z.number().int().positive().optional(),
  topP: z.number().min(0).max(1).optional(),
  stop: z.array(z.string().min(1).max(4096)).max(16).optional(),
});

/**
 * An immutable reference to a model with optional default parameters.
 *
 * Merge order for a request: per-request value → `Model` defaults →
 * `AIConfig` defaults → provider default.
 *
 * ```ts
 * const fast = new Model("gpt-4.1-mini", { temperature: 0 });
 * const res = await ai.generate({ prompt: "Hi", model: fast });
 * ```
 */
export class Model {
  readonly name: string;
  readonly defaults: Readonly<ModelDefaults>;

  constructor(name: string, defaults: ModelDefaults = {}) {
    if (typeof name !== "string" || name.trim().length === 0) {
      throw new ValidationError("Model name must be a non-empty string.");
    }
    const parsed = defaultsSchema.safeParse(defaults);
    if (!parsed.success) {
      throw new ValidationError("Invalid model defaults.", {
        issues: parsed.error.issues.map((issue) => ({
          path: issue.path.join("."),
          message: issue.message,
        })),
      });
    }
    this.name = name.trim();
    this.defaults = Object.freeze({ ...parsed.data });
  }

  /** Returns a copy of this model with additional (or replaced) defaults. */
  withDefaults(defaults: ModelDefaults): Model {
    return new Model(this.name, { ...this.defaults, ...defaults });
  }

  toString(): string {
    return this.name;
  }
}
