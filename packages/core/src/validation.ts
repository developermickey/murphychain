import { z } from "zod";
import { ValidationError } from "./Errors";
import { Model } from "./Model";
import { DEFAULT_RETRY_POLICY, type RetryPolicy } from "./retry";
import type {
  JsonSchema,
  Message,
  NormalizedTool,
  ToolChoice,
  ToolDefinition,
} from "./types";

/**
 * Runtime validation of every external input. Config is validated once at
 * construction; per-call requests are validated once per call — never twice.
 */

export const DEFAULT_MAX_MESSAGES = 10_000;
export const DEFAULT_MAX_INPUT_CHARS = 1_000_000;

export interface InputLimits {
  maxMessages: number;
  maxInputChars: number;
}

/** Zod issue details, safe for logs (paths and messages only). */
export interface ValidationIssue {
  path: string;
  message: string;
}

export function formatIssues(error: z.ZodError): ValidationIssue[] {
  return error.issues.map((issue) => ({
    path: issue.path.join("."),
    message: issue.message,
  }));
}

/** Parses `value` or throws a `ValidationError` carrying structured issues. */
export function parseOrThrow<T>(
  schema: z.ZodType<T>,
  value: unknown,
  message: string,
): T {
  const result = schema.safeParse(value);
  if (!result.success) {
    throw new ValidationError(message, { issues: formatIssues(result.error) });
  }
  return result.data;
}

/** Drops keys whose value is `undefined` (keeps optional fields strict). */
export function stripUndefined<T extends object>(value: T): T {
  const result: Record<string, unknown> = {};
  for (const [key, entry] of Object.entries(value)) {
    if (entry !== undefined) result[key] = entry;
  }
  return result as T;
}

// ---------------------------------------------------------------------------
// Messages
// ---------------------------------------------------------------------------

export const toolCallSchema = z.object({
  id: z.string().min(1),
  name: z.string().min(1),
  arguments: z.record(z.string(), z.unknown()),
});

export const messageSchema = z
  .object({
    role: z.enum(["system", "user", "assistant", "tool"]),
    content: z.string(),
    name: z.string().min(1).optional(),
    toolCallId: z.string().min(1).optional(),
    toolCalls: z.array(toolCallSchema).max(100).optional(),
  })
  .superRefine((message, ctx) => {
    if (message.role === "tool") {
      if (message.toolCallId === undefined) {
        ctx.addIssue({
          code: "custom",
          path: ["toolCallId"],
          message: 'role "tool" messages require "toolCallId"',
        });
      }
      if (message.name === undefined) {
        ctx.addIssue({
          code: "custom",
          path: ["name"],
          message: 'role "tool" messages require "name" (the tool that produced the result)',
        });
      }
    }
    if (message.toolCalls !== undefined && message.role !== "assistant") {
      ctx.addIssue({
        code: "custom",
        path: ["toolCalls"],
        message: 'only role "assistant" messages may carry "toolCalls"',
      });
    }
  });

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

export const toolChoiceSchema = z.union([
  z.enum(["auto", "none", "required"]),
  z.object({ name: z.string().min(1).max(64) }).strict(),
]);

/** Detects a Zod v4 schema (v4 tags schemas with `_zod`). */
export function isZodSchema(value: unknown): value is z.ZodType {
  return typeof value === "object" && value !== null && "_zod" in value;
}

/** Detects a Zod v3 schema so we can give an actionable upgrade error. */
function isZod3Schema(value: unknown): boolean {
  return (
    typeof value === "object" &&
    value !== null &&
    "_def" in value &&
    "safeParse" in value &&
    !("_zod" in value)
  );
}

function isPlainJsonObject(value: unknown): value is JsonSchema {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const proto: unknown = Object.getPrototypeOf(value);
  return proto === Object.prototype || proto === null;
}

export const toolDefinitionSchema = z
  .object({
    name: z
      .string()
      .regex(
        /^[A-Za-z0-9_-]{1,64}$/,
        'Tool names must match [A-Za-z0-9_-]{1,64} (letters, digits, "_", "-")',
      ),
    description: z.string().max(10_000),
    // Type-only: the superRefine below performs (and explains) the checks.
    parameters: z.custom<ToolDefinition["parameters"]>(() => true),
  })
  .superRefine((tool, ctx) => {
    if (isZod3Schema(tool.parameters)) {
      ctx.addIssue({
        code: "custom",
        path: ["parameters"],
        message: "Zod v3 schemas are not supported — use Zod v4 (>=4.0)",
      });
    } else if (!isZodSchema(tool.parameters) && !isPlainJsonObject(tool.parameters)) {
      ctx.addIssue({
        code: "custom",
        path: ["parameters"],
        message: 'parameters must be a Zod schema or a plain JSON Schema object',
      });
    }
  });

/**
 * Normalizes a user-facing tool definition: Zod schemas are converted to
 * JSON Schema exactly once (engine-side), so provider adapters only ever
 * deal with plain JSON Schema.
 */
export function normalizeTool(tool: ToolDefinition): NormalizedTool {
  const parsed = parseOrThrow(
    toolDefinitionSchema,
    tool,
    `Invalid tool definition${typeof tool?.name === "string" ? ` "${tool.name}"` : ""}.`,
  );
  if (isZodSchema(parsed.parameters)) {
    let jsonSchema: JsonSchema;
    try {
      jsonSchema = z.toJSONSchema(parsed.parameters, { io: "input" }) as JsonSchema;
    } catch (error) {
      throw new ValidationError(
        `Tool "${parsed.name}": Zod schema cannot be converted to JSON Schema (${
          error instanceof Error ? error.message : String(error)
        }). Avoid transforms and unsupported types in tool parameter schemas.`,
        { tool: parsed.name },
      );
    }
    return { name: parsed.name, description: parsed.description, jsonSchema: stripSchema(jsonSchema) };
  }
  return {
    name: parsed.name,
    description: parsed.description,
    jsonSchema: parsed.parameters as JsonSchema,
  };
}

/** Removes keywords that some provider APIs reject as unknown fields. */
function stripSchema(schema: JsonSchema): JsonSchema {
  const { $schema: _ignored, ...rest } = schema;
  return rest;
}

// ---------------------------------------------------------------------------
// Generation requests
// ---------------------------------------------------------------------------

const temperatureSchema = z.number().finite().min(0);
const maxTokensSchema = z.number().int().positive();
const topPSchema = z.number().min(0).max(1);
const stopSchema = z.array(z.string().min(1).max(4096)).max(16);

export const generateRequestSchema = z
  .object({
    messages: z.array(messageSchema).min(1).optional(),
    prompt: z.string().min(1).optional(),
    system: z.string().optional(),
    model: z.union([z.string().min(1), z.instanceof(Model)]).optional(),
    temperature: temperatureSchema.optional(),
    maxTokens: maxTokensSchema.optional(),
    topP: topPSchema.optional(),
    stop: stopSchema.optional(),
    tools: z.array(toolDefinitionSchema).optional(),
    toolChoice: toolChoiceSchema.optional(),
    output: z
      .custom<z.ZodType>((value) => isZodSchema(value), {
        message: "output must be a Zod v4 schema (Zod v3 and plain JSON Schema are not supported)",
      })
      .optional(),
  })
  .strict()
  .superRefine((request, ctx) => {
    if (request.messages === undefined && request.prompt === undefined) {
      ctx.addIssue({
        code: "custom",
        path: ["prompt"],
        message: 'provide either "messages" or "prompt"',
      });
    }
    if (request.messages !== undefined && request.prompt !== undefined) {
      ctx.addIssue({
        code: "custom",
        path: ["prompt"],
        message: 'provide only one of "messages" or "prompt"',
      });
    }
    if (
      request.toolChoice !== undefined &&
      (request.tools === undefined || request.tools.length === 0)
    ) {
      ctx.addIssue({
        code: "custom",
        path: ["toolChoice"],
        message: '"toolChoice" requires at least one tool in "tools"',
      });
    }
    if (request.output !== undefined && request.tools !== undefined) {
      ctx.addIssue({
        code: "custom",
        path: ["output"],
        message: '"output" cannot be combined with "tools"',
      });
    }
  });

// ---------------------------------------------------------------------------
// Retry policy
// ---------------------------------------------------------------------------

export const retryPolicySchema = z.object({
  retries: z.number().int().min(0).max(20),
  initialDelayMs: z.number().min(0).max(60_000),
  maxDelayMs: z.number().min(0).max(300_000),
  backoffFactor: z.number().min(1).max(10),
  jitter: z.boolean(),
});

/** Merges a partial policy (or a bare retry count) over the defaults. */
export function normalizeRetryPolicy(input: unknown): RetryPolicy {
  const merged: RetryPolicy =
    typeof input === "number"
      ? { ...DEFAULT_RETRY_POLICY, retries: input }
      : { ...DEFAULT_RETRY_POLICY, ...(input ?? {}) };
  return parseOrThrow(retryPolicySchema, merged, "Invalid retry policy.");
}

// ---------------------------------------------------------------------------
// Limits
// ---------------------------------------------------------------------------

/** Enforces message-count and total-size budgets before any network work. */
export function assertInputLimits(messages: readonly Message[], limits: InputLimits): void {
  if (messages.length > limits.maxMessages) {
    throw new ValidationError(
      `Message count ${messages.length} exceeds the configured limit of ${limits.maxMessages}.`,
      { count: messages.length, limit: limits.maxMessages },
    );
  }
  let total = 0;
  for (const message of messages) {
    total += message.content.length;
    if (total > limits.maxInputChars) {
      throw new ValidationError(
        `Input exceeds the configured limit of ${limits.maxInputChars} characters.`,
        { limit: limits.maxInputChars },
      );
    }
  }
}

/** Validates an option value that must be a finite number ≥ 0. */
export function assertNonNegativeNumber(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    throw new ValidationError(`"${field}" must be a finite number >= 0.`, { field });
  }
  return value;
}

/** Validates an option value that must be an integer ≥ 1. Returns the value. */
export function assertPositiveInteger(value: unknown, field: string): number {
  if (typeof value !== "number" || !Number.isInteger(value) || value < 1) {
    throw new ValidationError(`"${field}" must be an integer >= 1.`, { field });
  }
  return value;
}

/** Validates a plain-object option (metadata maps). */
export function assertMetadata(value: unknown, field: string): Record<string, unknown> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new ValidationError(`"${field}" must be a plain object.`, { field });
  }
  return value as Record<string, unknown>;
}

// Convenience re-exports used by the AI facade.
export { stopSchema, temperatureSchema, maxTokensSchema, topPSchema };
export type { ToolChoice };
