import { z } from "zod";
import { ValidationError, truncateMessage } from "./Errors";
import type { JsonSchema, Message } from "./types";
import { type ValidationIssue, formatIssues, isZodSchema } from "./validation";

/**
 * Engine-level structured output.
 *
 * Structured output is implemented once, in the engine, so it works with every
 * provider: the request carries a Zod schema, the engine appends a JSON-only
 * instruction message, parses the model's reply (tolerating code fences and
 * surrounding prose), and validates it against the schema. Failures trigger a
 * bounded repair loop that re-asks with the concrete validation problems.
 *
 * Providers receive nothing new — the instruction travels as an ordinary
 * message — which is why no provider capability gate applies.
 */

/**
 * Total requests per structured-output call: the initial attempt plus at most
 * two repair attempts. Bounded to prevent runaway token spend on a model that
 * cannot produce valid JSON for the schema.
 */
export const STRUCTURED_OUTPUT_MAX_ATTEMPTS = 3;

/** Issues echoed back to the model are capped to bound prompt growth. */
const MAX_REPAIR_ISSUES = 10;
const MAX_REPAIR_ISSUE_CHARS = 500;

/** A resolved structured-output request: validated schema + its instruction. */
export interface OutputSpec {
  schema: z.ZodType;
  /** The JSON-only instruction appended as a trailing user message. */
  instruction: string;
}

/**
 * Validates that `schema` is a Zod v4 schema, converts it to JSON Schema once,
 * and builds the instruction message. Throws `ValidationError` eagerly — a
 * non-convertible schema fails before any request is sent.
 */
export function buildOutputSpec(schema: unknown): OutputSpec {
  if (!isZodSchema(schema)) {
    throw new ValidationError(
      'Structured "output" must be a Zod v4 schema (Zod v3 and plain JSON Schema are not accepted).',
      { field: "output" },
    );
  }
  const jsonSchema = toJsonSchema(schema, "output");
  return { schema, instruction: buildInstruction(jsonSchema) };
}

/** Converts a Zod schema to plain JSON Schema (input side), or throws. */
export function toJsonSchema(schema: z.ZodType, field: string): JsonSchema {
  try {
    const converted = z.toJSONSchema(schema, { io: "input" }) as JsonSchema;
    // `$schema` is noise for providers and rejected by some strict APIs.
    const { $schema: _ignored, ...rest } = converted;
    return rest;
  } catch (error) {
    throw new ValidationError(
      `Structured "output" schema cannot be converted to JSON Schema (${
        error instanceof Error ? error.message : String(error)
      }). Avoid transforms and unsupported types in the output schema.`,
      { field },
    );
  }
}

function buildInstruction(jsonSchema: JsonSchema): string {
  return [
    "Your reply must be a single JSON value that validates against this JSON Schema:",
    JSON.stringify(jsonSchema),
    "Reply with only the JSON value — no prose, no explanations, and no markdown code fences.",
  ].join("\n");
}

/**
 * Builds the repair message sent after a failed attempt: it names the concrete
 * problems so the model can correct them, and re-states the JSON-only rule.
 */
export function buildRepairMessage(issues: readonly ValidationIssue[]): string {
  const lines = issues
    .slice(0, MAX_REPAIR_ISSUES)
    .map((issue) => `- ${issue.path || "(root)"}: ${issue.message.slice(0, MAX_REPAIR_ISSUE_CHARS)}`);
  return [
    "Your previous reply could not be used.",
    "Fix these problems and reply with a single JSON value that validates against the JSON Schema given earlier:",
    ...lines,
    "Reply with only the corrected JSON value — no prose, no explanations, and no markdown code fences.",
  ].join("\n");
}

/** Outcome of parsing + validating a model reply. */
export type StructuredParseResult =
  | { readonly ok: true; readonly value: unknown }
  | { readonly ok: false; readonly issues: ValidationIssue[] };

/**
 * Messages appended after a failed attempt: the model's own reply (truncated)
 * so it can see what it produced, followed by the repair instructions.
 */
export function buildRepairExchange(replyText: string, issues: readonly ValidationIssue[]): Message[] {
  return [
    { role: "assistant", content: truncateMessage(replyText, 4_000) },
    { role: "user", content: buildRepairMessage(issues) },
  ];
}

/**
 * Parses a model reply and validates it against `schema`. Accepts raw JSON,
 * markdown-fenced JSON, and JSON embedded in surrounding prose.
 */
export function parseAndValidate(text: string, schema: z.ZodType): StructuredParseResult {
  const extracted = extractJsonCandidate(text);
  if (!extracted.ok) {
    return { ok: false, issues: [{ path: "", message: extracted.error }] };
  }
  const parsed = schema.safeParse(extracted.value);
  if (!parsed.success) {
    return { ok: false, issues: formatIssues(parsed.error) };
  }
  return { ok: true, value: parsed.data };
}

/**
 * Extracts a JSON value from model text: direct parse first, then the first
 * balanced object/array substring (models sometimes wrap JSON in prose).
 */
export function extractJsonCandidate(
  text: string,
): { readonly ok: true; readonly value: unknown } | { readonly ok: false; readonly error: string } {
  const trimmed = text.trim();
  if (trimmed.length === 0) {
    return { ok: false, error: "the reply was empty" };
  }
  const unfenced = stripCodeFences(trimmed);
  const direct = tryParse(unfenced);
  if (direct.ok) return direct;
  const candidate = findBalancedJson(trimmed);
  if (candidate !== undefined && candidate !== unfenced) {
    const scanned = tryParse(candidate);
    if (scanned.ok) return scanned;
  }
  return { ok: false, error: `the reply is not valid JSON (${direct.error})` };
}

/** Strips one wrapping markdown code fence (```json … ```), if present. */
function stripCodeFences(text: string): string {
  const match = /^```[A-Za-z0-9_-]*[ \t]*\r?\n([\s\S]*?)\r?\n?```$/.exec(text);
  return match !== null ? match[1]!.trim() : text;
}

function tryParse(
  text: string,
): { readonly ok: true; readonly value: unknown } | { readonly ok: false; readonly error: string } {
  try {
    return { ok: true, value: JSON.parse(text) };
  } catch (error) {
    return { ok: false, error: error instanceof Error ? error.message : String(error) };
  }
}

/**
 * Returns the first balanced JSON object/array substring, or undefined.
 * String- and escape-aware so braces inside JSON strings don't unbalance it.
 */
function findBalancedJson(text: string): string | undefined {
  const start = text.search(/[{[]/);
  if (start === -1) return undefined;
  const open = text[start]!;
  const close = open === "{" ? "}" : "]";
  let depth = 0;
  let inString = false;
  let escaped = false;
  for (let i = start; i < text.length; i++) {
    const char = text[i]!;
    if (inString) {
      if (escaped) escaped = false;
      else if (char === "\\") escaped = true;
      else if (char === '"') inString = false;
      continue;
    }
    if (char === '"') {
      inString = true;
    } else if (char === open) {
      depth += 1;
    } else if (char === close) {
      depth -= 1;
      if (depth === 0) return text.slice(start, i + 1);
    }
  }
  return undefined;
}
