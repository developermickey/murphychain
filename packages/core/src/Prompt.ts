import { PromptTemplateError } from "./Errors";

/**
 * Prompt templates with `{variable}` placeholders.
 *
 * - `{name}` inserts a variable (strict: missing *and* unknown variables are
 *   errors, so typos fail loudly instead of rendering `undefined`).
 * - `{{` / `}}` insert literal braces (JSON in prompts stays intact).
 * - Anything that is not a valid placeholder (e.g. `{"key": 1}`) is literal.
 */

type Part = { kind: "text"; value: string } | { kind: "var"; name: string };

const TOKEN = /\{\{|\}\}|\{([^{}]*)\}/g;
const VALID_NAME = /^[A-Za-z_][A-Za-z0-9_]*$/;

export type PromptValue = string | number | boolean;

export class PromptTemplate {
  readonly source: string;
  private readonly parts: readonly Part[];
  private readonly names: readonly string[];

  private constructor(source: string, parts: Part[], names: string[]) {
    this.source = source;
    this.parts = Object.freeze(parts);
    this.names = Object.freeze(names);
  }

  /** Parses a template source string. */
  static parse(source: string): PromptTemplate {
    if (typeof source !== "string") {
      throw new PromptTemplateError("Template source must be a string.");
    }
    const parts: Part[] = [];
    const names: string[] = [];
    const seen = new Set<string>();
    let lastIndex = 0;
    TOKEN.lastIndex = 0;
    let match = TOKEN.exec(source);
    while (match !== null) {
      if (match.index > lastIndex) {
        parts.push({ kind: "text", value: source.slice(lastIndex, match.index) });
      }
      const raw = match[0];
      const captured = match[1];
      if (raw === "{{") {
        parts.push({ kind: "text", value: "{" });
      } else if (raw === "}}") {
        parts.push({ kind: "text", value: "}" });
      } else {
        const name = (captured ?? "").trim();
        if (VALID_NAME.test(name)) {
          parts.push({ kind: "var", name });
          if (!seen.has(name)) {
            seen.add(name);
            names.push(name);
          }
        } else {
          // Not a placeholder (e.g. JSON braces) — keep it literal.
          parts.push({ kind: "text", value: raw });
        }
      }
      lastIndex = match.index + raw.length;
      match = TOKEN.exec(source);
    }
    if (lastIndex < source.length) {
      parts.push({ kind: "text", value: source.slice(lastIndex) });
    }
    return new PromptTemplate(source, parts, names);
  }

  /** Variable names required by this template, in order of appearance. */
  get variables(): readonly string[] {
    return this.names;
  }

  /**
   * Renders the template. Every declared variable must be provided and no
   * unknown keys may be passed. Values must be strings, numbers, or
   * booleans — objects and arrays are rejected rather than stringified into
   * `[object Object]`.
   */
  format(values: Record<string, PromptValue>): string {
    if (typeof values !== "object" || values === null || Array.isArray(values)) {
      throw new PromptTemplateError("Template variables must be a plain object.");
    }
    const missing = this.names.filter(
      (name) => !Object.hasOwn(values, name) || values[name] === undefined,
    );
    if (missing.length > 0) {
      throw new PromptTemplateError(
        `Missing template variables: ${missing.join(", ")}.`,
        { missing, expected: [...this.names] },
      );
    }
    const unknown = Object.keys(values).filter((key) => !this.names.includes(key));
    if (unknown.length > 0) {
      throw new PromptTemplateError(
        `Unknown template variables: ${unknown.join(", ")}.`,
        { unknown, expected: [...this.names] },
      );
    }
    let output = "";
    for (const part of this.parts) {
      if (part.kind === "text") {
        output += part.value;
      } else {
        output += String(this.read(values, part.name));
      }
    }
    return output;
  }

  /**
   * Returns a new template with some variables pre-filled. Unknown keys are
   * rejected; missing keys are allowed (they remain required at format time).
   */
  partial(values: Record<string, PromptValue>): PromptTemplate {
    if (typeof values !== "object" || values === null || Array.isArray(values)) {
      throw new PromptTemplateError("Template variables must be a plain object.");
    }
    const unknown = Object.keys(values).filter((key) => !this.names.includes(key));
    if (unknown.length > 0) {
      throw new PromptTemplateError(
        `Unknown template variables: ${unknown.join(", ")}.`,
        { unknown, expected: [...this.names] },
      );
    }
    const parts: Part[] = [];
    const names: string[] = [];
    for (const part of this.parts) {
      if (part.kind === "var" && Object.hasOwn(values, part.name)) {
        parts.push({ kind: "text", value: String(this.read(values, part.name)) });
      } else if (part.kind === "var") {
        parts.push(part);
        if (!names.includes(part.name)) names.push(part.name);
      } else {
        parts.push(part);
      }
    }
    return new PromptTemplate(this.source, parts, names);
  }

  private read(values: Record<string, PromptValue>, name: string): PromptValue {
    const value = values[name];
    if (value === undefined || value === null || typeof value === "object") {
      throw new PromptTemplateError(
        `Variable "${name}" must be a string, number, or boolean.`,
        { variable: name },
      );
    }
    return value;
  }
}

/** Entry point for prompt templates: `Prompt.template("…{topic}…")`. */
export class Prompt {
  private constructor() {}

  static template(source: string): PromptTemplate {
    return PromptTemplate.parse(source);
  }
}
