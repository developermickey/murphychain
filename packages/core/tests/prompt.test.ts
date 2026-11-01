import { describe, expect, it } from "vitest";
import { Prompt, PromptTemplateError } from "@murphychain/core";

describe("Prompt.template", () => {
  it("substitutes variables", () => {
    const template = Prompt.template("Explain {topic} to {audience}.");
    expect(template.format({ topic: "AI", audience: "a 5-year-old" })).toBe(
      "Explain AI to a 5-year-old.",
    );
  });

  it("exposes variables in order of appearance, deduplicated", () => {
    const template = Prompt.template("{b} then {a} then {b}");
    expect(template.variables).toEqual(["b", "a"]);
  });

  it("keeps the source available for debugging", () => {
    expect(Prompt.template("hi {name}").source).toBe("hi {name}");
  });

  it("renders numbers and booleans", () => {
    const template = Prompt.template("{count} items, done={done}");
    expect(template.format({ count: 3, done: true })).toBe("3 items, done=true");
  });

  it("escapes doubled braces into literal braces", () => {
    const template = Prompt.template("JSON: {{\"key\": {value}}}");
    expect(template.format({ value: 1 })).toBe('JSON: {"key": 1}');
  });

  it("treats invalid placeholder names as literal text", () => {
    const template = Prompt.template('send {{"key": 1}} to { valid }');
    expect(template.variables).toEqual(["valid"]);
    expect(template.format({ valid: "x" })).toBe('send {"key": 1} to x');
  });

  it("keeps unbalanced braces literal", () => {
    const template = Prompt.template("a {broken and } also {name}");
    expect(template.variables).toEqual(["name"]);
    expect(template.format({ name: "x" })).toBe("a {broken and } also x");
  });

  it("rejects missing variables with the missing names", () => {
    const template = Prompt.template("{a} {b}");
    try {
      template.format({ a: "1" });
      expect.unreachable("should have thrown");
    } catch (error) {
      expect(error).toBeInstanceOf(PromptTemplateError);
      expect((error as PromptTemplateError).message).toContain("b");
      expect((error as PromptTemplateError).details.missing).toEqual(["b"]);
    }
  });

  it("rejects unknown variables", () => {
    const template = Prompt.template("{a}");
    expect(() => template.format({ a: "1", b: "2" })).toThrowError(PromptTemplateError);
  });

  it("rejects object and array values instead of stringifying them", () => {
    const template = Prompt.template("{data}");
    expect(() => template.format({ data: { x: 1 } } as never)).toThrowError(PromptTemplateError);
    expect(() => template.format({ data: [1] } as never)).toThrowError(PromptTemplateError);
  });

  it("rejects non-object values", () => {
    const template = Prompt.template("{a}");
    expect(() => template.format("nope" as never)).toThrowError(PromptTemplateError);
    expect(() => template.format(null as never)).toThrowError(PromptTemplateError);
  });

  it("rejects a non-string source", () => {
    expect(() => Prompt.template(42 as never)).toThrowError(PromptTemplateError);
  });
});

describe("PromptTemplate.partial", () => {
  it("pre-fills variables and keeps the rest required", () => {
    const template = Prompt.template("{greeting}, {name}!");
    const bound = template.partial({ greeting: "Hello" });
    expect(bound.variables).toEqual(["name"]);
    expect(bound.format({ name: "Ada" })).toBe("Hello, Ada!");
  });

  it("can pre-fill everything", () => {
    const bound = Prompt.template("{a}-{b}").partial({ a: 1, b: 2 });
    expect(bound.variables).toEqual([]);
    expect(bound.format({})).toBe("1-2");
  });

  it("returns a new template and leaves the original intact", () => {
    const template = Prompt.template("{a} {b}");
    template.partial({ a: "x" });
    expect(template.variables).toEqual(["a", "b"]);
  });

  it("rejects unknown keys", () => {
    expect(() => Prompt.template("{a}").partial({ z: 1 })).toThrowError(PromptTemplateError);
  });

  it("rejects non-object values", () => {
    expect(() => Prompt.template("{a}").partial(null as never)).toThrowError(PromptTemplateError);
  });
});
