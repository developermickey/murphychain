import { describe, expect, it } from "vitest";
import { Model, ValidationError } from "@murphychain/core";

describe("Model", () => {
  it("stores a trimmed name and frozen defaults", () => {
    const model = new Model(" gpt-4.1-mini ", { temperature: 0 });
    expect(model.name).toBe("gpt-4.1-mini");
    expect(Object.isFrozen(model.defaults)).toBe(true);
    expect(model.toString()).toBe("gpt-4.1-mini");
  });

  it("accepts a bare name", () => {
    expect(new Model("m").defaults).toEqual({});
  });

  it("rejects an empty name", () => {
    expect(() => new Model("")).toThrowError(ValidationError);
    expect(() => new Model("   ")).toThrowError(ValidationError);
    expect(() => new Model(42 as never)).toThrowError(ValidationError);
  });

  it("rejects invalid defaults", () => {
    expect(() => new Model("m", { temperature: -1 })).toThrowError(ValidationError);
    expect(() => new Model("m", { maxTokens: 0 })).toThrowError(ValidationError);
    expect(() => new Model("m", { topP: 2 })).toThrowError(ValidationError);
    expect(() => new Model("m", { stop: [""] })).toThrowError(ValidationError);
  });

  it("withDefaults merges over existing defaults without mutating the original", () => {
    const base = new Model("m", { temperature: 0.2, maxTokens: 100 });
    const derived = base.withDefaults({ temperature: 0.9 });
    expect(derived.name).toBe("m");
    expect(derived.defaults).toEqual({ temperature: 0.9, maxTokens: 100 });
    expect(base.defaults.temperature).toBe(0.2);
  });
});
