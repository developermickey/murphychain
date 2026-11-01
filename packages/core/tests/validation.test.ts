import { describe, expect, it } from "vitest";
import { z } from "zod";
import {
  DEFAULT_RETRY_POLICY,
  ValidationError,
  normalizeTool,
} from "@murphychain/core";
import {
  assertInputLimits,
  assertMetadata,
  assertNonNegativeNumber,
  isZodSchema,
  normalizeRetryPolicy,
  parseOrThrow,
  stripUndefined,
} from "../src/validation";

describe("normalizeTool", () => {
  it("converts a Zod schema to JSON Schema exactly once, stripping $schema", () => {
    const tool = normalizeTool({
      name: "get_weather",
      description: "Get the weather",
      parameters: z.object({ city: z.string() }),
    });
    expect(tool.name).toBe("get_weather");
    expect(tool.description).toBe("Get the weather");
    expect(tool.jsonSchema).toEqual({
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"],
    });
    expect(tool.jsonSchema.$schema).toBeUndefined();
  });

  it("passes plain JSON Schema objects through", () => {
    const schema = { type: "object", properties: { q: { type: "string" } } };
    const tool = normalizeTool({ name: "t", description: "d", parameters: schema });
    expect(tool.jsonSchema).toEqual(schema);
  });

  it("rejects invalid tool names", () => {
    expect(() => normalizeTool({ name: "bad name", description: "d", parameters: {} })).toThrowError(
      ValidationError,
    );
    expect(() => normalizeTool({ name: "x".repeat(65), description: "d", parameters: {} })).toThrowError(
      ValidationError,
    );
  });

  it("rejects non-schema parameters", () => {
    expect(() =>
      normalizeTool({ name: "t", description: "d", parameters: 42 as never }),
    ).toThrowError(ValidationError);
  });

  it("rejects unconvertible Zod schemas with an actionable message", () => {
    const custom = z.custom(() => true).transform(() => 1);
    expect(() => normalizeTool({ name: "t", description: "d", parameters: custom })).toThrowError(
      /cannot be converted to JSON Schema/,
    );
  });
});

describe("isZodSchema", () => {
  it("detects Zod v4 schemas and rejects lookalikes", () => {
    expect(isZodSchema(z.string())).toBe(true);
    expect(isZodSchema({})).toBe(false);
    expect(isZodSchema(null)).toBe(false);
  });
});

describe("normalizeRetryPolicy", () => {
  it("returns the defaults for undefined input", () => {
    expect(normalizeRetryPolicy(undefined)).toEqual(DEFAULT_RETRY_POLICY);
  });

  it("merges a bare retry count over the defaults", () => {
    expect(normalizeRetryPolicy(5)).toEqual({ ...DEFAULT_RETRY_POLICY, retries: 5 });
  });

  it("merges a partial policy over the defaults", () => {
    expect(normalizeRetryPolicy({ jitter: false, initialDelayMs: 10 })).toEqual({
      ...DEFAULT_RETRY_POLICY,
      jitter: false,
      initialDelayMs: 10,
    });
  });

  it("rejects out-of-range values", () => {
    expect(() => normalizeRetryPolicy(-1)).toThrowError(ValidationError);
    expect(() => normalizeRetryPolicy({ retries: 99 })).toThrowError(ValidationError);
    expect(() => normalizeRetryPolicy({ backoffFactor: 0.5 })).toThrowError(ValidationError);
    expect(() => normalizeRetryPolicy({ jitter: "yes" as never })).toThrowError(ValidationError);
  });
});

describe("input limits", () => {
  it("rejects too many messages", () => {
    const messages = Array.from({ length: 3 }, () => ({ role: "user" as const, content: "x" }));
    expect(() => assertInputLimits(messages, { maxMessages: 2, maxInputChars: 100 })).toThrowError(
      ValidationError,
    );
  });

  it("rejects totals beyond maxInputChars", () => {
    const messages = [
      { role: "user" as const, content: "abcd" },
      { role: "user" as const, content: "efgh" },
    ];
    expect(() => assertInputLimits(messages, { maxMessages: 10, maxInputChars: 7 })).toThrowError(
      ValidationError,
    );
  });

  it("accepts inputs within limits", () => {
    const messages = [{ role: "user" as const, content: "ok" }];
    expect(() => assertInputLimits(messages, { maxMessages: 10, maxInputChars: 10 })).not.toThrow();
  });
});

describe("option validators", () => {
  it("assertNonNegativeNumber accepts zero and rejects junk", () => {
    expect(assertNonNegativeNumber(0, "x")).toBe(0);
    expect(() => assertNonNegativeNumber(-1, "x")).toThrowError(ValidationError);
    expect(() => assertNonNegativeNumber(Number.NaN, "x")).toThrowError(ValidationError);
    expect(() => assertNonNegativeNumber("1", "x")).toThrowError(ValidationError);
  });

  it("assertMetadata accepts plain objects and rejects arrays", () => {
    expect(assertMetadata({ a: 1 }, "m")).toEqual({ a: 1 });
    expect(() => assertMetadata([], "m")).toThrowError(ValidationError);
    expect(() => assertMetadata(null, "m")).toThrowError(ValidationError);
  });

  it("stripUndefined drops undefined entries", () => {
    expect(stripUndefined({ a: 1, b: undefined, c: "x" })).toEqual({ a: 1, c: "x" });
  });

  it("parseOrThrow raises ValidationError with structured issues", () => {
    try {
      parseOrThrow(z.object({ a: z.string() }), { a: 1 }, "bad");
      expect.unreachable("should have thrown");
    } catch (error) {
      expect(error).toBeInstanceOf(ValidationError);
      expect((error as ValidationError).details.issues).toEqual([
        expect.objectContaining({ path: "a", message: expect.stringContaining("string") }),
      ]);
    }
  });
});
