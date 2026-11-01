import { afterEach, describe, expect, it } from "vitest";
import {
  ConfigurationError,
  listProviders,
  registerProvider,
  resolveProviderFactory,
  unregisterProvider,
} from "@murphychain/core";

const scratch = ["reg-a", "reg-b", "reg-c", "my-llm"];

afterEach(() => {
  for (const name of scratch) unregisterProvider(name);
});

const factory = (): never => {
  throw new Error("not used in tests");
};

describe("provider registry", () => {
  it("registers, resolves, and lists providers sorted", () => {
    registerProvider("reg-b", factory);
    registerProvider("reg-a", factory);
    expect(listProviders()).toEqual(["reg-a", "reg-b"]);
    expect(resolveProviderFactory("reg-a")).toBe(factory);
    expect(resolveProviderFactory("reg-b")).toBe(factory);
  });

  it("normalizes names to lowercase", () => {
    registerProvider("My-LLM", factory);
    expect(listProviders()).toContain("my-llm");
    expect(resolveProviderFactory("MY-LLM")).toBe(factory);
  });

  it("returns undefined for unknown providers", () => {
    expect(resolveProviderFactory("nope")).toBeUndefined();
    expect(unregisterProvider("nope")).toBe(false);
  });

  it("rejects duplicate registrations unless override is set", () => {
    registerProvider("reg-a", factory);
    expect(() => registerProvider("reg-a", factory)).toThrowError(ConfigurationError);
    const replacement = (): never => {
      throw new Error("replacement");
    };
    registerProvider("reg-a", replacement, { override: true });
    expect(resolveProviderFactory("reg-a")).toBe(replacement);
  });

  it("validates names", () => {
    expect(() => registerProvider("", factory)).toThrowError(ConfigurationError);
    expect(() => registerProvider("   ", factory)).toThrowError(ConfigurationError);
    expect(() => registerProvider("1abc", factory)).toThrowError(ConfigurationError);
    expect(() => registerProvider("has space", factory)).toThrowError(ConfigurationError);
    expect(() => registerProvider("has_underscore", factory)).toThrowError(ConfigurationError);
    expect(() => registerProvider(42 as never, factory)).toThrowError(ConfigurationError);
  });

  it("rejects non-function factories", () => {
    expect(() => registerProvider("reg-a", 42 as never)).toThrowError(ConfigurationError);
  });

  it("unregisters previously registered providers", () => {
    registerProvider("reg-c", factory);
    expect(unregisterProvider("reg-c")).toBe(true);
    expect(resolveProviderFactory("reg-c")).toBeUndefined();
  });
});
