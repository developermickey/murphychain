import { describe, expect, it } from "vitest";
import { createOpenAIProvider } from "@murphychain/openai";

/**
 * Live integration tests against the real OpenAI API.
 *
 * Skipped unless OPENAI_API_KEY is set, so CI (and offline development) never
 * depends on external services. Run manually:
 *
 *   OPENAI_API_KEY=sk-... pnpm --filter @murphychain/openai test
 */

const apiKey = process.env.OPENAI_API_KEY;
const model = process.env.OPENAI_MODEL ?? "gpt-4.1-mini";

describe.skipIf(apiKey === undefined || apiKey.length === 0)("live OpenAI", () => {
  it("generates text with usage accounting", async () => {
    const provider = createOpenAIProvider({ apiKey: () => apiKey, model });
    const response = await provider.generate({
      messages: [{ role: "user", content: "Reply with exactly: pong" }],
    });
    expect(response.text.toLowerCase()).toContain("pong");
    expect(response.usage.totalTokens).toBeGreaterThan(0);
    expect(response.finishReason).toBe("stop");
  });

  it("streams text deltas", async () => {
    const provider = createOpenAIProvider({ apiKey: () => apiKey, model });
    let text = "";
    let sawTerminal = false;
    for await (const chunk of provider.stream({
      messages: [{ role: "user", content: "Count to three, digits only, comma separated." }],
    })) {
      text += chunk.text;
      if (chunk.finishReason !== undefined) sawTerminal = true;
    }
    expect(text.length).toBeGreaterThan(0);
    expect(sawTerminal).toBe(true);
  });

  it("embeds text", async () => {
    const provider = createOpenAIProvider({ apiKey: () => apiKey });
    const vector = await provider.embed("hello world");
    expect(vector.length).toBeGreaterThan(0);
    expect(Number.isFinite(vector[0])).toBe(true);
  });
});
