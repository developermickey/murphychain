import { describe, expect, it } from "vitest";
import { createGeminiProvider } from "@murphychain/gemini";

/**
 * Live integration tests against the real Gemini API.
 *
 * Skipped unless GEMINI_API_KEY (or GOOGLE_API_KEY) is set, so CI (and
 * offline development) never depends on external services. Run manually:
 *
 *   GEMINI_API_KEY=AIza... pnpm --filter @murphychain/gemini test
 */

const apiKey = process.env.GEMINI_API_KEY ?? process.env.GOOGLE_API_KEY;
const model = process.env.GEMINI_MODEL ?? "gemini-2.5-flash";

describe.skipIf(apiKey === undefined || apiKey.length === 0)("live Gemini", () => {
  it("generates text with usage accounting", async () => {
    const provider = createGeminiProvider({ apiKey: () => apiKey, model });
    const response = await provider.generate({
      messages: [{ role: "user", content: "Reply with exactly: pong" }],
    });
    expect(response.text.toLowerCase()).toContain("pong");
    expect(response.usage.totalTokens).toBeGreaterThan(0);
  });

  it("streams text deltas", async () => {
    const provider = createGeminiProvider({ apiKey: () => apiKey, model });
    let text = "";
    for await (const chunk of provider.stream({
      messages: [{ role: "user", content: "Count to three, digits only, comma separated." }],
    })) {
      text += chunk.text;
    }
    expect(text.length).toBeGreaterThan(0);
  });

  it("embeds text", async () => {
    const provider = createGeminiProvider({ apiKey: () => apiKey });
    const vector = await provider.embed("hello world");
    expect(vector.length).toBeGreaterThan(0);
    expect(Number.isFinite(vector[0])).toBe(true);
  });
});
