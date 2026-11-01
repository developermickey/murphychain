import { fileURLToPath } from "node:url";
import { defineConfig } from "vitest/config";

const src = (path: string): string =>
  fileURLToPath(new URL(path, import.meta.url));

export default defineConfig({
  resolve: {
    alias: {
      "@murphychain/core": src("./packages/core/src/index.ts"),
      "@murphychain/openai": src("./packages/providers/openai/src/index.ts"),
      "@murphychain/gemini": src("./packages/providers/gemini/src/index.ts"),
      murphychain: src("./packages/murphychain/src/index.ts"),
    },
  },
  test: {
    include: ["packages/**/tests/**/*.test.ts"],
    environment: "node",
    testTimeout: 10_000,
    hookTimeout: 10_000,
  },
});
