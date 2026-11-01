# @murphychain/gemini

## 0.2.0

### Minor Changes

- Workflows: tools, chains, and structured output.
  
  - `Tool.define` — named, schema-validated tools with Zod-inferred argument
    types, eager definition validation, and cached JSON Schema conversion.
    `executeToolCalls` dispatches model-requested calls sequentially and
    serializes failures (redacted) so the model can self-correct.
  - `Chain` — explicit, named step sequences over shared state with per-step
    timing, `ChainStepError` on failure (or `{ onError: "continue" }` to
    aggregate), and abort-signal cancellation between steps.
  - Structured output — `generate({ output: schema })` and
    `generateObject(input, schema)` validate the model's JSON reply against a
    Zod v4 schema, portably across every provider (engine-level, no provider
    capability needed). Invalid replies trigger a bounded repair loop (3
    attempts) that re-asks with the concrete validation issues; exhaustion
    throws `OutputValidationError` carrying the issues and attempt count.
    Results are exposed as `response.object`.

### Patch Changes

- Updated dependencies
  - @murphychain/core@0.2.0
