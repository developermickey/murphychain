# @murphychain/core

## 0.3.0

### Minor Changes

- **Agent runtime and memory.**
  
  - `Agent`: a bounded tool-calling loop — seed (system → memory → input), ask with tools, execute requested calls, repeat until the model answers or `maxSteps` (default 10) is spent. The final step's tool calls are never executed and never persisted; memory updates only after a run that made progress; instances are single-flight; the caller's `signal` is checked between steps; `timeoutMs` applies per model call; `onToolError: "return" | "throw"`; content-free `step`/`tool`/`finish` events and a per-step trace in `AgentResult`.
  - `Memory` abstraction with two implementations: `BufferMemory` (bounded FIFO with atomic assistant-toolCall + tool-result eviction) and `SummarizingMemory` (recent messages verbatim, older ones folded into a rolling summary via one `generate()` call, bounded to 4000 chars).
  - `executeToolCalls` now reports each `ExecutedToolCall` (with `durationMs`) through a new isolated `onCallComplete` callback.
  - `EventBus` generalized to any event map (existing `AIEventMap` types unchanged); new exported `EventListener<M, K>` type.
  - Tool Zod → JSON Schema conversion is cached per tool and reused across agent steps.
  - Offline example `pnpm example:agent` runs in CI.

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
