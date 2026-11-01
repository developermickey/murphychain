import { ChainStepError, ExecutionCancelledError, ValidationError } from "./Errors";

/**
 * Chains: an explicit, typed sequence of steps over shared state.
 *
 * A chain is the smallest possible "workflow": each step receives the current
 * state, returns the next state (or a partial merge), and runs only after the
 * previous step succeeded. Unlike prompt-only pipelines, chains give you real
 * control flow — branches and loops live in ordinary TypeScript around
 * `await chain.run(state)`.
 *
 * ```ts
 * const chain = new Chain<{ topic: string; summary?: string }>()
 *   .step("draft", async (state) => ({
 *     summary: (await ai.generate(`Summarize ${state.topic}`)).text,
 *   }))
 *   .step("polish", async (state) => ({
 *     summary: state.summary?.toUpperCase(),
 *   }));
 *
 * const { state, steps } = await chain.run({ topic: "gravity" });
 * ```
 *
 * Steps run sequentially; each step's output becomes the next step's input.
 * Failures abort the chain with a {@link ChainStepError} naming the step, and
 * an aborted `signal` stops it between steps.
 */

/** What a chain step receives besides the state. */
export interface ChainStepContext {
  /** The step's registered name. */
  name: string;
  /** Zero-based position in the chain. */
  index: number;
  /** Cancellation for the whole run (shared across steps). */
  signal?: AbortSignal | undefined;
}

/**
 * One step: receives the current state, returns the next state (a full state
 * or a partial to merge over it). May be async.
 */
export type ChainStep<TState> = (
  state: TState,
  context: ChainStepContext,
) => TState | Partial<TState> | Promise<TState | Partial<TState>>;

/** Options for {@link Chain.run}. */
export interface ChainRunOptions {
  /** Cancels the run: no further steps start after it fires. */
  signal?: AbortSignal | undefined;
}

/** Timing record for one executed step. */
export interface ChainStepTrace {
  name: string;
  index: number;
  durationMs: number;
}

/** Result of a chain run: final state plus per-step timing. */
export interface ChainRunResult<TState> {
  state: TState;
  steps: readonly ChainStepTrace[];
}

/** How a chain step failure is surfaced. */
export type ChainErrorMode = "throw" | "continue";

/**
 * A named, ordered pipeline of steps over shared state.
 *
 * Steps are added with {@link Chain.step} (chainable) and executed with
 * {@link Chain.run}. The chain itself is reusable: `run()` never mutates the
 * registered steps, so the same instance can serve many runs.
 */
export class Chain<TState extends object = Record<string, unknown>> {
  private readonly steps: Array<{ name: string; step: ChainStep<TState> }> = [];
  private readonly chainName: string;
  private readonly errorMode: ChainErrorMode;

  constructor(options: { name?: string; onError?: ChainErrorMode } = {}) {
    this.chainName = options.name ?? "chain";
    this.errorMode = options.onError ?? "throw";
  }

  /** Registers a step. Names must be unique within a chain. */
  step(name: string, step: ChainStep<TState>): this {
    if (typeof name !== "string" || name.trim().length === 0) {
      throw new ValidationError("Chain step name must be a non-empty string.");
    }
    if (typeof step !== "function") {
      throw new ValidationError(`Chain step "${name}" must be a function.`);
    }
    if (this.steps.some((entry) => entry.name === name)) {
      throw new ValidationError(`Chain step "${name}" is already registered.`, { step: name });
    }
    this.steps.push({ name, step });
    return this;
  }

  /** Names of the registered steps, in execution order. */
  get stepNames(): readonly string[] {
    return this.steps.map((entry) => entry.name);
  }

  /** Number of registered steps. */
  get length(): number {
    return this.steps.length;
  }

  /**
   * Runs every step in order over `initialState`.
   *
   * - Each step receives the previous step's state and may return a partial,
   *   which is shallow-merged over it.
   * - The first failure stops the chain (unless constructed with
   *   `onError: "continue"`, which records the failure and keeps going).
   * - A fired `signal` prevents subsequent steps from starting; the step that
   *   was already running sees the signal via its context.
   */
  async run(initialState: TState, options: ChainRunOptions = {}): Promise<ChainRunResult<TState>> {
    if (typeof initialState !== "object" || initialState === null) {
      throw new ValidationError("Chain.run() requires an initial state object.");
    }
    if (this.steps.length === 0) {
      throw new ValidationError("Chain has no steps; add one with .step(name, fn).");
    }

    let state: TState = initialState;
    const traces: ChainStepTrace[] = [];
    const failures: Array<{ step: string; error: unknown }> = [];

    for (let index = 0; index < this.steps.length; index++) {
      if (isAborted(options.signal)) {
        throw new ExecutionCancelledError(`Chain "${this.chainName}" cancelled before step "${
          this.steps[index]!.name
        }".`);
      }
      const { name, step } = this.steps[index]!;
      const startedAt = Date.now();
      let next: unknown;
      let stepSucceeded = true;
      try {
        next = await step(state, {
          name,
          index,
          ...(options.signal !== undefined ? { signal: options.signal } : {}),
        });
      } catch (error) {
        stepSucceeded = false;
        if (this.errorMode === "continue" && !(error instanceof ExecutionCancelledError)) {
          failures.push({ step: name, error });
        } else if (error instanceof ExecutionCancelledError || isAborted(options.signal)) {
          throw error instanceof ExecutionCancelledError
            ? error
            : new ExecutionCancelledError(
                `Chain "${this.chainName}" cancelled during step "${name}".`,
              );
        } else {
          throw wrapStepError(this.chainName, name, index, error);
        }
      }
      // Merging happens outside the catch so a contract violation (step
      // returned a non-object) surfaces as ValidationError, not wrapped.
      if (stepSucceeded) {
        state = mergeState(name, state, next);
      }
      traces.push({ name, index, durationMs: Date.now() - startedAt });
    }

    if (failures.length > 0) {
      throw new ChainStepError(
        this.chainName,
        failures.map((failure) => failure.step).join(", "),
        {
          index: -1,
          details: { failedSteps: failures.map((failure) => failure.step) },
          cause: failures[0]!.error,
        },
      );
    }

    return { state, steps: traces };
  }
}

/** True when the signal exists and has fired (call-safe for undefined). */
function isAborted(signal: AbortSignal | undefined): boolean {
  return signal !== undefined && signal.aborted;
}

/** Merges a step's return value over the current state. */
function mergeState<TState>(stepName: string, state: TState, next: unknown): TState {
  if (typeof next !== "object" || next === null) {
    throw new ValidationError(
      `Chain step "${stepName}" must return state (an object), received ${
        next === null ? "null" : typeof next
      }.`,
      { step: stepName },
    );
  }
  return { ...state, ...next };
}

function wrapStepError(chainName: string, stepName: string, index: number, error: unknown): Error {
  if (error instanceof ChainStepError) return error;
  const message = error instanceof Error ? error.message : String(error);
  return new ChainStepError(chainName, stepName, {
    index,
    cause: error,
    details: { originalMessage: message.slice(0, 500) },
  });
}
