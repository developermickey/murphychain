import { ExecutionCancelledError, ExecutionTimeoutError } from "./Errors";

/**
 * Per-request execution context: identity, deadline, cancellation, and
 * metadata propagation.
 *
 * A single `AbortController` backs each request. The caller's signal is
 * linked to it, and a deadline timer aborts it on timeout. The abort *reason*
 * is recorded so the engine can distinguish timeouts from user cancellation.
 */

export type AbortReason = "timeout" | "user";

export interface ExecutionContextOptions {
  provider: string;
  model: string;
  /** Deadline in milliseconds. `0` or omitted disables the deadline. */
  timeoutMs?: number | undefined;
  /** Caller-provided cancellation signal. */
  signal?: AbortSignal | undefined;
  /** Merged engine + call metadata (frozen copy). */
  metadata?: Readonly<Record<string, unknown>> | undefined;
}

export class ExecutionContext {
  readonly requestId: string;
  readonly provider: string;
  readonly model: string;
  readonly startedAt: number;
  readonly metadata: Readonly<Record<string, unknown>>;

  private readonly controller = new AbortController();
  private readonly timeoutMs: number;
  private timeoutHandle: ReturnType<typeof setTimeout> | undefined;
  private deadlineAt: number | undefined;
  private externalSignal: AbortSignal | undefined;
  private externalListener: (() => void) | undefined;
  private abortedReason: AbortReason | undefined;
  private disposed = false;

  constructor(options: ExecutionContextOptions) {
    this.requestId = `req_${crypto.randomUUID()}`;
    this.provider = options.provider;
    this.model = options.model;
    this.startedAt = Date.now();
    this.timeoutMs = options.timeoutMs ?? 0;
    this.metadata = Object.freeze({ ...(options.metadata ?? {}) });

    if (options.signal !== undefined) this.linkExternal(options.signal);
    if (this.timeoutMs > 0) this.startDeadline();
  }

  /** The signal providers must honour for cancellation. */
  get signal(): AbortSignal {
    return this.controller.signal;
  }

  get aborted(): boolean {
    return this.controller.signal.aborted;
  }

  /** Why the request was aborted, if it was. */
  get reason(): AbortReason | undefined {
    return this.abortedReason;
  }

  /** Milliseconds elapsed since the context was created. */
  get elapsedMs(): number {
    return Date.now() - this.startedAt;
  }

  /** Absolute timestamp of the deadline, when one is active. */
  get deadline(): number | undefined {
    return this.deadlineAt;
  }

  /** Remaining time before the deadline, or `undefined` when disabled. */
  remainingMs(): number | undefined {
    if (this.deadlineAt === undefined) return undefined;
    return Math.max(0, this.deadlineAt - Date.now());
  }

  /** Aborts the request as a user cancellation (idempotent). */
  cancel(): void {
    this.abort("user");
  }

  /**
   * Aborts with a timeout reason if not already aborted (idempotent).
   * Used by the deadline timer.
   */
  abort(reason: AbortReason): void {
    if (this.controller.signal.aborted) return;
    this.abortedReason = reason;
    this.controller.abort(reason);
    // One-shot cleanup: no listener/timer may outlive the abort.
    this.clearDeadline();
    this.unlinkExternal();
  }

  /** Throws the abort-mapped error when the request has been aborted. */
  throwIfAborted(): void {
    if (this.controller.signal.aborted) throw this.abortError();
  }

  /** The error corresponding to the current abort reason. */
  abortError(): ExecutionTimeoutError | ExecutionCancelledError {
    if (this.abortedReason === "timeout") {
      return new ExecutionTimeoutError(`Request timed out after ${this.timeoutMs}ms.`, {
        timeoutMs: this.timeoutMs,
      });
    }
    return new ExecutionCancelledError("Request cancelled by caller.", { reason: "user" });
  }

  /**
   * Stops the deadline timer without aborting. Used by streaming once the
   * first chunk arrives: time-to-first-chunk is bounded, but a long
   * generation must not be killed mid-output.
   */
  clearDeadline(): void {
    if (this.timeoutHandle !== undefined) {
      clearTimeout(this.timeoutHandle);
      this.timeoutHandle = undefined;
    }
    this.deadlineAt = undefined;
  }

  /** Releases all timers and listeners. Safe to call multiple times. */
  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.clearDeadline();
    this.unlinkExternal();
  }

  private startDeadline(): void {
    if (this.disposed || this.deadlineAt !== undefined) return;
    this.deadlineAt = Date.now() + this.timeoutMs;
    this.timeoutHandle = setTimeout(() => {
      this.abort("timeout");
    }, this.timeoutMs);
  }

  private linkExternal(signal: AbortSignal): void {
    if (signal.aborted) {
      this.abort("user");
      return;
    }
    const onAbort = (): void => {
      this.abort("user");
    };
    signal.addEventListener("abort", onAbort, { once: true });
    this.externalSignal = signal;
    this.externalListener = onAbort;
  }

  private unlinkExternal(): void {
    if (this.externalSignal !== undefined && this.externalListener !== undefined) {
      this.externalSignal.removeEventListener("abort", this.externalListener);
    }
    this.externalSignal = undefined;
    this.externalListener = undefined;
  }
}
