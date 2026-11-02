import type { MurphyChainError } from "./Errors";
import type { FinishReason, TokenUsage } from "./types";

/**
 * Typed, content-free observability events.
 *
 * Events intentionally never carry prompt or completion text: listeners may
 * forward to log aggregators, and logs must not become a side channel for
 * sensitive data. Counts and lengths are sufficient for tracing throughput,
 * latency, and token usage.
 */

export interface RequestStartEvent {
  requestId: string;
  provider: string;
  model: string;
  /** Number of messages sent (including system). */
  messageCount: number;
  /** Total character count of all message contents. */
  totalChars: number;
  /** User-defined context propagated from `AIConfig.metadata` / call options. */
  metadata: Readonly<Record<string, unknown>>;
}

export interface RequestSuccessEvent {
  requestId: string;
  provider: string;
  model: string;
  /** Wall-clock latency including retries (and, for streams, full iteration). */
  latencyMs: number;
  usage: TokenUsage;
  finishReason: FinishReason;
  /** 1 for non-streaming requests, number of chunks for streams. */
  chunkCount: number;
}

export interface RequestErrorEvent {
  requestId: string;
  provider: string;
  model: string;
  latencyMs: number;
  error: MurphyChainError;
}

export interface RetryEvent {
  requestId: string;
  /** 1-based number of the attempt that is about to run. */
  attempt: number;
  delayMs: number;
  error: MurphyChainError;
}

export interface ChunkEvent {
  requestId: string;
  index: number;
  textLength: number;
}

export interface AbortEvent {
  requestId: string;
  reason: "timeout" | "user";
}

/** Map of all events emitted by the engine. */
export type AIEventMap = {
  "request:start": RequestStartEvent;
  "request:success": RequestSuccessEvent;
  "request:error": RequestErrorEvent;
  retry: RetryEvent;
  chunk: ChunkEvent;
  abort: AbortEvent;
};

export type AIEventName = keyof AIEventMap;
/** Listener for a specific event of an event map. */
export type EventListener<M extends Record<string, unknown>, K extends keyof M> = (
  event: M[K],
) => void;
/** Listener for a specific engine event. */
export type AIEventListener<K extends AIEventName> = EventListener<AIEventMap, K>;

type AnyListener<M> = (event: M[keyof M]) => void;

/**
 * Minimal typed event bus.
 *
 * Listener exceptions are isolated: a throwing listener can never break the
 * request that triggered the event. Listeners are invoked synchronously and
 * should not perform long-running work.
 */
export class EventBus<M extends Record<string, unknown>> {
  private readonly listeners = new Map<keyof M, Set<AnyListener<M>>>();

  /** Subscribes to an event. Returns an unsubscribe function. */
  on<K extends keyof M>(event: K, listener: EventListener<M, K>): () => void {
    return this.add(event, listener as AnyListener<M>, false);
  }

  /** Subscribes to the next occurrence of an event. Returns an unsubscribe function. */
  once<K extends keyof M>(event: K, listener: EventListener<M, K>): () => void {
    return this.add(event, listener as AnyListener<M>, true);
  }

  /** Removes a previously registered listener. */
  off<K extends keyof M>(event: K, listener: EventListener<M, K>): void {
    this.listeners.get(event)?.delete(listener as AnyListener<M>);
  }

  /** Emits an event to all listeners. Listener errors are swallowed. */
  emit<K extends keyof M>(event: K, payload: M[K]): void {
    const set = this.listeners.get(event);
    if (set === undefined || set.size === 0) return;
    // Snapshot so listeners may unsubscribe during dispatch.
    for (const listener of [...set]) {
      try {
        (listener as (p: M[K]) => void)(payload);
      } catch {
        // Observability must never affect request execution.
      }
    }
  }

  removeAllListeners(): void {
    this.listeners.clear();
  }

  private add(event: keyof M, listener: AnyListener<M>, once: boolean): () => void {
    let set = this.listeners.get(event);
    if (set === undefined) {
      set = new Set();
      this.listeners.set(event, set);
    }
    const wrapped: AnyListener<M> = once
      ? (payload) => {
          set?.delete(wrapped);
          listener(payload);
        }
      : listener;
    set.add(wrapped);
    return () => {
      set?.delete(wrapped);
    };
  }
}
