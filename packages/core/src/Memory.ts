import { AI } from "./AI";
import { ValidationError, truncateMessage } from "./Errors";
import type { Message } from "./types";

/**
 * Memory: the conversation history an agent carries across runs.
 *
 * Memory is deliberately a small, honest abstraction — a bounded sequence of
 * messages with a retention policy. Two implementations ship:
 *
 * - {@link BufferMemory}: keeps the most recent N messages (FIFO eviction).
 * - {@link SummarizingMemory}: keeps recent messages verbatim and folds older
 *   ones into a rolling summary via one `generate()` call.
 *
 * Extend this class to build custom retention policies; the agent validates
 * instances with `instanceof Memory`.
 *
 * Memory stores message content in-process only; it is never sent anywhere
 * except back to the configured provider on the next run.
 */
export abstract class Memory {
  /** The retained messages, oldest first. */
  abstract get messages(): readonly Message[];
  /** Appends messages, applying the retention policy. May summarize (async). */
  abstract remember(messages: readonly Message[]): Promise<void>;
  /** Drops all retained messages (including any rolling summary). */
  abstract clear(): void;
}

/**
 * The single place retention is enforced, shared by both implementations:
 * an assistant message that requested tool calls and the `role: "tool"`
 * messages that answer it form one atomic unit and are always evicted
 * together (splitting them yields conversations most providers reject).
 *
 * Returns the retained suffix of `messages` (same object when nothing was
 * evicted, so the common no-op path allocates nothing).
 */
export function evictOldestUnits(messages: Message[], maxMessages: number): Message[] {
  let start = 0;
  while (start < messages.length && messages.length - start > maxMessages) {
    start += 1;
    // Never leave orphaned tool results at the front: skip past every tool
    // message that answers the unit we just evicted.
    while (start < messages.length && messages[start]!.role === "tool") {
      start += 1;
    }
  }
  return start === 0 ? messages : messages.slice(start);
}

/** Merges a shallow copy (tool-call arrays copied, arguments shared). */
function cloneMessages(messages: readonly Message[]): Message[] {
  return messages.map((message) =>
    message.toolCalls !== undefined
      ? { ...message, toolCalls: message.toolCalls.map((call) => ({ ...call })) }
      : { ...message },
  );
}

export interface BufferMemoryOptions {
  /**
   * Maximum messages retained. Every message counts, including tool results.
   * Default: 100.
   */
  maxMessages?: number | undefined;
}

/** Bounded FIFO memory: remembers the most recent messages, evicts the oldest. */
export class BufferMemory extends Memory {
  private buffer: Message[] = [];
  private readonly maxMessages: number;

  constructor(options: BufferMemoryOptions = {}) {
    super();
    const maxMessages = options.maxMessages ?? 100;
    if (!Number.isInteger(maxMessages) || maxMessages < 1) {
      throw new ValidationError('"maxMessages" must be an integer >= 1.', {
        field: "maxMessages",
      });
    }
    this.maxMessages = maxMessages;
  }

  get messages(): readonly Message[] {
    return this.buffer;
  }

  async remember(messages: readonly Message[]): Promise<void> {
    if (messages.length === 0) return;
    this.buffer = evictOldestUnits([...this.buffer, ...cloneMessages(messages)], this.maxMessages);
  }

  clear(): void {
    this.buffer = [];
  }
}

export interface SummarizingMemoryOptions {
  /** Engine used for the summarization call (same provider as the agent). */
  ai: AI;
  /** Messages kept verbatim, never summarized. Must be < maxMessages. Default: 10. */
  keepRecent?: number | undefined;
  /** Total messages retained (summary counts as one). Must be > keepRecent. Default: 40. */
  maxMessages?: number | undefined;
}

const SUMMARIZE_SYSTEM_PROMPT =
  "You maintain a running summary of a conversation so it can continue within a fixed context budget. " +
  "Reply with only the updated summary — no preamble, no headings.";

/**
 * Memory with a rolling summary: recent messages stay verbatim; when the
 * buffer overflows, the oldest messages are folded into a summary by a single
 * `generate()` call and replaced by one summary message.
 *
 * The summary is stored as a `user`-role message (bracketed and labeled) —
 * every provider accepts mid-conversation user messages, whereas mid-run
 * system messages have provider-specific semantics.
 */
export class SummarizingMemory extends Memory {
  private buffer: Message[] = [];
  private summary: string | undefined;

  private readonly ai: AI;
  private readonly keepRecent: number;
  private readonly maxMessages: number;

  constructor(options: SummarizingMemoryOptions) {
    super();
    if (!(options.ai instanceof AI)) {
      throw new ValidationError('"ai" must be an AI instance.', { field: "ai" });
    }
    const keepRecent = options.keepRecent ?? 10;
    const maxMessages = options.maxMessages ?? 40;
    if (!Number.isInteger(keepRecent) || keepRecent < 1) {
      throw new ValidationError('"keepRecent" must be an integer >= 1.', { field: "keepRecent" });
    }
    if (!Number.isInteger(maxMessages) || maxMessages <= keepRecent) {
      throw new ValidationError('"maxMessages" must be an integer > keepRecent.', {
        field: "maxMessages",
      });
    }
    this.ai = options.ai;
    this.keepRecent = keepRecent;
    this.maxMessages = maxMessages;
  }

  get messages(): readonly Message[] {
    const head: Message[] =
      this.summary !== undefined
        ? [{ role: "user", content: `[Conversation summary so far]\n${this.summary}` }]
        : [];
    return [...head, ...this.buffer];
  }

  async remember(messages: readonly Message[]): Promise<void> {
    if (messages.length === 0) return;
    this.buffer = [...this.buffer, ...cloneMessages(messages)];

    // One summarization call per remember() at most: fold everything outside
    // the keep-recent window (evictOldestUnits may retain extra messages to
    // avoid splitting a tool-call unit), then the buffer fits the bound.
    while (this.buffer.length > this.maxMessages) {
      const retained = evictOldestUnits(this.buffer, this.keepRecent);
      const folded = this.buffer.slice(0, this.buffer.length - retained.length);
      // Unreachable while keepRecent < maxMessages, but a library must not
      // spin forever on state it did not anticipate.
      if (folded.length === 0) break;
      this.summary = await this.summarize(this.summary, folded);
      this.buffer = retained;
    }
  }

  clear(): void {
    this.buffer = [];
    this.summary = undefined;
  }

  /** One summarization request: existing summary + folded messages → new summary. */
  private async summarize(
    previous: string | undefined,
    folded: readonly Message[],
  ): Promise<string> {
    const transcript = folded
      .map((message) => `${message.role}: ${truncateMessage(message.content, 2_000)}`)
      .join("\n");
    const userContent = [
      previous !== undefined ? `Existing summary:\n${previous}` : undefined,
      `New messages to fold into the summary:\n${transcript}`,
      "Write the updated summary now.",
    ]
      .filter((line): line is string => line !== undefined)
      .join("\n\n");

    const response = await this.ai.generate({
      system: SUMMARIZE_SYSTEM_PROMPT,
      messages: [{ role: "user", content: userContent }],
    });
    const summary = response.text.trim();
    if (summary.length === 0) {
      throw new ValidationError("Summarization returned an empty summary.", {
        field: "memory",
      });
    }
    // Bound the summary itself so a rambling model cannot grow it forever.
    return truncateMessage(summary, 4_000);
  }
}
