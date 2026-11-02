import { z } from "zod";
import { AI } from "./AI";
import type { CallOptions } from "./AI";
import { EventBus } from "./Events";
import type { EventListener } from "./Events";
import { ExecutionCancelledError, ValidationError } from "./Errors";
import { Memory } from "./Memory";
import { Tool, executeToolCalls } from "./Tool";
import type { AnyTool, ExecutedToolCall } from "./Tool";
import { ZERO_USAGE } from "./types";
import type {
  AIResponse,
  FinishReason,
  Message,
  TokenUsage,
  ToolCall,
  ToolDefinition,
} from "./types";
import { messageSchema, parseOrThrow } from "./validation";

/**
 * Agent runtime: the bounded tool-calling loop.
 *
 * An agent owns a configured {@link AI} instance, a set of {@link Tool}s, an
 * optional {@link Memory}, and a hard step budget. Each `run()`:
 *
 * 1. seeds a conversation (system prompt → remembered history → input),
 * 2. asks the model (with the tools attached),
 * 3. executes any tool calls the model requested and feeds the results back,
 * 4. repeats until the model answers in text or the step budget is spent.
 *
 * The loop is **bounded by construction**: `maxSteps` caps model calls, the
 * caller's `AbortSignal` is checked between steps and forwarded to every
 * request and tool, and a tool-call round on the final step is *not*
 * executed (its results would have no consumer).
 *
 * ```ts
 * const agent = new Agent({
 *   name: "researcher",
 *   system: "You research topics using the tools provided.",
 *   tools: [search],
 *   ai: new AI({ provider: "openai", model: "gpt-4.1", apiKey }),
 *   maxSteps: 8,
 * });
 *
 * const result = await agent.run("What changed in TypeScript 6?");
 * result.response.text;  // final answer
 * result.steps;          // per-step trace (tool calls, timings)
 * ```
 */

/** Default model calls per run when `maxSteps` is not configured. */
export const DEFAULT_AGENT_MAX_STEPS = 10;

// ---------------------------------------------------------------------------
// Events (content-free: names, counts, timings — never message content)
// ---------------------------------------------------------------------------

/** Emitted after every model call of a run. */
export interface AgentStepEvent {
  agent: string;
  /** Zero-based step index. */
  index: number;
  /** Number of tool calls the model requested in this step (0 = final answer). */
  toolCalls: number;
  finishReason: FinishReason;
  /** Latency of this model call, including engine retries. */
  latencyMs: number;
  usage: TokenUsage;
}

/** Emitted after each individual tool execution. */
export interface AgentToolEvent {
  agent: string;
  /** Step index this call belongs to. */
  step: number;
  /** Tool name (developer-defined, never model content). */
  name: string;
  /** False when the tool was unknown, arguments were invalid, or execute threw. */
  ok: boolean;
  durationMs: number;
}

/** Emitted once when a run completes (successfully or by step exhaustion). */
export interface AgentFinishEvent {
  agent: string;
  /** Number of model calls made. */
  steps: number;
  stopped: "completed" | "max_steps";
  usage: TokenUsage;
  elapsedMs: number;
}

/** Map of all events emitted by an agent. */
export type AgentEventMap = {
  step: AgentStepEvent;
  tool: AgentToolEvent;
  finish: AgentFinishEvent;
};

export type AgentEventName = keyof AgentEventMap;
export type AgentEventListener<K extends AgentEventName> = EventListener<AgentEventMap, K>;

// ---------------------------------------------------------------------------
// Configuration & results
// ---------------------------------------------------------------------------

/** Configuration of an {@link Agent}. */
export interface AgentConfig {
  /** Agent name used in errors and events. Default: `"agent"`. */
  name?: string | undefined;
  /** The engine: owns provider, model, retry, timeout, and event config. */
  ai: AI;
  /** System prompt prepended to every model call (not stored in memory). */
  system?: string | undefined;
  /** Tools the model may request. Must be uniquely named. */
  tools?: readonly AnyTool[] | undefined;
  /** Hard cap on model calls per run. Default: {@link DEFAULT_AGENT_MAX_STEPS}. */
  maxSteps?: number | undefined;
  /** Conversation history carried across runs. Omit for stateless agents. */
  memory?: Memory | undefined;
  /**
   * What happens when a tool call fails (unknown tool, invalid arguments,
   * execute threw):
   *
   * - `"return"` (default): the failure is serialized as an `{"error": ...}`
   *   tool result so the model can recover on its own.
   * - `"throw"`: the error propagates and the run fails fast.
   */
  onToolError?: "return" | "throw" | undefined;
}

/** Per-run execution controls. */
export interface AgentRunOptions {
  /** Cancellation signal: checked between steps, forwarded to requests/tools. */
  signal?: AbortSignal | undefined;
  /** Deadline in ms **per model call** (not for the whole run). */
  timeoutMs?: number | undefined;
  /** Context propagated to the underlying `request:start` events. */
  metadata?: Record<string, unknown> | undefined;
  /** Override the configured step budget for this run. */
  maxSteps?: number | undefined;
}

/** Trace of one model call within a run. */
export interface AgentStep {
  /** Zero-based step index. */
  index: number;
  /** The model's response for this step. */
  response: AIResponse;
  /** Tool calls requested in this step (unexecuted on the final step). */
  toolCalls: ToolCall[];
  /** `role: "tool"` results appended after this step (empty on final step). */
  toolMessages: Message[];
  /** Wall-clock duration of this step (model call + tool execution). */
  durationMs: number;
}

/** Result of an agent run. */
export interface AgentResult {
  /** The model's final response (on `max_steps`, the last, unanswered one). */
  response: AIResponse;
  /** The full conversation of this run: system + memory + input + exchanges. */
  messages: Message[];
  /** One trace entry per model call, in order. */
  steps: AgentStep[];
  /** Token usage summed across all model calls. */
  usage: TokenUsage;
  /** Why the loop ended. */
  stopped: "completed" | "max_steps";
  /** Wall-clock duration of the whole run. */
  elapsedMs: number;
}

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

/**
 * A tool-calling agent: bounded loop, optional memory, typed events.
 *
 * An agent instance is **single-flight**: concurrent `run()` calls on the
 * same instance throw (memory and the step budget are shared state). Create
 * one agent per concurrent task.
 */
export class Agent {
  readonly name: string;
  readonly tools: readonly AnyTool[];
  readonly maxSteps: number;

  private readonly ai: AI;
  private readonly system: string | undefined;
  private readonly memory: Memory | undefined;
  private readonly onToolError: "return" | "throw";
  private readonly toolDefinitions: ToolDefinition[];
  private readonly bus = new EventBus<AgentEventMap>();
  private running = false;

  constructor(config: AgentConfig) {
    if (typeof config !== "object" || config === null) {
      throw new ValidationError("Agent requires a configuration object.");
    }
    if (!(config.ai instanceof AI)) {
      throw new ValidationError('"ai" must be an AI instance (new AI({ provider, model })).', {
        field: "ai",
      });
    }
    this.ai = config.ai;

    if (config.name !== undefined) {
      if (typeof config.name !== "string" || config.name.trim().length === 0) {
        throw new ValidationError('"name" must be a non-empty string.', { field: "name" });
      }
      this.name = config.name.trim();
    } else {
      this.name = "agent";
    }

    if (config.system !== undefined) {
      if (typeof config.system !== "string") {
        throw new ValidationError('"system" must be a string.', { field: "system" });
      }
      this.system = config.system;
    }

    const tools = config.tools ?? [];
    if (!Array.isArray(tools)) {
      throw new ValidationError('"tools" must be an array of Tool instances.', { field: "tools" });
    }
    const seen = new Set<string>();
    for (const tool of tools) {
      if (!(tool instanceof Tool)) {
        throw new ValidationError(
          `Agent tools must be Tool instances created with Tool.define (received ${typeof tool}).`,
          { field: "tools" },
        );
      }
      if (seen.has(tool.name)) {
        throw new ValidationError(`Duplicate tool name "${tool.name}" in agent tools.`, {
          field: "tools",
        });
      }
      seen.add(tool.name);
    }
    this.tools = [...tools];
    // Zod → JSON Schema conversion happens once per tool (cached in
    // `tool.normalized`); every step reuses the plain JSON Schema instead of
    // re-converting on each model call.
    this.toolDefinitions = this.tools.map((tool) => ({
      name: tool.name,
      description: tool.description,
      parameters: tool.normalized.jsonSchema,
    }));

    this.maxSteps = config.maxSteps ?? DEFAULT_AGENT_MAX_STEPS;
    assertPositiveInteger(this.maxSteps, "maxSteps");

    if (config.memory !== undefined && !(config.memory instanceof Memory)) {
      throw new ValidationError('"memory" must be a Memory instance (BufferMemory, …).', {
        field: "memory",
      });
    }
    this.memory = config.memory;

    if (config.onToolError !== undefined && config.onToolError !== "return" && config.onToolError !== "throw") {
      throw new ValidationError('"onToolError" must be "return" or "throw".', {
        field: "onToolError",
      });
    }
    this.onToolError = config.onToolError ?? "return";
  }

  /** Subscribes to an agent event. Returns an unsubscribe function. */
  on<K extends AgentEventName>(event: K, listener: AgentEventListener<K>): () => void {
    return this.bus.on(event, listener);
  }

  /** Subscribes to the next occurrence of an agent event. */
  once<K extends AgentEventName>(event: K, listener: AgentEventListener<K>): () => void {
    return this.bus.once(event, listener);
  }

  /** Removes a previously registered listener. */
  off<K extends AgentEventName>(event: K, listener: AgentEventListener<K>): void {
    this.bus.off(event, listener);
  }

  /**
   * Runs the agent loop to completion (or step exhaustion).
   *
   * Throws `ValidationError` for bad input, `ExecutionCancelledError` when
   * the signal aborts (between steps or inside a request/tool), and whatever
   * the engine/tools throw otherwise. Memory is updated **only after a run
   * that made progress** — a failed run leaves history untouched.
   */
  async run(input: string | Message[], options: AgentRunOptions = {}): Promise<AgentResult> {
    if (typeof options !== "object" || options === null) {
      throw new ValidationError("Agent run options must be an object.");
    }
    if (this.running) {
      throw new ValidationError(
        `Agent "${this.name}" is already running; create one agent instance per concurrent task.`,
        { agent: this.name },
      );
    }
    const maxSteps = options.maxSteps ?? this.maxSteps;
    assertPositiveInteger(maxSteps, "maxSteps");
    const inputMessages = parseAgentInput(input, this.name);

    this.running = true;
    try {
      return await this.execute(inputMessages, maxSteps, options);
    } finally {
      this.running = false;
    }
  }

  // -------------------------------------------------------------------------
  // Loop
  // -------------------------------------------------------------------------

  private async execute(
    inputMessages: Message[],
    maxSteps: number,
    options: AgentRunOptions,
  ): Promise<AgentResult> {
    const conversation = this.buildSeed(inputMessages);
    const steps: AgentStep[] = [];
    let usage: TokenUsage = { ...ZERO_USAGE };
    const startedAt = Date.now();
    const callOptions = this.buildCallOptions(options);

    let index = 0;
    for (;;) {
      if (options.signal?.aborted === true) {
        throw new ExecutionCancelledError(
          `Agent "${this.name}" cancelled before step ${index}.`,
          { agent: this.name, step: index },
        );
      }
      const stepStartedAt = Date.now();
      const response = await this.ai.generate(
        {
          messages: conversation,
          ...(this.tools.length > 0 ? { tools: this.toolDefinitions } : {}),
        },
        callOptions,
      );
      usage = addUsage(usage, response.usage);
      conversation.push(response.message);
      const calls = response.message.toolCalls ?? [];

      // Terminal: the model answered in text.
      if (calls.length === 0) {
        steps.push({
          index,
          response,
          toolCalls: [],
          toolMessages: [],
          durationMs: Date.now() - stepStartedAt,
        });
        this.emitStep(index, response, 0, stepStartedAt);
        await this.persist(conversation);
        return this.finish(response, conversation, steps, usage, "completed", startedAt);
      }

      // Terminal: budget spent — do not execute calls nothing would consume.
      if (index === maxSteps - 1) {
        steps.push({
          index,
          response,
          toolCalls: calls,
          toolMessages: [],
          durationMs: Date.now() - stepStartedAt,
        });
        this.emitStep(index, response, calls.length, stepStartedAt);
        // The trailing assistant message requests tools that were never
        // answered; persisting it would produce a conversation most
        // providers reject on the next run. Store everything before it.
        await this.persist(conversation.slice(0, -1));
        return this.finish(response, conversation, steps, usage, "max_steps", startedAt);
      }

      const toolMessages = await executeToolCalls(this.tools, calls, {
        ...(options.signal !== undefined ? { signal: options.signal } : {}),
        onError: this.onToolError,
        onCallComplete: (executed) => this.emitTool(index, executed),
      });
      conversation.push(...toolMessages);
      steps.push({
        index,
        response,
        toolCalls: calls,
        toolMessages,
        durationMs: Date.now() - stepStartedAt,
      });
      this.emitStep(index, response, calls.length, stepStartedAt);
      index += 1;
    }
  }

  /** System prompt → remembered history → this run's input. */
  private buildSeed(inputMessages: Message[]): Message[] {
    const seed: Message[] = [];
    if (this.system !== undefined) {
      seed.push({ role: "system", content: this.system });
    }
    if (this.memory !== undefined) {
      seed.push(...this.memory.messages);
    }
    seed.push(...inputMessages);
    return seed;
  }

  private buildCallOptions(options: AgentRunOptions): CallOptions {
    return {
      ...(options.signal !== undefined ? { signal: options.signal } : {}),
      ...(options.timeoutMs !== undefined ? { timeoutMs: options.timeoutMs } : {}),
      ...(options.metadata !== undefined ? { metadata: options.metadata } : {}),
    };
  }

  private async persist(conversation: readonly Message[]): Promise<void> {
    if (this.memory === undefined) return;
    // The system prompt is configuration, not history: it is re-seeded from
    // `this.system` on every run and must not accumulate in memory.
    const first = conversation[0];
    const start =
      this.system !== undefined && first?.role === "system" && first.content === this.system
        ? 1
        : 0;
    await this.memory.remember(conversation.slice(start));
  }

  private finish(
    response: AIResponse,
    conversation: Message[],
    steps: AgentStep[],
    usage: TokenUsage,
    stopped: "completed" | "max_steps",
    startedAt: number,
  ): AgentResult {
    const elapsedMs = Date.now() - startedAt;
    this.bus.emit("finish", {
      agent: this.name,
      steps: steps.length,
      stopped,
      usage,
      elapsedMs,
    });
    return { response, messages: conversation, steps, usage, stopped, elapsedMs };
  }

  private emitStep(
    index: number,
    response: AIResponse,
    toolCalls: number,
    stepStartedAt: number,
  ): void {
    this.bus.emit("step", {
      agent: this.name,
      index,
      toolCalls,
      finishReason: response.finishReason,
      latencyMs: Date.now() - stepStartedAt,
      usage: response.usage,
    });
  }

  private emitTool(step: number, executed: ExecutedToolCall): void {
    this.bus.emit("tool", {
      agent: this.name,
      step,
      name: executed.call.name,
      ok: executed.ok,
      durationMs: executed.durationMs,
    });
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Validates run input: a non-empty prompt string or a valid message list. */
function parseAgentInput(input: string | Message[], agentName: string): Message[] {
  if (typeof input === "string") {
    if (input.length === 0) {
      throw new ValidationError(`Agent "${agentName}" received an empty prompt.`, {
        agent: agentName,
      });
    }
    return [{ role: "user", content: input }];
  }
  if (Array.isArray(input)) {
    return parseOrThrow(
      z.array(messageSchema).min(1),
      input,
      `Agent "${agentName}" received invalid input messages.`,
    );
  }
  throw new ValidationError(
    `Agent "${agentName}" expects a prompt string or an array of messages.`,
    { agent: agentName },
  );
}

function assertPositiveInteger(value: number, field: string): void {
  if (!Number.isInteger(value) || value < 1) {
    throw new ValidationError(`"${field}" must be an integer >= 1.`, { field });
  }
}

/** Adds token usage across steps. */
function addUsage(base: TokenUsage, add: TokenUsage): TokenUsage {
  return {
    promptTokens: base.promptTokens + add.promptTokens,
    completionTokens: base.completionTokens + add.completionTokens,
    totalTokens: base.totalTokens + add.totalTokens,
  };
}
