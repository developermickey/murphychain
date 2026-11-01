import type {
  ChatCompletion,
  ChatCompletionAssistantMessageParam,
  ChatCompletionChunk,
  ChatCompletionCreateParamsStreaming,
  ChatCompletionCreateParamsNonStreaming,
  ChatCompletionMessageParam,
  ChatCompletionNamedToolChoice,
  ChatCompletionSystemMessageParam,
  ChatCompletionTool,
  ChatCompletionToolChoiceOption,
  ChatCompletionToolMessageParam,
  ChatCompletionUserMessageParam,
} from "openai/resources/chat/completions/completions";
import type { CompletionUsage } from "openai/resources/completions";
import {
  ProviderError,
  ZERO_USAGE,
  type AIResponse,
  type FinishReason,
  type GenerateOptions,
  type Message,
  type NormalizedTool,
  type ToolCall,
  type ToolChoice,
  type TokenUsage,
} from "@murphychain/core";

const PROVIDER = "openai";

/** Maps a MurphyChain message to an OpenAI chat message param. */
export function toOpenAIMessage(message: Message): ChatCompletionMessageParam {
  switch (message.role) {
    case "system": {
      const param: ChatCompletionSystemMessageParam = {
        role: "system",
        content: message.content,
      };
      return param;
    }
    case "user": {
      const param: ChatCompletionUserMessageParam = {
        role: "user",
        content: message.content,
      };
      return param;
    }
    case "assistant": {
      const param: ChatCompletionAssistantMessageParam = {
        role: "assistant",
        content: message.content,
      };
      if (message.toolCalls !== undefined && message.toolCalls.length > 0) {
        param.tool_calls = message.toolCalls.map((call) => ({
          id: call.id,
          type: "function" as const,
          function: { name: call.name, arguments: JSON.stringify(call.arguments) },
        }));
      }
      return param;
    }
    case "tool": {
      if (message.toolCallId === undefined || message.name === undefined) {
        throw new ProviderError(
          PROVIDER,
          'Messages with role "tool" require "toolCallId" and "name".',
          { code: "PROVIDER_MAPPING_ERROR", retryable: false },
        );
      }
      const param: ChatCompletionToolMessageParam = {
        role: "tool",
        tool_call_id: message.toolCallId,
        content: message.content,
      };
      return param;
    }
  }
}

/** Maps normalized tools to OpenAI function tools. */
export function toOpenAITools(tools: readonly NormalizedTool[]): ChatCompletionTool[] {
  return tools.map((tool) => ({
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.jsonSchema,
    },
  }));
}

/** Maps a tool choice to OpenAI's tool_choice value. */
export function toOpenAIToolChoice(choice: ToolChoice): ChatCompletionToolChoiceOption {
  if (typeof choice === "string") return choice;
  const named: ChatCompletionNamedToolChoice = {
    type: "function",
    function: { name: choice.name },
  };
  return named;
}

/** Builds non-streaming chat params from engine options. */
export function buildNonStreamingParams(
  options: GenerateOptions,
  model: string,
): ChatCompletionCreateParamsNonStreaming {
  return {
    ...buildBaseParams(options, model),
    stream: false,
  };
}

/** Builds streaming chat params (usage is requested so the terminal chunk carries it). */
export function buildStreamingParams(
  options: GenerateOptions,
  model: string,
): ChatCompletionCreateParamsStreaming {
  return {
    ...buildBaseParams(options, model),
    stream: true,
    stream_options: { include_usage: true },
  };
}

/** Shared (stream-agnostic) request parameters. */
function buildBaseParams(
  options: GenerateOptions,
  model: string,
): Omit<ChatCompletionCreateParamsNonStreaming, "stream"> {
  const params: Omit<ChatCompletionCreateParamsNonStreaming, "stream"> = {
    model,
    messages: options.messages.map(toOpenAIMessage),
  };
  if (options.temperature !== undefined) params.temperature = options.temperature;
  // max_tokens is deprecated in favor of max_completion_tokens.
  if (options.maxTokens !== undefined) params.max_completion_tokens = options.maxTokens;
  if (options.topP !== undefined) params.top_p = options.topP;
  if (options.stop !== undefined) params.stop = options.stop;
  if (options.tools !== undefined && options.tools.length > 0) {
    params.tools = toOpenAITools(options.tools);
    if (options.toolChoice !== undefined) params.tool_choice = toOpenAIToolChoice(options.toolChoice);
  }
  return params;
}

/** Maps OpenAI finish reasons to the unified FinishReason. */
export function mapFinishReason(
  reason: "stop" | "length" | "tool_calls" | "content_filter" | "function_call" | null,
): FinishReason {
  switch (reason) {
    case "stop":
      return "stop";
    case "length":
      return "length";
    case "tool_calls":
    case "function_call":
      return "tool_calls";
    case "content_filter":
      return "content_filter";
    case null:
      return "unknown";
  }
}

/** Maps OpenAI usage to the unified TokenUsage (missing usage → zeros). */
export function mapUsage(usage: CompletionUsage | null | undefined): TokenUsage {
  if (usage == null) return ZERO_USAGE;
  return {
    promptTokens: usage.prompt_tokens ?? 0,
    completionTokens: usage.completion_tokens ?? 0,
    totalTokens: usage.total_tokens ?? 0,
  };
}

/** Parses tool-call arguments JSON defensively (invalid JSON → empty object). */
export function parseToolArguments(raw: string): Record<string, unknown> {
  if (raw.length === 0) return {};
  try {
    const parsed: unknown = JSON.parse(raw);
    if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    return {};
  } catch {
    return {};
  }
}

/** Maps OpenAI tool calls (message-level) to unified ToolCall[]. */
export function mapToolCalls(
  toolCalls: ChatCompletion["choices"][number]["message"]["tool_calls"],
): ToolCall[] | undefined {
  if (toolCalls === undefined || toolCalls.length === 0) return undefined;
  const functionCalls = toolCalls.filter(
    (call): call is Extract<typeof call, { type: "function" }> => call.type === "function",
  );
  if (functionCalls.length === 0) return undefined;
  return functionCalls.map((call) => ({
    id: call.id,
    name: call.function.name,
    arguments: parseToolArguments(call.function.arguments),
  }));
}

/** Assembles the unified AIResponse from a non-streaming completion. */
export function buildAIResponse(
  completion: ChatCompletion,
  requestedModel: string,
  includeRaw: boolean,
  raw: unknown,
): AIResponse {
  const choice = completion.choices[0];
  if (choice === undefined) {
    throw new ProviderError(PROVIDER, "OpenAI returned a completion with no choices.", {
      code: "PROVIDER_MALFORMED_RESPONSE",
      retryable: false,
    });
  }
  const text = choice.message.content ?? "";
  const toolCalls = mapToolCalls(choice.message.tool_calls);
  const response: AIResponse = {
    text,
    provider: PROVIDER,
    model: completion.model.length > 0 ? completion.model : requestedModel,
    message: {
      role: "assistant",
      content: text,
      ...(toolCalls !== undefined ? { toolCalls } : {}),
    },
    finishReason: mapFinishReason(choice.finish_reason),
    usage: mapUsage(completion.usage),
    latencyMs: 0, // stamped by the engine.
  };
  if (includeRaw) response.raw = raw;
  return response;
}

/**
 * Accumulates streamed tool-call deltas (arguments arrive fragmented across
 * chunks and must be concatenated before JSON parsing).
 */
export class ToolCallAccumulator {
  private readonly entries = new Map<number, { id: string; name: string; args: string }>();

  add(delta: ChatCompletionChunk.Choice.Delta.ToolCall[]): void {
    for (const part of delta) {
      const entry = this.entries.get(part.index) ?? { id: "", name: "", args: "" };
      if (part.id !== undefined) entry.id = part.id;
      if (part.function?.name !== undefined) entry.name = part.function.name;
      if (part.function?.arguments !== undefined) entry.args += part.function.arguments;
      this.entries.set(part.index, entry);
    }
  }

  /** Returns accumulated tool calls (stable order by delta index). */
  toToolCalls(): ToolCall[] | undefined {
    if (this.entries.size === 0) return undefined;
    const sorted = [...this.entries.entries()].sort(([a], [b]) => a - b);
    return sorted.map(([index, entry], position) => ({
      // OpenAI always provides an id for function tool calls; synthesize a
      // stable placeholder if a proxy strips it, so correlation still works.
      id: entry.id.length > 0 ? entry.id : `call_${index}_${position}`,
      name: entry.name,
      arguments: parseToolArguments(entry.args),
    }));
  }
}
