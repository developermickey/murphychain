import {
  FinishReason as GeminiFinishReason,
  FunctionCallingConfigMode,
  type Content,
  type FunctionCall,
  type FunctionDeclaration,
  type GenerateContentConfig,
  type GenerateContentResponse,
  type Part,
  type Tool,
  type ToolConfig,
} from "@google/genai";
import {
  ProviderContentFilterError,
  ProviderError,
  ZERO_USAGE,
  type FinishReason,
  type GenerateOptions,
  type Message,
  type NormalizedTool,
  type ToolCall,
  type ToolChoice,
  type TokenUsage,
} from "@murphychain/core";

const PROVIDER = "gemini";

/** Groups consecutive tool-result messages into one user-role Content. */
export function toGeminiContents(
  messages: readonly Message[],
): { contents: Content[]; systemInstruction: string | undefined } {
  const systemParts: string[] = [];
  const contents: Content[] = [];

  for (const message of messages) {
    if (message.role === "system") {
      systemParts.push(message.content);
      continue;
    }
    if (message.role === "tool") {
      const part: Part = {
        functionResponse: {
          ...(message.toolCallId !== undefined ? { id: message.toolCallId } : {}),
          name: message.name ?? "",
          response: toFunctionResponse(message.content),
        },
      };
      const last = contents[contents.length - 1];
      if (last !== undefined && last.role === "user" && isFunctionResponseOnly(last)) {
        last.parts?.push(part); // group consecutive results in one turn
      } else {
        contents.push({ role: "user", parts: [part] });
      }
      continue;
    }
    const parts: Part[] = [];
    if (message.content.length > 0) parts.push({ text: message.content });
    if (message.role === "assistant" && message.toolCalls !== undefined) {
      for (const call of message.toolCalls) {
        parts.push({
          functionCall: {
            ...(call.id.length > 0 ? { id: call.id } : {}),
            name: call.name,
            args: call.arguments,
          },
        });
      }
    }
    // Gemini requires at least one part per content.
    if (parts.length === 0) parts.push({ text: "" });
    contents.push({ role: message.role === "assistant" ? "model" : "user", parts });
  }

  return {
    contents,
    systemInstruction: systemParts.length > 0 ? systemParts.join("\n\n") : undefined,
  };
}

/**
 * Wraps a tool result string as a function response object. Parsed JSON
 * objects are passed through; anything else is wrapped in `{ output }`
 * (Gemini's documented convention for non-object results).
 */
function toFunctionResponse(content: string): Record<string, unknown> {
  try {
    const parsed: unknown = JSON.parse(content);
    if (typeof parsed === "object" && parsed !== null && !Array.isArray(parsed)) {
      return parsed as Record<string, unknown>;
    }
    return { output: parsed };
  } catch {
    return { output: content };
  }
}

function isFunctionResponseOnly(content: Content): boolean {
  const parts = content.parts ?? [];
  return parts.length > 0 && parts.every((part) => part.functionResponse !== undefined);
}

/** Maps normalized tools to a Gemini tool with JSON-schema declarations. */
export function toGeminiTools(tools: readonly NormalizedTool[]): Tool[] {
  const declarations: FunctionDeclaration[] = tools.map((tool) => ({
    name: tool.name,
    description: tool.description,
    parametersJsonSchema: tool.jsonSchema,
  }));
  return [{ functionDeclarations: declarations }];
}

/** Maps a tool choice to Gemini's ToolConfig. */
export function toGeminiToolConfig(choice: ToolChoice): ToolConfig {
  if (choice === "auto") {
    return { functionCallingConfig: { mode: FunctionCallingConfigMode.AUTO } };
  }
  if (choice === "none") {
    return { functionCallingConfig: { mode: FunctionCallingConfigMode.NONE } };
  }
  if (choice === "required") {
    return { functionCallingConfig: { mode: FunctionCallingConfigMode.ANY } };
  }
  return {
    functionCallingConfig: {
      mode: FunctionCallingConfigMode.ANY,
      allowedFunctionNames: [choice.name],
    },
  };
}

/** Builds the shared GenerateContentConfig from engine options. */
export function buildConfig(
  options: GenerateOptions,
  systemInstruction: string | undefined,
): GenerateContentConfig {
  const config: GenerateContentConfig = {};
  if (systemInstruction !== undefined) config.systemInstruction = systemInstruction;
  if (options.temperature !== undefined) config.temperature = options.temperature;
  if (options.maxTokens !== undefined) config.maxOutputTokens = options.maxTokens;
  if (options.topP !== undefined) config.topP = options.topP;
  if (options.stop !== undefined) config.stopSequences = options.stop;
  if (options.tools !== undefined && options.tools.length > 0) {
    config.tools = toGeminiTools(options.tools);
    if (options.toolChoice !== undefined) config.toolConfig = toGeminiToolConfig(options.toolChoice);
  }
  if (options.signal !== undefined) config.abortSignal = options.signal;
  return config;
}

/** Maps Gemini finish reasons to the unified FinishReason. */
export function mapFinishReason(reason: GeminiFinishReason | undefined): FinishReason {
  switch (reason) {
    case GeminiFinishReason.STOP:
      return "stop";
    case GeminiFinishReason.MAX_TOKENS:
      return "length";
    case GeminiFinishReason.SAFETY:
    case GeminiFinishReason.RECITATION:
    case GeminiFinishReason.LANGUAGE:
    case GeminiFinishReason.BLOCKLIST:
    case GeminiFinishReason.PROHIBITED_CONTENT:
    case GeminiFinishReason.SPII:
    case GeminiFinishReason.IMAGE_SAFETY:
      return "content_filter";
    case GeminiFinishReason.MALFORMED_FUNCTION_CALL:
    case GeminiFinishReason.UNEXPECTED_TOOL_CALL:
      return "error";
    case GeminiFinishReason.FINISH_REASON_UNSPECIFIED:
    case GeminiFinishReason.OTHER:
    default:
      return "unknown";
  }
}

/** Maps Gemini usage metadata to the unified TokenUsage. */
export function mapUsage(response: GenerateContentResponse): TokenUsage {
  const usage = response.usageMetadata;
  if (usage === undefined) return ZERO_USAGE;
  const promptTokens = usage.promptTokenCount ?? 0;
  const completionTokens = (usage.candidatesTokenCount ?? 0) + (usage.thoughtsTokenCount ?? 0);
  return {
    promptTokens,
    completionTokens,
    totalTokens: usage.totalTokenCount ?? promptTokens + completionTokens,
  };
}

/** Extracts text parts of a candidate, excluding thought (reasoning) parts. */
export function extractText(parts: readonly Part[] | undefined): string {
  if (parts === undefined) return "";
  let text = "";
  for (const part of parts) {
    if (part.text !== undefined && part.thought !== true) text += part.text;
  }
  return text;
}

/** Extracts functionCall parts as unified ToolCalls (ids synthesized when absent). */
export function extractToolCalls(
  parts: readonly Part[] | undefined,
  idPrefix: string,
): ToolCall[] | undefined {
  if (parts === undefined) return undefined;
  const calls: ToolCall[] = [];
  let index = 0;
  for (const part of parts) {
    const call: FunctionCall | undefined = part.functionCall;
    if (call === undefined) continue;
    calls.push({
      id: call.id !== undefined && call.id.length > 0 ? call.id : `${idPrefix}_${index}`,
      name: call.name ?? "",
      arguments: call.args ?? {},
    });
    index += 1;
  }
  return calls.length > 0 ? calls : undefined;
}

/** Returns the first candidate of a response, or undefined. */
export function firstCandidate(response: GenerateContentResponse) {
  return response.candidates?.[0];
}

/**
 * Validates a non-streaming response and rejects safety-blocked prompts with
 * a non-retryable content-filter error (never retry safety blocks).
 */
export function assertUsableResponse(response: GenerateContentResponse): void {
  if (response.candidates === undefined || response.candidates.length === 0) {
    const blockReason = response.promptFeedback?.blockReason;
    if (blockReason !== undefined) {
      throw new ProviderContentFilterError(
        PROVIDER,
        `Gemini blocked the prompt (reason: ${blockReason}).`,
        { details: { blockReason } },
      );
    }
    throw new ProviderError(PROVIDER, "Gemini returned a response with no candidates.", {
      code: "PROVIDER_MALFORMED_RESPONSE",
      retryable: false,
    });
  }
}
