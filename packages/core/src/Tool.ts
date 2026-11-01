import type { ZodType } from "zod";
import { ValidationError, redactSecrets, truncateMessage } from "./Errors";
import { formatIssues, isZodSchema, normalizeTool } from "./validation";
import type { JsonSchema, Message, NormalizedTool, ToolCall, ToolDefinition } from "./types";

/**
 * Tools: named, schema-validated functions the model can request.
 *
 * A {@link Tool} packages three things the engine needs:
 *
 * 1. the **definition** sent to the provider (name, description, JSON Schema
 *    parameters — converted from Zod once, at definition time),
 * 2. **argument validation**: model-produced arguments are parsed with the
 *    Zod schema before your code ever sees them,
 * 3. the **execute** function.
 *
 * Tool arguments are treated strictly as data: MurphyChain never executes
 * strings from a model response, and there is no `eval` anywhere in the
 * codebase. Use {@link executeToolCalls} (or `ai.runTools`) to dispatch the
 * calls a model requests.
 */

/** Options handed to a tool's execute function. */
export interface ToolExecuteOptions {
  /** Cancellation signal — pass it to any I/O the tool performs. */
  signal?: AbortSignal | undefined;
}

/** Configuration of a {@link Tool}. */
export interface ToolConfig<TArgs, TResult> {
  /** Tool name as exposed to the model (provider name rules apply). */
  name: string;
  /** What the tool does, in the model's language. Be specific. */
  description: string;
  /**
   * Argument schema: a Zod schema (validated; infers the execute argument
   * type) or a JSON Schema (passed through; execute receives `unknown`).
   */
  parameters: ZodType<TArgs> | JsonSchema;
  /** The function the engine calls with parsed arguments. */
  execute: (args: TArgs, options: ToolExecuteOptions) => TResult | Promise<TResult>;
}

/**
 * A heterogeneous tool list for dispatch helpers: they only need raw-args
 * execution, so argument/result types are intentionally erased here. This is
 * the single deliberate `any` in the public surface — dispatch forwards
 * `unknown` arguments and collects `unknown` results.
 */
export type AnyTool = Tool<any, any>; // eslint-disable-line @typescript-eslint/no-explicit-any

/** A tool that was executed (or failed to execute) on behalf of the model. */
export interface ExecutedToolCall {
  /** The call the model requested. */
  call: ToolCall;
  /** The `role: "tool"` message to append to the conversation. */
  message: Message;
  /** False when the tool was unknown, arguments were invalid, or execute threw. */
  ok: boolean;
  /** Parsed result when `ok`. */
  result?: unknown;
  /** Redacted failure summary when `!ok`. */
  error?: string;
}

/** Options for {@link executeToolCalls}. */
export interface ToolCallOptions {
  /** Cancellation signal forwarded to each tool. */
  signal?: AbortSignal | undefined;
  /**
   * What happens when a tool call fails (unknown tool, invalid arguments,
   * execute threw):
   *
   * - `"return"` (default): the failure is serialized into an `{"error": ...}`
   *   tool result so the model can recover on its own.
   * - `"throw"`: the error propagates to the caller (strict pipelines).
   */
  onError?: "return" | "throw" | undefined;
}

/**
 * A named function the model can invoke, with validated arguments.
 *
 * ```ts
 * const weather = Tool.define({
 *   name: "get_weather",
 *   description: "Get the current weather for a city",
 *   parameters: z.object({ city: z.string() }),
 *   execute: async ({ city }) => fetchWeather(city),
 * });
 * ```
 */
export class Tool<TArgs = Record<string, unknown>, TResult = unknown> {
  readonly name: string;
  readonly description: string;
  readonly parameters: ZodType | JsonSchema;

  private readonly executeFn: (
    args: TArgs,
    options: ToolExecuteOptions,
  ) => TResult | Promise<TResult>;
  private normalizedCache: NormalizedTool | undefined;

  private constructor(config: ToolConfig<TArgs, TResult>) {
    this.name = config.name;
    this.description = config.description;
    this.parameters = config.parameters;
    this.executeFn = config.execute;
  }

  /**
   * Defines a tool. Name, description, and parameter schema are validated
   * eagerly — a bad definition fails here, not mid-request.
   */
  static define<TArgs, TResult>(config: ToolConfig<TArgs, TResult>): Tool<TArgs, TResult> {
    if (typeof config !== "object" || config === null) {
      throw new ValidationError("Tool.define() requires a configuration object.");
    }
    if (typeof config.execute !== "function") {
      throw new ValidationError(
        `Tool "${String(config.name)}": execute must be a function.`,
        { tool: config.name },
      );
    }
    const tool = new Tool(config);
    // normalizeTool validates the name/description/schema and converts Zod to
    // JSON Schema exactly once — cached for every future request.
    tool.normalizedCache = normalizeTool({
      name: tool.name,
      description: tool.description,
      parameters: tool.parameters,
    });
    return tool;
  }

  /** The user-facing definition (what `ai.generate({ tools })` accepts). */
  get definition(): ToolDefinition {
    return { name: this.name, description: this.description, parameters: this.parameters };
  }

  /** Engine-normalized definition handed to provider adapters (cached). */
  get normalized(): NormalizedTool {
    this.normalizedCache ??= normalizeTool(this.definition);
    return this.normalizedCache;
  }

  /**
   * Validates model-produced arguments against the parameter schema.
   * Throws `ValidationError` (tool name + issues) on failure.
   */
  parseArguments(raw: unknown): TArgs {
    if (isZodSchema(this.parameters)) {
      const parsed = this.parameters.safeParse(raw);
      if (!parsed.success) {
        const issues = formatIssues(parsed.error);
        throw new ValidationError(
          `Tool "${this.name}" received invalid arguments: ${issues
            .map((issue) => `${issue.path || "(root)"}: ${issue.message}`)
            .join("; ")}`,
          { tool: this.name, issues },
        );
      }
      return parsed.data as TArgs;
    }
    // JSON Schema parameters can't be enforced without a validator dependency;
    // enforce the one structural invariant the dispatch contract requires.
    if (typeof raw !== "object" || raw === null || Array.isArray(raw)) {
      throw new ValidationError(
        `Tool "${this.name}" expects a JSON object of arguments.`,
        { tool: this.name, received: typeof raw },
      );
    }
    return raw as TArgs;
  }

  /** Parses arguments then runs the tool. */
  async run(rawArgs: unknown, options: ToolExecuteOptions = {}): Promise<TResult> {
    const args = this.parseArguments(rawArgs);
    return await this.executeFn(args, options);
  }

  /**
   * Serializes a tool result into the string sent back to the model:
   * strings pass through; everything else is JSON (`undefined` → `"null"`).
   */
  serializeResult(result: unknown): string {
    if (typeof result === "string") return result;
    if (result === undefined) return "null";
    try {
      return JSON.stringify(result) ?? "null";
    } catch {
      // Circular or otherwise unserializable results still get a reply.
      return String(result);
    }
  }
}

/**
 * Executes the tool calls a model requested and returns the `role: "tool"`
 * messages to append to the conversation, in call order.
 *
 * By default failures (unknown tool, invalid arguments, execute threw) are
 * serialized as `{"error": ...}` results with secrets redacted, so the model
 * can see what went wrong and correct itself. Pass `onError: "throw"` for
 * strict pipelines.
 */
export async function executeToolCalls(
  tools: readonly AnyTool[],
  calls: readonly ToolCall[],
  options: ToolCallOptions = {},
): Promise<Message[]> {
  const byName = new Map<string, AnyTool>();
  for (const tool of tools) byName.set(tool.name, tool);

  const messages: Message[] = [];
  for (const call of calls) {
    messages.push(await executeOne(byName, call, options));
  }
  return messages;
}

async function executeOne(
  byName: ReadonlyMap<string, AnyTool>,
  call: ToolCall,
  options: ToolCallOptions,
): Promise<Message> {
  const reply = (content: string): Message => ({
    role: "tool",
    name: call.name,
    toolCallId: call.id,
    content,
  });

  try {
    const tool = byName.get(call.name);
    if (tool === undefined) {
      const available = [...byName.keys()].join(", ") || "(none)";
      throw new ValidationError(
        `Unknown tool "${call.name}". Available tools: ${available}.`,
        { tool: call.name },
      );
    }
    const executeOptions: ToolExecuteOptions =
      options.signal !== undefined ? { signal: options.signal } : {};
    const result = await tool.run(call.arguments, executeOptions);
    return reply(tool.serializeResult(result));
  } catch (error) {
    if (options.onError === "throw") throw error;
    const message = error instanceof Error ? error.message : String(error);
    // Redact and bound what goes back to the model / into the conversation.
    return reply(JSON.stringify({ error: truncateMessage(redactSecrets(message)) }));
  }
}
