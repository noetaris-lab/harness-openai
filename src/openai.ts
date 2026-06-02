import type { LLM, Message, Tool, ToolCall, LLMResponse, LLMUsageEvent } from '@noetaris/harness-types'
import type { ObserverAware, Observer, StepContext } from '@noetaris/harness'
import OpenAISDK from 'openai'

/** Options for {@link OpenAI}. */
export interface OpenAIOptions {
  /** OpenAI API key. Defaults to the `OPENAI_API_KEY` environment variable. */
  apiKey?: string
  /**
   * Sampling temperature in [0, 2]. Higher values produce more random output.
   * When absent, the provider default applies.
   */
  temperature?: number
  /**
   * Maximum number of tokens to generate.
   * When absent, the provider default applies.
   */
  maxTokens?: number
  /**
   * Top-p nucleus sampling probability.
   * When absent, the provider default applies.
   */
  topP?: number
  /**
   * Random seed for deterministic sampling.
   * When absent, the provider default applies (non-deterministic).
   */
  seed?: number
}

type OpenAIToolCall = {
  id: string
  type: 'function'
  function: {
    name: string
    arguments: string
  }
}

type OpenAITool = {
  type: 'function'
  function: {
    name: string
    description: string
    parameters: Record<string, unknown>
  }
}

type OpenAIUserMessage = {
  role: 'user'
  content: string
}

type OpenAIAssistantMessage = {
  role: 'assistant'
  content: string | null
  tool_calls?: OpenAIToolCall[]
}

type OpenAIToolMessage = {
  role: 'tool'
  tool_call_id: string
  content: string
}

type OpenAIMessage = OpenAIUserMessage | OpenAIAssistantMessage | OpenAIToolMessage

function translateMessages(messages: Message[]): OpenAIMessage[] {
  return messages.map((msg): OpenAIMessage => {
    if (msg.role === 'user') {
      return { role: 'user', content: msg.content }
    }
    if (msg.role === 'assistant') {
      if (msg.toolCalls && msg.toolCalls.length > 0) {
        const toolCalls: OpenAIToolCall[] = msg.toolCalls.map((tc) => ({
          id: tc.id,
          type: 'function',
          function: { name: tc.name, arguments: JSON.stringify(tc.input) },
        }))
        const result: OpenAIAssistantMessage = {
          role: 'assistant',
          content: msg.content ?? null,
          tool_calls: toolCalls,
        }
        return result
      }
      return { role: 'assistant', content: msg.content ?? '' }
    }
    // role === 'tool'
    return { role: 'tool', tool_call_id: msg.toolCallId, content: msg.content }
  })
}

function translateTools(tools: Tool[]): OpenAITool[] {
  return tools.map((t) => ({
    type: 'function',
    function: {
      name: t.name,
      description: t.description,
      parameters: t.inputSchema,
    },
  }))
}

function mapFinishReason(finishReason: string): LLMResponse['stopReason'] {
  if (finishReason === 'stop') return 'end'
  if (finishReason === 'tool_calls') return 'tool_use'
  if (finishReason === 'length') return 'max_tokens'
  return 'end'
}

function parseToolCallInput(args: string): unknown {
  try {
    return JSON.parse(args) as unknown
  } catch {
    return args
  }
}

function normalizeResponse(response: { choices: Array<{ message: { content: string | null; tool_calls?: OpenAIToolCall[] }; finish_reason: string }>; usage: { prompt_tokens: number; completion_tokens: number } }): LLMResponse {
  const choice = response.choices[0]
  if (choice === undefined) {
    throw new Error('OpenAI response contained no choices')
  }

  const text = choice.message.content ?? ''
  const toolCalls: ToolCall[] = (choice.message.tool_calls ?? []).map((tc) => ({
    id: tc.id,
    name: tc.function.name,
    input: parseToolCallInput(tc.function.arguments),
  }))

  return {
    text,
    toolCalls,
    stopReason: mapFinishReason(choice.finish_reason),
  }
}

const ZEROED_STEP_CONTEXT: StepContext = { agentId: '', sessionId: '', stepName: '' }

/**
 * {@link LLM} adapter for the OpenAI Chat Completions API.
 *
 * Implements {@link ObserverAware} — emits an `'llm.response'` event with an
 * `LLMUsageEvent` payload after each successful invocation.
 *
 * @example
 * ```ts
 * const llm = new OpenAI('gpt-4o-mini')
 * const response = await llm.invoke(messages)
 * ```
 */
export class OpenAI implements LLM, ObserverAware {
  private readonly client: OpenAISDK
  private readonly model: string
  private readonly options?: OpenAIOptions
  private observer: Observer = {}
  private stepContext: StepContext = ZEROED_STEP_CONTEXT

  /**
   * @param model - OpenAI model ID, e.g. `'gpt-4o-mini'`.
   * @param options - Optional configuration including API key and generation params.
   */
  constructor(model: string, options?: OpenAIOptions) {
    this.model = model
    this.options = options
    this.client = new OpenAISDK({ apiKey: options?.apiKey })
  }

  bindObserver(observer: Observer): void {
    this.observer = observer
  }

  setStepContext(ctx: StepContext): void {
    this.stepContext = ctx
  }

  async invoke(messages: Message[], options?: { tools?: Tool[] }): Promise<LLMResponse> {
    const translatedMessages = translateMessages(messages)
    const tools = options?.tools

    const response = await this.client.chat.completions.create({
      model: this.model,
      messages: translatedMessages as OpenAISDK.Chat.Completions.ChatCompletionMessageParam[],
      ...(tools !== undefined ? { tools: translateTools(tools) as OpenAISDK.Chat.Completions.ChatCompletionTool[] } : {}),
      ...(this.options?.temperature !== undefined ? { temperature: this.options.temperature } : {}),
      ...(this.options?.maxTokens !== undefined ? { max_tokens: this.options.maxTokens } : {}),
      ...(this.options?.topP !== undefined ? { top_p: this.options.topP } : {}),
      ...(this.options?.seed !== undefined ? { seed: this.options.seed } : {}),
    })

    if (response.choices.length === 0) {
      throw new Error('OpenAI response contained no choices')
    }

    // noUncheckedIndexedAccess: we guard above so this is safe
    const firstChoice = response.choices[0] as NonNullable<typeof response.choices[0]>

    const rawToolCalls = firstChoice.message.tool_calls
    const toolCallsForNormalize: OpenAIToolCall[] | undefined = rawToolCalls?.map((tc) => {
      // as: SDK union type includes ChatCompletionMessageCustomToolCall which omits .function;
      // all practical tool calls from chat.completions have .function populated
      const sdkTc = tc as { id: string; function: { name: string; arguments: string } }
      return {
        id: sdkTc.id,
        type: 'function' as const,
        function: { name: sdkTc.function.name, arguments: sdkTc.function.arguments },
      }
    })

    const normalizedResponse = normalizeResponse({
      choices: [{
        message: {
          content: firstChoice.message.content,
          ...(toolCallsForNormalize !== undefined ? { tool_calls: toolCallsForNormalize } : {}),
        },
        finish_reason: firstChoice.finish_reason ?? 'stop',
      }],
      usage: {
        prompt_tokens: response.usage?.prompt_tokens ?? 0,
        completion_tokens: response.usage?.completion_tokens ?? 0,
      },
    })

    const event: LLMUsageEvent = {
      tokens:     { input: response.usage?.prompt_tokens ?? 0, output: response.usage?.completion_tokens ?? 0 },
      modelId:    this.model,
      stopReason: normalizedResponse.stopReason,
      providerName: 'openai',
    }
    this.observer.onEvent?.(this.stepContext, 'llm.response', event)

    return normalizedResponse
  }
}
