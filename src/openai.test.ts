import { describe, it, expect, vi, beforeEach } from 'vitest'
import { OpenAI } from './openai.js'

// minimal stop response used across multiple test cases
const minimalStopResponse = {
  choices: [{ message: { role: 'assistant', content: 'ok', tool_calls: undefined }, finish_reason: 'stop' }],
  usage: { prompt_tokens: 5, completion_tokens: 2 },
}

const mockCreate = vi.fn()

vi.mock('openai', () => {
  function MockOpenAISDK() {
    return { chat: { completions: { create: mockCreate } } }
  }
  return { default: MockOpenAISDK }
})

beforeEach(() => {
  vi.clearAllMocks()
  mockCreate.mockResolvedValue(minimalStopResponse)
})

describe('OpenAI', () => {

  describe('Group 1: Basic invocation and request shape', () => {

    it('sends correct model and user message and returns normalized LLMResponse', async () => {
      // arrange
      mockCreate.mockResolvedValue({
        choices: [{ message: { role: 'assistant', content: 'Hello back', tool_calls: undefined }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 10, completion_tokens: 5 },
      })
      const adapter = new OpenAI('gpt-4o')
      const observer = { onEvent: vi.fn() }
      adapter.bindObserver(observer)

      // act
      const result = await adapter.invoke([{ role: 'user', content: 'hello' }])

      // assert
      expect(mockCreate).toHaveBeenCalledWith(expect.objectContaining({ model: 'gpt-4o', messages: [{ role: 'user', content: 'hello' }] }))
      expect(result.text).toBe('Hello back')
      expect(result.toolCalls).toEqual([])
      expect(result.stopReason).toBe('end')
    })

    it('omits tools field from SDK call when options not provided', async () => {
      // arrange
      const adapter = new OpenAI('gpt-4o')

      // act
      const result = await adapter.invoke([{ role: 'user', content: 'ping' }])

      // assert
      expect(mockCreate).toHaveBeenCalledWith(expect.not.objectContaining({ tools: expect.anything() }))
      expect(result.text).toBe('ok')
    })

  })

  describe('Group 2: Message translation (harness → OpenAI format)', () => {

    it('translates Tool array to OpenAI tools format', async () => {
      // arrange
      const tool = { name: 'get_weather', description: 'Get current weather', inputSchema: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] } }
      mockCreate.mockResolvedValue({
        choices: [{ message: { role: 'assistant', content: null, tool_calls: [{ id: 'tc1', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Paris"}' } }] }, finish_reason: 'tool_calls' }],
        usage: { prompt_tokens: 20, completion_tokens: 8 },
      })
      const adapter = new OpenAI('gpt-4o')

      // act
      await adapter.invoke([{ role: 'user', content: 'What is the weather?' }], { tools: [tool] })

      // assert
      expect(mockCreate).toHaveBeenCalledWith(expect.objectContaining({
        tools: [{ type: 'function', function: { name: 'get_weather', description: 'Get current weather', parameters: { type: 'object', properties: { city: { type: 'string' } }, required: ['city'] } } }],
      }))
    })

    it('translates assistant message with toolCalls only — content is null', async () => {
      // arrange
      // omit content entirely (exactOptionalPropertyTypes: no explicit undefined)
      const assistantMsg: { role: 'assistant'; toolCalls: { id: string; name: string; input: { city: string } }[] } = { role: 'assistant', toolCalls: [{ id: 'tc1', name: 'get_weather', input: { city: 'Paris' } }] }
      const adapter = new OpenAI('gpt-4o')

      // act
      await adapter.invoke([assistantMsg])

      // assert
      expect(mockCreate).toHaveBeenCalledWith(expect.objectContaining({
        messages: [{ role: 'assistant', content: null, tool_calls: [{ id: 'tc1', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Paris"}' } }] }],
      }))
    })

    it('translates assistant message with both content and toolCalls', async () => {
      // arrange
      const assistantMsg = { role: 'assistant' as const, content: 'Calling tool', toolCalls: [{ id: 'tc2', name: 'lookup', input: { q: 'test' } }] }
      const adapter = new OpenAI('gpt-4o')

      // act
      await adapter.invoke([assistantMsg])

      // assert
      expect(mockCreate).toHaveBeenCalledWith(expect.objectContaining({
        messages: [{ role: 'assistant', content: 'Calling tool', tool_calls: [{ id: 'tc2', type: 'function', function: { name: 'lookup', arguments: '{"q":"test"}' } }] }],
      }))
    })

    it('translates a single tool message with correct role and fields', async () => {
      // arrange
      const toolMsg = { role: 'tool' as const, toolCallId: 'tc1', content: '{"temp":22}' }
      const adapter = new OpenAI('gpt-4o')

      // act
      await adapter.invoke([toolMsg])

      // assert
      expect(mockCreate).toHaveBeenCalledWith(expect.objectContaining({
        messages: [{ role: 'tool', tool_call_id: 'tc1', content: '{"temp":22}' }],
      }))
    })

    it('translates multiple consecutive tool messages as separate messages — no grouping', async () => {
      // arrange
      const msgs = [
        { role: 'tool' as const, toolCallId: 'tc1', content: 'result1' },
        { role: 'tool' as const, toolCallId: 'tc2', content: 'result2' },
      ]
      mockCreate.mockResolvedValue({
        choices: [{ message: { role: 'assistant', content: 'got both', tool_calls: undefined }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 20, completion_tokens: 5 },
      })
      const adapter = new OpenAI('gpt-4o')

      // act
      await adapter.invoke(msgs)

      // assert
      expect(mockCreate).toHaveBeenCalledWith(expect.objectContaining({
        messages: [
          { role: 'tool', tool_call_id: 'tc1', content: 'result1' },
          { role: 'tool', tool_call_id: 'tc2', content: 'result2' },
        ],
      }))
    })

    it('translates assistant message with no content and no toolCalls — content is empty string', async () => {
      // arrange
      // omit both optional properties entirely (exactOptionalPropertyTypes: no explicit undefined)
      const assistantMsg: { role: 'assistant' } = { role: 'assistant' }
      const adapter = new OpenAI('gpt-4o')

      // act
      await adapter.invoke([assistantMsg])

      // assert
      expect(mockCreate).toHaveBeenCalledWith(expect.objectContaining({
        messages: [{ role: 'assistant', content: '' }],
      }))
    })

  })

  describe('Group 3: Response normalization (OpenAI → LLMResponse)', () => {

    it('normalizes tool_calls response with stopReason tool_use and parsed input', async () => {
      // arrange
      mockCreate.mockResolvedValue({
        choices: [{ message: { role: 'assistant', content: null, tool_calls: [{ id: 'tc1', type: 'function', function: { name: 'get_weather', arguments: '{"city":"Paris"}' } }] }, finish_reason: 'tool_calls' }],
        usage: { prompt_tokens: 20, completion_tokens: 8 },
      })
      const adapter = new OpenAI('gpt-4o')

      // act
      const result = await adapter.invoke([{ role: 'user', content: 'weather?' }], { tools: [{ name: 'get_weather', description: 'Get weather', inputSchema: {} }] })

      // assert
      expect(result.toolCalls).toEqual([{ id: 'tc1', name: 'get_weather', input: { city: 'Paris' } }])
      expect(result.text).toBe('')
      expect(result.stopReason).toBe('tool_use')
    })

    it('maps finish_reason "length" to stopReason "max_tokens"', async () => {
      // arrange
      mockCreate.mockResolvedValue({
        choices: [{ message: { role: 'assistant', content: 'truncated...', tool_calls: undefined }, finish_reason: 'length' }],
        usage: { prompt_tokens: 100, completion_tokens: 200 },
      })
      const adapter = new OpenAI('gpt-4o')

      // act
      const result = await adapter.invoke([{ role: 'user', content: 'write a lot' }])

      // assert
      expect(result.stopReason).toBe('max_tokens')
    })

    it('maps unrecognized finish_reason to stopReason "end"', async () => {
      // arrange
      mockCreate.mockResolvedValue({
        choices: [{ message: { role: 'assistant', content: 'filtered', tool_calls: undefined }, finish_reason: 'content_filter' }],
        usage: { prompt_tokens: 10, completion_tokens: 3 },
      })
      const adapter = new OpenAI('gpt-4o')

      // act
      const result = await adapter.invoke([{ role: 'user', content: 'test' }])

      // assert
      expect(result.stopReason).toBe('end')
    })

    it('parses valid JSON arguments string to object in ToolCall.input', async () => {
      // arrange
      mockCreate.mockResolvedValue({
        choices: [{ message: { role: 'assistant', content: null, tool_calls: [{ id: 'tc1', type: 'function', function: { name: 'lookup', arguments: '{"q":"hello","limit":5}' } }] }, finish_reason: 'tool_calls' }],
        usage: { prompt_tokens: 10, completion_tokens: 4 },
      })
      const adapter = new OpenAI('gpt-4o')

      // act
      const result = await adapter.invoke([{ role: 'user', content: 'search' }], { tools: [{ name: 'lookup', description: 'search', inputSchema: {} }] })

      // assert
      expect(result.toolCalls[0]?.input).toEqual({ q: 'hello', limit: 5 })
    })

    it('uses raw string as input when arguments JSON is malformed — no throw', async () => {
      // arrange
      const malformedArgs = '{"city": "Pa'
      mockCreate.mockResolvedValue({
        choices: [{ message: { role: 'assistant', content: null, tool_calls: [{ id: 'tc1', type: 'function', function: { name: 'get_weather', arguments: malformedArgs } }] }, finish_reason: 'tool_calls' }],
        usage: { prompt_tokens: 10, completion_tokens: 4 },
      })
      const adapter = new OpenAI('gpt-4o')

      // act
      const result = await adapter.invoke([{ role: 'user', content: 'weather?' }], { tools: [{ name: 'get_weather', description: 'Get weather', inputSchema: {} }] })

      // assert
      expect(result.toolCalls[0]?.input).toBe(malformedArgs)
      expect(result.toolCalls[0]?.input).not.toBeUndefined()
    })

    it('maps null content to empty string', async () => {
      // arrange
      mockCreate.mockResolvedValue({
        choices: [{ message: { role: 'assistant', content: null, tool_calls: undefined }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 5, completion_tokens: 0 },
      })
      const adapter = new OpenAI('gpt-4o')

      // act
      const result = await adapter.invoke([{ role: 'user', content: 'hello' }])

      // assert
      expect(result.text).toBe('')
    })

    it('normalizes response with both content and tool_calls', async () => {
      // arrange
      mockCreate.mockResolvedValue({
        choices: [{ message: { role: 'assistant', content: 'Sure, fetching...', tool_calls: [{ id: 'tc1', type: 'function', function: { name: 'fetch', arguments: '{}' } }] }, finish_reason: 'tool_calls' }],
        usage: { prompt_tokens: 15, completion_tokens: 10 },
      })
      const adapter = new OpenAI('gpt-4o')

      // act
      const result = await adapter.invoke([{ role: 'user', content: 'go' }], { tools: [{ name: 'fetch', description: 'fetch data', inputSchema: {} }] })

      // assert
      expect(result.text).toBe('Sure, fetching...')
      expect(result.toolCalls).toHaveLength(1)
      expect(result.toolCalls[0]).toMatchObject({ id: 'tc1', name: 'fetch', input: {} })
      expect(result.stopReason).toBe('tool_use')
    })

  })

  describe('Group 4: Observer wiring and StepContext', () => {

    it('calls observer.onEvent with correct payload after successful invoke', async () => {
      // arrange
      mockCreate.mockResolvedValue({
        choices: [{ message: { role: 'assistant', content: 'hi', tool_calls: undefined }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 12, completion_tokens: 7 },
      })
      const adapter = new OpenAI('gpt-4o')
      const observer = { onEvent: vi.fn() }
      adapter.bindObserver(observer)
      const stepCtx = { agentId: 'agent-1', sessionId: 'sess-1', stepName: 'step-1' }
      adapter.setStepContext(stepCtx)

      // act
      await adapter.invoke([{ role: 'user', content: 'hello' }])

      // assert
      expect(observer.onEvent).toHaveBeenCalledOnce()
      expect(observer.onEvent).toHaveBeenCalledWith(
        { agentId: 'agent-1', sessionId: 'sess-1', stepName: 'step-1' },
        'llm.response',
        { tokens: { input: 12, output: 7 }, modelId: 'gpt-4o', stopReason: 'end' },
      )
    })

    it('does not throw when observer is NOOP ({} with no onEvent method)', async () => {
      // arrange
      const adapter = new OpenAI('gpt-4o')
      adapter.bindObserver({})

      // act / assert
      await expect(adapter.invoke([{ role: 'user', content: 'hi' }])).resolves.not.toThrow()
    })

    it('uses zeroed StepContext when setStepContext was never called', async () => {
      // arrange
      mockCreate.mockResolvedValue({
        choices: [{ message: { role: 'assistant', content: 'hi', tool_calls: undefined }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 5, completion_tokens: 2 },
      })
      const adapter = new OpenAI('gpt-4o')
      const observer = { onEvent: vi.fn() }
      adapter.bindObserver(observer)

      // act
      await adapter.invoke([{ role: 'user', content: 'hello' }])

      // assert
      expect(observer.onEvent).toHaveBeenCalledWith(
        { agentId: '', sessionId: '', stepName: '' },
        'llm.response',
        expect.any(Object),
      )
    })

    it('uses the StepContext from setStepContext in observer event', async () => {
      // arrange
      mockCreate.mockResolvedValue({
        choices: [{ message: { role: 'assistant', content: 'done', tool_calls: undefined }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 8, completion_tokens: 3 },
      })
      const adapter = new OpenAI('gpt-4o')
      const observer = { onEvent: vi.fn() }
      adapter.bindObserver(observer)
      adapter.setStepContext({ agentId: 'my-agent', sessionId: 'my-sess', stepName: 'my-step' })

      // act
      await adapter.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(observer.onEvent).toHaveBeenCalledWith(
        { agentId: 'my-agent', sessionId: 'my-sess', stepName: 'my-step' },
        'llm.response',
        expect.any(Object),
      )
    })

  })

  describe('Group 5: Error propagation', () => {

    it('propagates OpenAI API error from SDK — onEvent not called', async () => {
      // arrange
      const apiError = new Error('401 Unauthorized')
      mockCreate.mockRejectedValue(apiError)
      const adapter = new OpenAI('gpt-4o')
      const observer = { onEvent: vi.fn() }
      adapter.bindObserver(observer)

      // act / assert
      await expect(adapter.invoke([{ role: 'user', content: 'hi' }])).rejects.toThrow('401 Unauthorized')
      expect(observer.onEvent).not.toHaveBeenCalled()
    })

    it('propagates network error from SDK', async () => {
      // arrange
      const networkError = new Error('ECONNREFUSED')
      mockCreate.mockRejectedValue(networkError)
      const adapter = new OpenAI('gpt-4o')

      // act / assert
      await expect(adapter.invoke([{ role: 'user', content: 'hi' }])).rejects.toThrow('ECONNREFUSED')
    })

    it('throws specific error when choices array is empty — onEvent not called', async () => {
      // arrange
      mockCreate.mockResolvedValue({ choices: [], usage: { prompt_tokens: 5, completion_tokens: 0 } })
      const adapter = new OpenAI('gpt-4o')
      const observer = { onEvent: vi.fn() }
      adapter.bindObserver(observer)

      // act / assert
      await expect(adapter.invoke([{ role: 'user', content: 'hi' }])).rejects.toThrow('OpenAI response contained no choices')
      expect(observer.onEvent).not.toHaveBeenCalled()
    })

  })

  describe('Group 6: Edge cases and repeated calls', () => {

    it('passes empty messages array to the SDK without modification', async () => {
      // arrange
      const adapter = new OpenAI('gpt-4o')

      // act
      await adapter.invoke([])

      // assert
      expect(mockCreate).toHaveBeenCalledWith(expect.objectContaining({ messages: [] }))
    })

    it('second bindObserver call replaces first — subsequent invokes use new observer', async () => {
      // arrange
      const adapter = new OpenAI('gpt-4o')
      const firstObserver = { onEvent: vi.fn() }
      const secondObserver = { onEvent: vi.fn() }
      adapter.bindObserver(firstObserver)
      adapter.bindObserver(secondObserver)

      // act
      await adapter.invoke([{ role: 'user', content: 'hello' }])

      // assert
      expect(secondObserver.onEvent).toHaveBeenCalledOnce()
      expect(firstObserver.onEvent).not.toHaveBeenCalled()
    })

    it('last setStepContext call wins when called multiple times before invoke', async () => {
      // arrange
      const adapter = new OpenAI('gpt-4o')
      const observer = { onEvent: vi.fn() }
      adapter.bindObserver(observer)
      adapter.setStepContext({ agentId: 'a1', sessionId: 's1', stepName: 'step-old' })
      adapter.setStepContext({ agentId: 'a1', sessionId: 's1', stepName: 'step-new' })

      // act
      await adapter.invoke([{ role: 'user', content: 'hi' }])

      // assert
      expect(observer.onEvent).toHaveBeenCalledWith(
        { agentId: 'a1', sessionId: 's1', stepName: 'step-new' },
        'llm.response',
        expect.any(Object),
      )
    })

  })

})
