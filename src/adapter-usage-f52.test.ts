import { describe, it, expect, vi, beforeEach } from 'vitest'
import type { Message } from '@noetaris/harness-types'

// mockCreate is declared in outer scope so the mock factory can close over it
const mockCreate = vi.fn()

vi.mock('openai', () => {
  function MockOpenAISDK() {
    return { chat: { completions: { create: mockCreate } } }
  }
  return { default: MockOpenAISDK }
})

vi.mock('@noetaris/harness-openai-models', () => ({
  getContextWindow: vi.fn().mockReturnValue(128000),
}))

import { OpenAI } from './openai.js'
import { MockOpenAI } from './mock-openai.js'
import { getContextWindow } from '@noetaris/harness-openai-models'

const messages: Message[] = [{ role: 'user', content: 'Hello' }]

const minimalStopResponse = {
  choices: [{ message: { content: 'Hello', tool_calls: [] }, finish_reason: 'stop' }],
  usage: { prompt_tokens: 30, completion_tokens: 15 },
}

beforeEach(() => {
  vi.clearAllMocks()
  vi.mocked(getContextWindow).mockReturnValue(128000)
  mockCreate.mockResolvedValue(minimalStopResponse)
})

describe('OpenAI — AdapterUsageF52', () => {

  describe('Group 4: Static Table Lookup', () => {

    it('returns contextWindowSize from getContextWindow for a known model ID; token counts correct', async () => {
      // arrange
      vi.mocked(getContextWindow).mockReturnValue(128000)
      mockCreate.mockResolvedValue({
        choices: [{ message: { content: 'Hello', tool_calls: [] }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 30, completion_tokens: 15 },
      })
      const observer = { onEvent: vi.fn() }
      const openai = new OpenAI('gpt-4o', { apiKey: 'test-key' })
      openai.bindObserver(observer)

      // act
      const result = await openai.invoke(messages)

      // assert
      expect(getContextWindow).toHaveBeenCalledWith('gpt-4o')
      expect(result.usage.contextWindowSize).toBe(128000)
      expect(result.usage.inputTokens).toBe(30)
      expect(result.usage.outputTokens).toBe(15)
      const event = observer.onEvent.mock.calls.find((call) => call[1] === 'llm.response')?.[2]
      expect(event.contextWindowSize).toBe(128000)
    })

    it('returns contextWindowSize as undefined for an unknown model; invoke still returns valid LLMResponse', async () => {
      // arrange
      vi.mocked(getContextWindow).mockReturnValue(undefined)
      mockCreate.mockResolvedValue({
        choices: [{ message: { content: 'OK' }, finish_reason: 'stop' }],
        usage: { prompt_tokens: 5, completion_tokens: 2 },
      })
      const openai = new OpenAI('gpt-unknown-v99', { apiKey: 'test-key' })

      // act
      const result = await openai.invoke(messages)

      // assert
      expect(result.usage.contextWindowSize).toBeUndefined()
      expect(result.usage.inputTokens).toBe(5)
      expect(result.usage.outputTokens).toBe(2)
      expect(result.text).toBe('OK')
    })

  })

})

describe('MockOpenAI — AdapterUsageF52', () => {

  describe('Group 6: Fixed Zero Usage', () => {

    it('invoke returns usage = { inputTokens: 0, outputTokens: 0 } with no contextWindowSize; emitted event has no contextWindowSize', async () => {
      // arrange
      const observer = { onEvent: vi.fn() }
      const mockOpenAI = new MockOpenAI({
        text: 'Ok',
        toolCalls: [],
        stopReason: 'end',
        usage: { inputTokens: 0, outputTokens: 0 },
      })
      mockOpenAI.bindObserver(observer)

      // act
      const result = await mockOpenAI.invoke(messages)

      // assert
      expect(result.usage).toEqual({ inputTokens: 0, outputTokens: 0 })
      expect(result.usage.contextWindowSize).toBeUndefined()
      const event = observer.onEvent.mock.calls.find((call) => call[1] === 'llm.response')?.[2]
      expect(event).not.toHaveProperty('contextWindowSize')
    })

  })

})
