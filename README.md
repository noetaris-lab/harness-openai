# @noetaris/harness-openai

OpenAI adapter for [@noetaris/harness](../core).

## Overview

`@noetaris/harness-openai` provides an `OpenAI` class that implements the `LLM` and `ObserverAware` interfaces from `@noetaris/harness`. It handles translation between the harness message format and the OpenAI `chat.completions.create` format, and emits telemetry events (token usage, model ID) through an attached `Observer`.

## Installation

```sh
pnpm add @noetaris/harness-openai
```

Peer dependencies:

```sh
pnpm add @noetaris/harness @noetaris/harness-types @noetaris/harness-openai-models
```

`@noetaris/harness-openai-models` supplies the context-window lookup table used to
populate `contextWindowSize` on responses and usage events.

Requires Node.js ≥ 22.

## Usage

```ts
import { OpenAI } from '@noetaris/harness-openai'

// The model ID is the required first argument; options are optional.
const llm = new OpenAI('gpt-4o-mini', {
  apiKey: process.env.OPENAI_API_KEY, // defaults to the OPENAI_API_KEY env var
})

// Wire into a harness provider slot
h.provide('model', runtime())

const agent = createAgent(h, { prompts: { system: '...' } })
const run = agent.run(initialState, { model: llm })
```

## API

### `OpenAI`

```ts
new OpenAI(model: string, options?: OpenAIOptions)
```

Implements `LLM` and `ObserverAware`. `OpenAIOptions` accepts `apiKey` and the
generation parameters `temperature`, `maxTokens`, `topP`, and `seed`.

- **`invoke(messages, options?)`** — translates harness `Message[]` and `Tool[]` to OpenAI format, calls `chat.completions.create()`, and maps the response back to an `LLMResponse` (including `tool_calls` extraction and a required `usage: { inputTokens, outputTokens, contextWindowSize? }` field; `contextWindowSize` is resolved from `@noetaris/harness-openai-models`).
- **`bindObserver(observer)`** — attaches an `Observer`. Each `invoke` emits an `"llm.request"` event (`{ modelId, providerName: 'openai' }`) before the call and an `"llm.response"` event (`{ tokens: { input, output }, modelId, stopReason, providerName, contextWindowSize? }`) from the response `usage` after it.
- **`setStepContext(ctx)`** — sets the `StepContext` attached to emitted events; called by the harness before each step.

### `MockOpenAI`

A deterministic test double for use in tests and demos without a real API key.

## Related Packages

- [`@noetaris/harness`](https://github.com/noetaris-lab/harness) — core execution engine
- [`@noetaris/harness-types`](https://github.com/noetaris-lab/harness-types) — shared LLM type contract
- [`@noetaris/harness-openai-models`](https://github.com/noetaris-lab/harness-openai-models) — OpenAI context-window lookup table
- [`@noetaris/harness-anthropic`](https://github.com/noetaris-lab/harness-anthropic) — Anthropic Claude adapter
- [`@noetaris/harness-google`](https://github.com/noetaris-lab/harness-google) — Google Gemini adapter
- [`@noetaris/harness-otel`](https://github.com/noetaris-lab/harness-otel) — OpenTelemetry observer bridge

## License

MIT
