# @noetaris/harness-openai

OpenAI adapter for [@noetaris/harness](../core).

> **Status:** not yet released. Implementation tracked in F21.

## Overview

`@noetaris/harness-openai` provides an `OpenAI` class that implements the `LLM` and `ObserverAware` interfaces from `@noetaris/harness`. It handles translation between the harness message format and the OpenAI `chat.completions.create` format, and emits telemetry events (token usage, model ID) through an attached `Observer`.

## Installation

```sh
pnpm add @noetaris/harness-openai
```

Peer dependencies:

```sh
pnpm add @noetaris/harness @noetaris/harness-types
```

Requires Node.js ≥ 22.

## Usage

```ts
import { OpenAI } from '@noetaris/harness-openai'

const llm = new OpenAI({ apiKey: process.env.OPENAI_API_KEY })

// Wire into a harness provider slot
h.provide('model', runtime())

const agent = createAgent(h, { prompts: { system: '...' } })
const run = agent.run(initialState, { model: llm })
```

## API

### `OpenAI`

Implements `LLM` and `ObserverAware`.

- **`invoke(messages, options?)`** — translates harness `Message[]` and `Tool[]` to OpenAI format, calls `chat.completions.create()`, and maps the response back to `LLMResponse` (including `tool_calls` extraction).
- **`bindObserver(observer)`** — attaches an `Observer`; after each `invoke`, emits an `"llm.response"` event with `{ tokens: { input, output }, modelId }` from `usage` in the response.

### `MockOpenAI`

A deterministic test double for use in tests and demos without a real API key.

## Related Packages

- [`@noetaris/harness`](https://github.com/noetaris-lab/harness) — core execution engine
- [`@noetaris/harness-types`](https://github.com/noetaris-lab/harness-types) — shared LLM type contract
- [`@noetaris/harness-anthropic`](https://github.com/noetaris-lab/harness-anthropic) — Anthropic Claude adapter
- [`@noetaris/harness-google`](https://github.com/noetaris-lab/harness-google) — Google Gemini adapter
- [`@noetaris/harness-otel`](https://github.com/noetaris-lab/harness-otel) — OpenTelemetry observer bridge

## License

MIT
