import type { UIMessage } from 'ai'
import { createDeepSeek } from '@ai-sdk/deepseek'
import { convertToModelMessages, streamText } from 'ai'
import { createError, readBody } from 'h3'

export const maxDuration = 30

const DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant that can answer questions and help with tasks'

interface ChatRequestBody {
  messages: UIMessage[]
  model: string
  webSearch?: boolean
}

export default defineLazyEventHandler(async () =>
  defineEventHandler(async (event) => {
    const { messages, model, webSearch = false } = await readBody<ChatRequestBody>(event)

    if (!Array.isArray(messages) || messages.length === 0) {
      throw createError({
        statusCode: 400,
        statusMessage: 'Missing messages payload',
      })
    }

    if (!model) {
      throw createError({
        statusCode: 400,
        statusMessage: 'Missing model value',
      })
    }

    const config = useRuntimeConfig(event)

    // Initialize providers
    const deepseek = createDeepSeek({
      apiKey: config.deepseekApiKey,
    })

    // Select the correct provider and model based on model name
    let modelInstance
    if (webSearch) {
      // Perplexity model (if needed)
      modelInstance = 'perplexity/sonar'
    }
    else if (model.startsWith('deepseek/')) {
      // DeepSeek model
      const modelName = model.replace('deepseek/', '')
      modelInstance = deepseek(modelName)
    }
    else {
      throw createError({
        statusCode: 400,
        statusMessage: `Unsupported model: ${model}`,
      })
    }

    const result = streamText({
      model: modelInstance,
      messages: convertToModelMessages(messages),
      system: DEFAULT_SYSTEM_PROMPT,
    })

    return result.toUIMessageStreamResponse({
      sendSources: true,
      sendReasoning: true,
    })
  }),
)
