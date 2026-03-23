import type { UIMessage } from 'ai'
import { convertToModelMessages, createGateway, streamText } from 'ai'
import { createError, readBody } from 'h3'

export const maxDuration = 30

const DEFAULT_SYSTEM_PROMPT = 'You are a helpful assistant that can answer questions and help with tasks'
const DEFAULT_MODEL = 'openai/gpt-4o'

interface ChatRequestBody {
  messages: UIMessage[]
  model?: string
  webSearch?: boolean
}

export default defineLazyEventHandler(async () => {
  const apiKey = useRuntimeConfig().aiGatewayApiKey

  if (!apiKey) {
    throw createError({
      statusCode: 500,
      statusMessage: 'Missing AI Gateway API key',
    })
  }

  const gateway = createGateway({
    apiKey,
  })

  return defineEventHandler(async (event) => {
    const { messages, model, webSearch = false } = await readBody<ChatRequestBody>(event)

    if (!Array.isArray(messages) || messages.length === 0) {
      throw createError({
        statusCode: 400,
        statusMessage: 'Missing messages payload',
      })
    }

    const selectedModel = webSearch ? 'perplexity/sonar' : (model || DEFAULT_MODEL)

    const result = streamText({
      model: gateway(selectedModel),
      messages: await convertToModelMessages(messages),
      system: DEFAULT_SYSTEM_PROMPT,
    })

    return result.toUIMessageStreamResponse({
      sendSources: true,
      sendReasoning: true,
    })
  })
})
