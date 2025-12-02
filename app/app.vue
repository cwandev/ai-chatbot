<script setup lang="ts">
import type { ChatStatus, SourceUrlUIPart, UIMessage } from 'ai'
import type { PromptInputMessage } from '@/components/ai-elements/prompt-input'
import { Chat } from '@ai-sdk/vue'
import { CopyIcon, GlobeIcon, RefreshCcwIcon } from 'lucide-vue-next'
import { computed, ref } from 'vue'
import { Conversation, ConversationContent, ConversationScrollButton } from '@/components/ai-elements/conversation'
import { Loader } from '@/components/ai-elements/loader'
import { Message, MessageAction, MessageActions, MessageContent, MessageResponse } from '@/components/ai-elements/message'
import {
  PromptInput,
  PromptInputActionAddAttachments,
  PromptInputActionMenu,
  PromptInputActionMenuContent,
  PromptInputActionMenuTrigger,
  PromptInputAttachment,
  PromptInputAttachments,
  PromptInputBody,
  PromptInputButton,
  PromptInputFooter,
  PromptInputHeader,
  PromptInputSelect,
  PromptInputSelectContent,
  PromptInputSelectItem,
  PromptInputSelectTrigger,
  PromptInputSelectValue,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
  usePromptInputProvider,
} from '@/components/ai-elements/prompt-input'
import { Reasoning, ReasoningContent, ReasoningTrigger } from '@/components/ai-elements/reasoning'
import { Source, Sources, SourcesContent, SourcesTrigger } from '@/components/ai-elements/sources'

const models = [
  { name: 'Deepseek Chat', value: 'deepseek/deepseek-chat' },
  { name: 'Deepseek R1', value: 'deepseek/deepseek-reasoner' },
] as const

const chat = new Chat({})
const model = ref(models[0].value)
const webSearch = ref(false)

const status = computed<ChatStatus>(() => chat.status)
const messages = computed(() => chat.messages)
const lastMessageId = computed(() => messages.value.at(-1)?.id ?? null)
const lastAssistantMessageId = computed(() => {
  for (let index = messages.value.length - 1; index >= 0; index -= 1) {
    const current = messages.value[index]
    if (current && current.role === 'assistant')
      return current.id
  }
  return null
})

async function handleSubmit(message: PromptInputMessage) {
  const hasText = Boolean(message.text?.trim())
  const hasAttachments = Boolean(message.files?.length)

  if (!hasText && !hasAttachments)
    return

  try {
    await chat.sendMessage(
      {
        text: hasText ? message.text : 'Sent with attachments',
        files: hasAttachments ? message.files : undefined,
      },
      {
        body: {
          model: model.value,
          webSearch: webSearch.value,
        },
      },
    )
  }
  catch (error) {
    console.error('Failed to send message', error)
  }
}

function handlePromptError(error: { code: string, message: string }) {
  console.error(`Input error (${error.code})`, error.message)
}

const promptInput = usePromptInputProvider({
  onSubmit: handleSubmit,
  onError: handlePromptError,
})

const hasPendingInput = computed(() => {
  return Boolean(promptInput.textInput.value.trim()) || promptInput.files.value.length > 0
})

const submitDisabled = computed(() => !hasPendingInput.value && !status.value)

function getSourceUrlParts(message: UIMessage) {
  return message.parts.filter((part): part is SourceUrlUIPart => part.type === 'source-url')
}

function shouldShowActions(message: UIMessage, partIndex: number) {
  if (message.role !== 'assistant')
    return false
  if (lastAssistantMessageId.value !== message.id)
    return false
  return isLastTextPart(message, partIndex)
}

function isLastTextPart(message: UIMessage, partIndex: number) {
  for (let index = partIndex + 1; index < message.parts.length; index += 1) {
    const nextPart = message.parts[index]
    if (nextPart && nextPart.type === 'text')
      return false
  }
  return true
}

function isReasoningStreaming(message: UIMessage, partIndex: number) {
  return status.value === 'streaming'
    && message.id === lastMessageId.value
    && partIndex === message.parts.length - 1
}

function toggleWebSearch() {
  webSearch.value = !webSearch.value
}

async function copyToClipboard(text: string) {
  if (!text)
    return

  if (typeof navigator === 'undefined' || !navigator.clipboard)
    return

  try {
    await navigator.clipboard.writeText(text)
  }
  catch (error) {
    console.error('Failed to copy to clipboard', error)
  }
}

function handleRegenerate() {
  chat.regenerate({
    body: {
      model: model.value,
      webSearch: webSearch.value,
    },
  })
}
</script>

<template>
  <div class="min-h-screen bg-background">
    <NuxtRouteAnnouncer />
    <div class="relative mx-auto size-full h-screen max-w-4xl p-6">
      <div class="flex h-full flex-col">
        <Conversation class="h-full">
          <ConversationContent>
            <div
              v-for="message in messages"
              :key="message.id"
            >
              <Sources
                v-if="message.role === 'assistant' && getSourceUrlParts(message).length > 0"
              >
                <SourcesTrigger :count="getSourceUrlParts(message).length" />
                <SourcesContent
                  v-for="(source, index) in getSourceUrlParts(message)"
                  :key="`${message.id}-source-${index}`"
                >
                  <Source
                    :href="source.url"
                    :title="source.title ?? source.url"
                  />
                </SourcesContent>
              </Sources>

              <template
                v-for="(part, partIndex) in message.parts"
                :key="`${message.id}-${partIndex}`"
              >
                <Message
                  v-if="part.type === 'text'"
                  :from="message.role"
                >
                  <div>
                    <MessageContent>
                      <MessageResponse :content="part.text" />
                    </MessageContent>

                    <MessageActions v-if="shouldShowActions(message, partIndex)">
                      <MessageAction
                        label="Retry"
                        @click="handleRegenerate"
                      >
                        <RefreshCcwIcon class="size-3" />
                      </MessageAction>
                      <MessageAction
                        label="Copy"
                        @click="copyToClipboard(part.text)"
                      >
                        <CopyIcon class="size-3" />
                      </MessageAction>
                    </MessageActions>
                  </div>
                </Message>

                <Reasoning
                  v-else-if="part.type === 'reasoning'"
                  class="w-full"
                  :is-streaming="isReasoningStreaming(message, partIndex)"
                >
                  <ReasoningTrigger />
                  <ReasoningContent :content="part.text" />
                </Reasoning>
              </template>
            </div>

            <Loader v-if="status === 'submitted'" class="mx-auto" />
          </ConversationContent>

          <ConversationScrollButton />
        </Conversation>

        <PromptInput class="mt-4" global-drop multiple>
          <PromptInputHeader>
            <PromptInputAttachments>
              <template #default="{ file }">
                <PromptInputAttachment :file="file" />
              </template>
            </PromptInputAttachments>
          </PromptInputHeader>

          <PromptInputBody>
            <PromptInputTextarea />
          </PromptInputBody>

          <PromptInputFooter>
            <PromptInputTools>
              <PromptInputActionMenu>
                <PromptInputActionMenuTrigger />
                <PromptInputActionMenuContent>
                  <PromptInputActionAddAttachments />
                </PromptInputActionMenuContent>
              </PromptInputActionMenu>

              <PromptInputButton
                :variant="webSearch ? 'default' : 'ghost'"
                @click="toggleWebSearch"
              >
                <GlobeIcon class="size-4" />
                <span>Search</span>
              </PromptInputButton>

              <PromptInputSelect v-model="model">
                <PromptInputSelectTrigger>
                  <PromptInputSelectValue />
                </PromptInputSelectTrigger>
                <PromptInputSelectContent>
                  <PromptInputSelectItem
                    v-for="item in models"
                    :key="item.value"
                    :value="item.value"
                  >
                    {{ item.name }}
                  </PromptInputSelectItem>
                </PromptInputSelectContent>
              </PromptInputSelect>
            </PromptInputTools>

            <PromptInputSubmit
              :disabled="submitDisabled"
              :status="status"
            />
          </PromptInputFooter>
        </PromptInput>
      </div>
    </div>
  </div>
</template>
