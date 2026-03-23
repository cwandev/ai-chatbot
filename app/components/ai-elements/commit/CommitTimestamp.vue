<script setup lang="ts">
import type { HTMLAttributes } from 'vue'
import { computed } from 'vue'
import { cn } from '@/lib/utils'

interface Props extends /* @vue-ignore */ HTMLAttributes {
  date: Date
  class?: HTMLAttributes['class']
}

const props = defineProps<Props>()

const formatted = computed(() => {
  return new Intl.RelativeTimeFormat('en', {
    numeric: 'auto',
  }).format(
    Math.round((props.date.getTime() - Date.now()) / (1000 * 60 * 60 * 24)),
    'day',
  )
})
</script>

<template>
  <time
    :class="cn('text-xs', props.class)"
    :datetime="props.date.toISOString()"
    v-bind="$attrs"
  >
    <slot>{{ formatted }}</slot>
  </time>
</template>
