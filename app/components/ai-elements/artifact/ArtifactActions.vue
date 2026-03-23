<script setup lang="ts">
import type { HTMLAttributes } from 'vue'
import { computed } from 'vue'
import { cn } from '@/lib/utils'

const props = defineProps<{
  class?: HTMLAttributes['class']
}>()
const classes = computed(() => cn('flex items-center gap-1', props.class))
</script>

<template>
  <div
    :class="classes"
    v-bind="$attrs"
  >
    <slot />
  </div>
</template>
