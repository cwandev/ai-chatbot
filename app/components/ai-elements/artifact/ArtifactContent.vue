<script setup lang="ts">
import type { HTMLAttributes } from 'vue'
import { computed } from 'vue'
import { cn } from '@/lib/utils'

const props = defineProps<{
  class?: HTMLAttributes['class']
}>()

const classes = computed(() => cn('flex-1 overflow-auto p-4', props.class))
</script>

<template>
  <div
    :class="classes"
    v-bind="$attrs"
  >
    <slot />
  </div>
</template>
