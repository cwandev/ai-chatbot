<script setup lang="ts">
import type { HTMLAttributes } from 'vue'
import { computed } from 'vue'
import { cn } from '@/lib/utils'

const props = defineProps<{
  class?: HTMLAttributes['class']
}>()

const classes = computed(() => cn('text-muted-foreground text-sm', props.class))
</script>

<template>
  <p
    :class="classes"
    v-bind="$attrs"
  >
    <slot />
  </p>
</template>
