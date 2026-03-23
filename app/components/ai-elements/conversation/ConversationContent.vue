<script setup lang="ts">
import type { HTMLAttributes } from 'vue'
import { computed } from 'vue'
import { cn } from '@/lib/utils'

interface Props {
  class?: HTMLAttributes['class']
}

const props = defineProps<Props>()

const classes = computed(() => cn(
  'flex flex-col gap-8 p-4',
  props.class,
))
</script>

<template>
  <div :class="classes">
    <slot />
  </div>
</template>
