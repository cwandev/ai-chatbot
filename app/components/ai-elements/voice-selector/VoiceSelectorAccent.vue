<script setup lang="ts">
import type { HTMLAttributes } from 'vue'
import { computed } from 'vue'
import { cn } from '@/lib/utils'

type Accent
  = | 'american'
    | 'british'
    | 'australian'
    | 'canadian'
    | 'irish'
    | 'scottish'
    | 'indian'
    | 'south-african'
    | 'new-zealand'
    | 'spanish'
    | 'french'
    | 'german'
    | 'italian'
    | 'portuguese'
    | 'brazilian'
    | 'mexican'
    | 'argentinian'
    | 'japanese'
    | 'chinese'
    | 'korean'
    | 'russian'
    | 'arabic'
    | 'dutch'
    | 'swedish'
    | 'norwegian'
    | 'danish'
    | 'finnish'
    | 'polish'
    | 'turkish'
    | 'greek'
    | (string & {})

interface Props extends /* @vue-ignore */ HTMLAttributes {
  class?: HTMLAttributes['class']
  value?: Accent
}

const props = defineProps<Props>()

const emoji = computed(() => {
  switch (props.value) {
    case 'american':
      return '🇺🇸'
    case 'british':
      return '🇬🇧'
    case 'australian':
      return '🇦🇺'
    case 'canadian':
      return '🇨🇦'
    case 'irish':
      return '🇮🇪'
    case 'scottish':
      return '🏴󠁧󠁢󠁳󠁣󠁴󠁿'
    case 'indian':
      return '🇮🇳'
    case 'south-african':
      return '🇿🇦'
    case 'new-zealand':
      return '🇳🇿'
    case 'spanish':
      return '🇪🇸'
    case 'french':
      return '🇫🇷'
    case 'german':
      return '🇩🇪'
    case 'italian':
      return '🇮🇹'
    case 'portuguese':
      return '🇵🇹'
    case 'brazilian':
      return '🇧🇷'
    case 'mexican':
      return '🇲🇽'
    case 'argentinian':
      return '🇦🇷'
    case 'japanese':
      return '🇯🇵'
    case 'chinese':
      return '🇨🇳'
    case 'korean':
      return '🇰🇷'
    case 'russian':
      return '🇷🇺'
    case 'arabic':
      return '🇸🇦'
    case 'dutch':
      return '🇳🇱'
    case 'swedish':
      return '🇸🇪'
    case 'norwegian':
      return '🇳🇴'
    case 'danish':
      return '🇩🇰'
    case 'finnish':
      return '🇫🇮'
    case 'polish':
      return '🇵🇱'
    case 'turkish':
      return '🇹🇷'
    case 'greek':
      return '🇬🇷'
    default:
      return null
  }
})
</script>

<template>
  <span
    :class="cn('text-muted-foreground text-xs', props.class)"
  >
    <slot>
      {{ emoji }}
    </slot>
  </span>
</template>
