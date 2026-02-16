import process from 'node:process'
import tailwindcss from '@tailwindcss/vite'

// https://nuxt.com/docs/api/configuration/nuxt-config
export default defineNuxtConfig({
  compatibilityDate: '2025-07-15',
  devtools: {
    enabled: false,
  },
  css: ['~/assets/css/tailwind.css'],

  vite: {
    plugins: [
      // @ts-expect-error Vite plugin type is not properly typed
      tailwindcss(),
    ],
  },

  runtimeConfig: {
    aiGatewayApiKey: process.env.NUXT_AI_GATEWAY_API_KEY,
  },

  modules: [
    'shadcn-nuxt',
    '@nuxtjs/color-mode',
  ],

  colorMode: {
    classSuffix: '',
  },

  shadcn: {
    /**
     * Prefix for all the imported component.
     * @default "Ui"
     */
    prefix: '',
    /**
     * Directory that the component lives in.
     * Will respect the Nuxt aliases.
     * @link https://nuxt.com/docs/api/nuxt-config#alias
     * @default "@/components/ui"
     */
    componentDir: [
      '@/components/ui',
      // AI elements
      {
        path: '@/components/ai-elements',
        prefix: '',
      },
    ],
  },
})
