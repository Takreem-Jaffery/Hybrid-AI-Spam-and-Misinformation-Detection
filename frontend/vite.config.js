import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// In development every API call is forwarded to the FastAPI server, so the browser only talks
// to one origin (no CORS needed). Override the target with API_PROXY_TARGET if needed.
const target = process.env.API_PROXY_TARGET || 'http://127.0.0.1:8000'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/predict': { target, changeOrigin: true },
      '/article': { target, changeOrigin: true },
      '/advice': { target, changeOrigin: true },
      '/health': { target, changeOrigin: true },
    },
  },
})
