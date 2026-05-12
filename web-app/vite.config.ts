import { defineConfig } from 'vite'
import type { Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import fs from 'node:fs'
import path from 'node:path'
import os from 'node:os'

function discoveryPlugin(): Plugin {
  const registryPath = path.join(os.homedir(), '.ghostlobby', 'sessions.json')

  return {
    name: 'ghostlobby-discovery',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        if (req.url === '/api/discover') {
          res.setHeader('Content-Type', 'application/json')
          try {
            res.end(fs.readFileSync(registryPath, 'utf-8'))
          } catch {
            res.end('[]')
          }
          return
        }
        next()
      })
    },
  }
}

export default defineConfig({
  plugins: [discoveryPlugin(), react()],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
  },
  server: {
    port: 5173,
    proxy: {
      '/proxy': {
        target: 'http://localhost:3000',
        changeOrigin: true,
        ws: true,
        router: (req: { url?: string }) => {
          const match = req.url?.match(/^\/proxy\/(\d+)/)
          if (match) return `http://localhost:${match[1]}`
          return 'http://localhost:3000'
        },
        rewrite: (path: string) => path.replace(/^\/proxy\/\d+/, ''),
      },
    },
  },
})
