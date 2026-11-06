import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import tailwindcss from '@tailwindcss/vite';

// The application config proxies /neon to the backend. This separate config
// serves the real built iframe from public/neon and never contacts that API.
export default defineConfig({
  root: fileURLToPath(new URL('../../', import.meta.url)),
  publicDir: 'public',
  plugins: [react(), tailwindcss(), {
    name: 'neon-backend-asset-alias',
    configureServer(server) {
      // The real backend also mounts these assets at the origin root.
      server.middlewares.use((request, _response, next) => {
        if (request.url?.startsWith('/Neon-gh/')) {
          request.url = `/neon${request.url}`;
        }
        next();
      });
    },
  }],
  server: {
    host: '127.0.0.1',
    port: 5178,
    strictPort: true,
    proxy: {},
    hmr: false,
  },
});
