import { defineConfig } from 'vite';

export default defineConfig({
  // Read Vite env files from the repo root; the page entry stays in src/app.
  envDir: '../..',
  root: 'src/app',
  // Assets (models / textures / environment maps / gallery covers) are published
  // as-is to the site root
  publicDir: '../../assets',
  server: {
    port: 5173,
    open: true,
  },
  build: {
    outDir: '../../dist/app',
    // The output dir is outside the project root (dist/app), so vite won't empty it
    // by default and old chunks pile up
    emptyOutDir: true,
  },
  // The render Worker and main thread share core / engine modules, which produces a shared chunk —
  // only the ES format is supported (the default iife cannot code-split)
  worker: {
    format: 'es',
  },
});
