import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: { port: 8000, host: true },
  build: {
    outDir: 'dist',
    sourcemap: true,
    // the map, the charts and the app itself change at different rates
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (/node_modules\/(react|react-dom|react-router|react-router-dom|scheduler)\//.test(id)) return 'react';
          if (id.includes('node_modules/leaflet/')) return 'leaflet';
          if (id.includes('node_modules/echarts/') || id.includes('node_modules/zrender/')) return 'echarts';
          return undefined;
        },
      },
    },
  },
});
