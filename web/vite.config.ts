import { defineConfig } from "vite";

export default defineConfig({
  // Allow serving the WASM file with correct MIME type
  assetsInclude: ["**/*.wasm"],
  server: {
    headers: {
      // Required for SharedArrayBuffer / WebGPU in some browsers
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  optimizeDeps: {
    // Don't bundle the WASM package — it is loaded dynamically
    exclude: ["webgpu-fmidx"],
  },
});
