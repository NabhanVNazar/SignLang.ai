import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const apiKey = process.env.SIGNLEARN_API_KEY;
const apiTarget = process.env.SIGNLEARN_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: apiTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
        configure: (proxy) => {
          proxy.on("proxyReq", (proxyRequest, request) => {
            const requestPath = request.url ?? "";
            if (requestPath === "/api/health" || requestPath === "/health") {
              return;
            }
            if (apiKey) {
              proxyRequest.setHeader("Authorization", `Bearer ${apiKey}`);
            }
          });
        },
      },
    },
  },
});
