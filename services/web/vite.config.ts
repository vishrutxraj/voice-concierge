import { defineConfig } from "vitest/config";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// The gateway serves the built app at /app (see services/gateway/app/main.py),
// so asset URLs must be rooted there. In dev, Vite proxies the gateway's
// routes so the browser still sees a single origin -- same as production, which
// is why the client never hardcodes a host or needs CORS.
const GATEWAY = process.env.GATEWAY_URL ?? "http://127.0.0.1:8000";

export default defineConfig({
  base: "/app/",
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      "/ws": { target: GATEWAY, ws: true },
      "/api": GATEWAY,
      "/call": GATEWAY,
      "/health": GATEWAY,
    },
  },
  test: { environment: "node" },
});
