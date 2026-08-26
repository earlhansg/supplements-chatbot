import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // node-redis opens raw TCP sockets and must run as a real Node module rather
  // than being bundled into the server build.
  serverExternalPackages: ["redis"],
};

export default nextConfig;
