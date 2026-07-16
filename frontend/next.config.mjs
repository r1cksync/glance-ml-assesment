/**
 * Static-export config: the app builds to `out/` and is uploaded to AWS
 * Amplify manual hosting. No server runtime — every page is a client SPA
 * page that talks to the FastAPI backend at NEXT_PUBLIC_API_URL.
 * @type {import('next').NextConfig}
 */
const nextConfig = {
  output: "export",
  images: { unoptimized: true },
  trailingSlash: true,
};

export default nextConfig;
