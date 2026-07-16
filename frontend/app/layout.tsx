import type { Metadata } from "next";
import { Inter } from "next/font/google";

import "./globals.css";

import Nav from "@/components/Nav";
import { ToastProvider } from "@/components/Toast";
import { AuthProvider } from "@/lib/auth-context";

const inter = Inter({ subsets: ["latin"], variable: "--font-inter" });

export const metadata: Metadata = {
  title: "Fashion Retrieval",
  description:
    "Multimodal fashion image retrieval — garment regions, scene context, and structural attribute binding.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className={inter.variable}>
      <body className="min-h-screen font-sans antialiased">
        <AuthProvider>
          <ToastProvider>
            <Nav />
            <main className="mx-auto w-full max-w-7xl px-4 pb-20 sm:px-6 lg:px-8">
              {children}
            </main>
          </ToastProvider>
        </AuthProvider>
      </body>
    </html>
  );
}
