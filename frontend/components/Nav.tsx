"use client";

/**
 * Top navigation: brand, Search / Admin links (role-gated), API health dot
 * (polls /healthz), signed-in email and logout.
 */

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { healthz } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";

type Health = "unknown" | "ok" | "down";

const HEALTH_DOT: Record<Health, string> = {
  unknown: "bg-zinc-500",
  ok: "bg-emerald-400",
  down: "bg-red-500",
};

const HEALTH_LABEL: Record<Health, string> = {
  unknown: "API status unknown",
  ok: "API healthy",
  down: "API unreachable",
};

function NavLink({ href, label }: { href: string; label: string }) {
  const pathname = usePathname();
  const active = pathname === href || pathname.startsWith(`${href}/`);
  return (
    <Link
      href={href}
      className={`rounded-lg px-3 py-1.5 text-sm transition-colors ${
        active
          ? "bg-zinc-800/80 font-medium text-white"
          : "text-zinc-400 hover:bg-zinc-900 hover:text-zinc-100"
      }`}
    >
      {label}
    </Link>
  );
}

export default function Nav() {
  const { user, role, logout } = useAuth();
  const router = useRouter();
  const [health, setHealth] = useState<Health>("unknown");
  const [signingOut, setSigningOut] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      const ok = await healthz();
      if (!cancelled) setHealth(ok ? "ok" : "down");
    };
    void check();
    const timer = window.setInterval(check, 20_000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  const handleLogout = async () => {
    setSigningOut(true);
    try {
      await logout();
      router.push("/login");
    } finally {
      setSigningOut(false);
    }
  };

  return (
    <header className="sticky top-0 z-40 border-b border-zinc-800/70 bg-zinc-950/80 backdrop-blur">
      <div className="mx-auto flex h-14 w-full max-w-7xl items-center gap-4 px-4 sm:px-6 lg:px-8">
        <Link href="/" className="flex items-center gap-2.5">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-gradient-to-br from-violet-500 to-violet-700 text-[11px] font-bold text-white shadow-lg shadow-violet-900/40">
            FR
          </span>
          <span className="hidden text-sm font-semibold tracking-tight text-zinc-100 sm:block">
            Fashion Retrieval
          </span>
        </Link>

        {user && (
          <nav className="flex items-center gap-1">
            <NavLink href="/search" label="Search" />
            {role === "admin" && <NavLink href="/admin" label="Admin" />}
          </nav>
        )}

        <div className="ml-auto flex items-center gap-3">
          <span
            className="flex items-center gap-1.5 text-[11px] text-zinc-500"
            title={HEALTH_LABEL[health]}
          >
            <span className="relative flex h-2 w-2">
              {health === "ok" && (
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-emerald-400 opacity-40" />
              )}
              <span
                className={`relative inline-flex h-2 w-2 rounded-full ${HEALTH_DOT[health]}`}
              />
            </span>
            API
          </span>

          {user ? (
            <>
              <span
                className="hidden max-w-48 truncate text-xs text-zinc-500 md:block"
                title={`${user.email} (${user.role})`}
              >
                {user.email}
              </span>
              <button
                type="button"
                onClick={handleLogout}
                disabled={signingOut}
                className="rounded-lg border border-zinc-800 px-3 py-1.5 text-xs font-medium text-zinc-400 transition-colors hover:border-zinc-700 hover:text-zinc-100 disabled:opacity-50"
              >
                {signingOut ? "Signing out…" : "Sign out"}
              </button>
            </>
          ) : (
            <Link
              href="/login"
              className="rounded-lg border border-violet-500/40 bg-violet-500/10 px-3 py-1.5 text-xs font-medium text-violet-300 transition-colors hover:bg-violet-500/20"
            >
              Sign in
            </Link>
          )}
        </div>
      </div>
    </header>
  );
}
