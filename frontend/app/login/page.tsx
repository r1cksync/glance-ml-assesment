"use client";

/**
 * Sign-in / registration page with a one-click demo account. The demo button
 * tries to log in with the demo credentials and transparently registers the
 * account first if it does not exist yet.
 */

import { useRouter } from "next/navigation";
import { useEffect, useState, type FormEvent } from "react";

import Spinner from "@/components/Spinner";
import { errorMessage } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";

const DEMO_EMAIL = "demo@fashionretrieval.dev";
const DEMO_PASSWORD = "DemoPass123!";

type Tab = "login" | "register";

export default function LoginPage() {
  const { user, ready, login, register } = useAuth();
  const router = useRouter();

  const [tab, setTab] = useState<Tab>("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [demoLoading, setDemoLoading] = useState(false);

  // already signed in? straight to search.
  useEffect(() => {
    if (ready && user) router.replace("/search");
  }, [ready, user, router]);

  const switchTab = (t: Tab) => {
    setTab(t);
    setError(null);
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (submitting) return;
    setError(null);
    setSubmitting(true);
    try {
      if (tab === "login") {
        await login(email.trim(), password);
      } else {
        await register(email.trim(), password);
      }
      router.push("/search");
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setSubmitting(false);
    }
  };

  const handleDemo = async () => {
    if (demoLoading) return;
    setError(null);
    setDemoLoading(true);
    try {
      try {
        await login(DEMO_EMAIL, DEMO_PASSWORD);
      } catch {
        // account may not exist on this deployment yet — create it, then sign in
        await register(DEMO_EMAIL, DEMO_PASSWORD);
      }
      router.push("/search");
    } catch (err) {
      setError(errorMessage(err));
    } finally {
      setDemoLoading(false);
    }
  };

  return (
    <div className="mx-auto flex min-h-[calc(100vh-8rem)] max-w-md flex-col justify-center py-10">
      <div className="mb-8 text-center">
        <h1 className="text-2xl font-semibold tracking-tight text-white">
          Fashion Retrieval
        </h1>
        <p className="mt-2 text-sm text-zinc-500">
          Search a fashion image index by garments, colors, scenes and style.
        </p>
      </div>

      <div className="card p-6 sm:p-8">
        {/* tabs */}
        <div className="mb-6 grid grid-cols-2 gap-1 rounded-xl bg-zinc-950/70 p-1">
          {(["login", "register"] as const).map((t) => (
            <button
              key={t}
              type="button"
              onClick={() => switchTab(t)}
              className={`rounded-lg py-2 text-sm font-medium transition-colors ${
                tab === t
                  ? "bg-zinc-800 text-white shadow-sm"
                  : "text-zinc-500 hover:text-zinc-300"
              }`}
            >
              {t === "login" ? "Sign in" : "Create account"}
            </button>
          ))}
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label
              htmlFor="email"
              className="mb-1.5 block text-xs font-medium uppercase tracking-wider text-zinc-500"
            >
              Email
            </label>
            <input
              id="email"
              type="email"
              required
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@example.com"
              className="input"
            />
          </div>
          <div>
            <label
              htmlFor="password"
              className="mb-1.5 block text-xs font-medium uppercase tracking-wider text-zinc-500"
            >
              Password
            </label>
            <input
              id="password"
              type="password"
              required
              minLength={8}
              autoComplete={tab === "login" ? "current-password" : "new-password"}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder="At least 8 characters"
              className="input"
            />
          </div>

          {error && (
            <p
              role="alert"
              className="rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-300"
            >
              {error}
            </p>
          )}

          <button type="submit" disabled={submitting} className="btn-primary w-full">
            {submitting && <Spinner size={16} className="text-white" />}
            {tab === "login" ? "Sign in" : "Create account"}
          </button>
        </form>

        <div className="my-6 flex items-center gap-3">
          <span className="h-px flex-1 bg-zinc-800" />
          <span className="text-[11px] uppercase tracking-wider text-zinc-600">
            or
          </span>
          <span className="h-px flex-1 bg-zinc-800" />
        </div>

        <button
          type="button"
          onClick={handleDemo}
          disabled={demoLoading}
          className="w-full rounded-xl border border-violet-500/40 bg-violet-500/10 px-4 py-3 text-sm font-medium text-violet-300 transition-colors hover:bg-violet-500/20 focus:outline-none focus:ring-2 focus:ring-violet-500/40 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <span className="flex items-center justify-center gap-2">
            {demoLoading && <Spinner size={16} />}
            Try the demo account
          </span>
          <span className="mt-1 block text-[11px] font-normal text-violet-400/70">
            {DEMO_EMAIL}
          </span>
        </button>
      </div>
    </div>
  );
}
