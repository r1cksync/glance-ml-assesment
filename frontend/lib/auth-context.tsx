"use client";

/**
 * Auth context: holds the signed-in user (email + role) in React state while
 * the access token itself lives in lib/api.ts module memory. On mount it
 * attempts refresh -> me to silently restore the session from the httpOnly
 * refresh cookie.
 */

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import * as api from "@/lib/api";

export interface AuthUser {
  email: string;
  role: string;
}

interface AuthContextValue {
  user: AuthUser | null;
  role: string | null;
  /** true once the initial session-restore attempt has settled */
  ready: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const restored = await api.refresh();
        if (restored) {
          const who = await api.me();
          if (!cancelled && who.email) {
            setUser({ email: who.email, role: who.role });
          }
        }
      } catch {
        /* no session to restore */
      }
      if (!cancelled) setReady(true);
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  const login = useCallback(async (email: string, password: string) => {
    const { role } = await api.login(email, password);
    let resolved: AuthUser = { email, role };
    try {
      const who = await api.me();
      resolved = { email: who.email || email, role: who.role || role };
    } catch {
      /* /auth/me is best-effort — the login itself succeeded */
    }
    setUser(resolved);
  }, []);

  const register = useCallback(
    async (email: string, password: string) => {
      await api.register(email, password);
      await login(email, password);
    },
    [login],
  );

  const logout = useCallback(async () => {
    try {
      await api.logout();
    } catch {
      /* cookie may already be gone; clear local state regardless */
    }
    setUser(null);
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({ user, role: user?.role ?? null, ready, login, register, logout }),
    [user, ready, login, register, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used inside <AuthProvider>");
  return ctx;
}
