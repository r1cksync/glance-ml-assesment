"use client";

/** Root route: redirect to /search when authenticated, otherwise /login. */

import { useRouter } from "next/navigation";
import { useEffect } from "react";

import Spinner from "@/components/Spinner";
import { useAuth } from "@/lib/auth-context";

export default function HomePage() {
  const { user, ready } = useAuth();
  const router = useRouter();

  useEffect(() => {
    if (!ready) return;
    router.replace(user ? "/search" : "/login");
  }, [ready, user, router]);

  return (
    <div className="flex min-h-[60vh] items-center justify-center">
      <Spinner size={28} />
    </div>
  );
}
