"use client";

/**
 * Admin console (role-gated): index stats cards, attribute distributions as
 * CSS horizontal bar charts (color rows keep their canonical swatch), the
 * engine configuration tables, and a confirm-guarded "Reload index from S3"
 * action hitting POST /admin/reindex.
 *
 * The stats payload shape is rendered tolerantly: scalar fields become stat
 * cards, nested objects become key/value tables — so backend additions show
 * up without frontend changes.
 */

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import Spinner from "@/components/Spinner";
import { useToast } from "@/components/Toast";
import * as api from "@/lib/api";
import { errorMessage } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";
import { colorHex } from "@/lib/colors";

// ── tolerant payload helpers ─────────────────────────────────────────────────

function prettifyKey(key: string): string {
  return key.replace(/[_-]+/g, " ").trim();
}

function formatValue(v: unknown): string {
  if (typeof v === "number") {
    return Number.isInteger(v) ? v.toLocaleString() : v.toFixed(4);
  }
  if (v === null || v === undefined) return "—";
  if (typeof v === "boolean") return v ? "yes" : "no";
  if (Array.isArray(v)) return v.map((x) => String(x)).join(", ");
  if (typeof v === "object") return JSON.stringify(v);
  return String(v);
}

/** Split a stats payload into scalar cards and nested key/value tables. */
function splitStats(stats: Record<string, unknown>): {
  cards: [string, string][];
  tables: [string, [string, string][]][];
} {
  const cards: [string, string][] = [];
  const tables: [string, [string, string][]][] = [];
  for (const [key, value] of Object.entries(stats)) {
    if (value !== null && typeof value === "object" && !Array.isArray(value)) {
      const rows = Object.entries(value as Record<string, unknown>).map(
        ([k, v]) => [prettifyKey(k), formatValue(v)] as [string, string],
      );
      if (rows.length) tables.push([prettifyKey(key), rows]);
    } else {
      cards.push([prettifyKey(key), formatValue(value)]);
    }
  }
  return { cards, tables };
}

/** Normalize a distribution section ({label: count} map OR [{value,count}] list). */
function normalizeDist(section: unknown): [string, number][] {
  if (!section) return [];
  let entries: [string, number][] = [];
  if (Array.isArray(section)) {
    entries = section
      .map((item): [string, number] | null => {
        if (item === null || typeof item !== "object") return null;
        const o = item as Record<string, unknown>;
        const label = o.value ?? o.name ?? o.label ?? o.key;
        const count = o.count ?? o.n ?? o.total;
        if (typeof label !== "string" || typeof count !== "number") return null;
        return [label, count];
      })
      .filter((e): e is [string, number] => e !== null);
  } else if (typeof section === "object") {
    entries = Object.entries(section as Record<string, unknown>)
      .filter((e): e is [string, number] => typeof e[1] === "number")
      .map(([label, count]) => [label, count]);
  }
  return entries.sort((a, b) => b[1] - a[1]);
}

function pickSection(
  dist: Record<string, unknown>,
  keys: string[],
): [string, number][] {
  for (const key of keys) {
    if (key in dist) {
      const rows = normalizeDist(dist[key]);
      if (rows.length) return rows;
    }
  }
  return [];
}

// ── chart ────────────────────────────────────────────────────────────────────

function DistChart({
  title,
  rows,
  colorFor,
  swatch = false,
}: {
  title: string;
  rows: [string, number][];
  colorFor: (label: string) => string;
  swatch?: boolean;
}) {
  if (rows.length === 0) return null;
  const max = Math.max(...rows.map(([, c]) => c), 1);
  return (
    <div className="card p-5">
      <h3 className="mb-4 text-xs font-semibold uppercase tracking-wider text-zinc-500">
        {title}
      </h3>
      <div className="space-y-2">
        {rows.map(([label, count]) => (
          <div key={label} className="flex items-center gap-2.5">
            <span
              className="flex w-24 shrink-0 items-center gap-1.5 truncate text-xs text-zinc-400"
              title={label}
            >
              {swatch && (
                <span
                  className="h-2.5 w-2.5 shrink-0 rounded-full ring-1 ring-white/20"
                  style={{ background: colorFor(label) }}
                />
              )}
              <span className="truncate">{label}</span>
            </span>
            <div className="h-2 flex-1 overflow-hidden rounded-full bg-zinc-800/80">
              <div
                className="h-full rounded-full transition-[width] duration-700 ease-out"
                style={{
                  width: `${(count / max) * 100}%`,
                  background: colorFor(label),
                }}
              />
            </div>
            <span className="w-10 shrink-0 text-right text-xs tabular-nums text-zinc-500">
              {count.toLocaleString()}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── page ─────────────────────────────────────────────────────────────────────

export default function AdminPage() {
  const { user, role, ready } = useAuth();
  const router = useRouter();
  const toast = useToast();

  const [stats, setStats] = useState<Record<string, unknown> | null>(null);
  const [dist, setDist] = useState<Record<string, unknown> | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [reindexing, setReindexing] = useState(false);

  const isAdmin = role === "admin";

  useEffect(() => {
    if (ready && !user) router.replace("/login");
  }, [ready, user, router]);

  const load = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    const [statsRes, distRes] = await Promise.allSettled([
      api.adminStats(),
      api.adminDistribution(),
    ]);
    if (statsRes.status === "fulfilled") setStats(statsRes.value);
    if (distRes.status === "fulfilled") setDist(distRes.value);
    if (statsRes.status === "rejected" && distRes.status === "rejected") {
      setLoadError(errorMessage(statsRes.reason));
    }
    setLoading(false);
  }, []);

  useEffect(() => {
    if (ready && user && isAdmin) void load();
  }, [ready, user, isAdmin, load]);

  const handleReindex = async () => {
    if (
      !window.confirm(
        "Reload the index from S3? Queries keep working, but results may be briefly stale while shards swap.",
      )
    ) {
      return;
    }
    setReindexing(true);
    try {
      const res = await api.reindex();
      const msg =
        (typeof res.detail === "string" && res.detail) ||
        (typeof res.status === "string" && res.status) ||
        (typeof res.message === "string" && res.message) ||
        "Reindex triggered";
      toast.push(msg, "success");
      void load();
    } catch (err) {
      toast.push(errorMessage(err), "error");
    } finally {
      setReindexing(false);
    }
  };

  if (!ready || !user) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <Spinner size={28} />
      </div>
    );
  }

  if (!isAdmin) {
    return (
      <div className="mx-auto max-w-md py-24">
        <div className="card px-6 py-10 text-center">
          <p className="text-sm font-medium text-zinc-200">
            Admin access required
          </p>
          <p className="mt-2 text-sm text-zinc-500">
            You are signed in as{" "}
            <span className="text-zinc-300">{user.email}</span> with role{" "}
            <span className="rounded bg-zinc-800 px-1.5 py-0.5 font-mono text-xs text-zinc-300">
              {role ?? "user"}
            </span>
            . Ask an administrator to elevate your account.
          </p>
        </div>
      </div>
    );
  }

  const { cards, tables } = stats
    ? splitStats(stats)
    : { cards: [] as [string, string][], tables: [] as [string, [string, string][]][] };

  const typeRows = dist ? pickSection(dist, ["types", "garment_types", "type"]) : [];
  const colorRows = dist ? pickSection(dist, ["colors", "color"]) : [];
  const sceneRows = dist
    ? pickSection(dist, ["scene_types", "scenes", "scene_type"])
    : [];
  const formalityRows = dist
    ? pickSection(dist, ["formality", "formalities"])
    : [];

  return (
    <div className="py-8">
      {/* header */}
      <div className="mb-6 flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight text-white">
            Admin console
          </h1>
          <p className="mt-1 text-sm text-zinc-500">
            Index health, attribute distributions and engine configuration.
          </p>
        </div>
        <button
          type="button"
          onClick={handleReindex}
          disabled={reindexing}
          className="btn-primary"
        >
          {reindexing && <Spinner size={16} className="text-white" />}
          {reindexing ? "Reloading…" : "Reload index from S3"}
        </button>
      </div>

      {loading ? (
        <div className="space-y-6">
          <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
            {Array.from({ length: 4 }, (_, i) => (
              <div key={i} className="skeleton h-24" />
            ))}
          </div>
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <div className="skeleton h-64" />
            <div className="skeleton h-64" />
          </div>
        </div>
      ) : loadError ? (
        <div className="card flex flex-col items-center gap-4 px-6 py-16 text-center">
          <p className="text-sm text-red-300">{loadError}</p>
          <button type="button" onClick={() => void load()} className="btn-ghost">
            Retry
          </button>
        </div>
      ) : (
        <div className="space-y-8">
          {/* stat cards */}
          {cards.length > 0 && (
            <div className="grid grid-cols-2 gap-4 md:grid-cols-4">
              {cards.map(([label, value]) => (
                <div key={label} className="card p-4">
                  <p className="truncate text-[11px] font-medium uppercase tracking-wider text-zinc-500">
                    {label}
                  </p>
                  <p
                    className="mt-1.5 truncate text-2xl font-semibold tabular-nums text-white"
                    title={value}
                  >
                    {value}
                  </p>
                </div>
              ))}
            </div>
          )}

          {/* attribute distributions */}
          {(typeRows.length || colorRows.length || sceneRows.length || formalityRows.length) > 0 && (
            <section>
              <h2 className="mb-4 text-sm font-semibold text-zinc-300">
                Attribute distribution
              </h2>
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <DistChart
                  title="Garment types"
                  rows={typeRows}
                  colorFor={() => "#8b5cf6"}
                />
                <DistChart
                  title="Colors"
                  rows={colorRows}
                  colorFor={(label) => colorHex(label, "#71717a")}
                  swatch
                />
                <DistChart
                  title="Scene types"
                  rows={sceneRows}
                  colorFor={() => "#38bdf8"}
                />
                <DistChart
                  title="Formality"
                  rows={formalityRows}
                  colorFor={() => "#34d399"}
                />
              </div>
            </section>
          )}

          {/* engine configuration tables */}
          {tables.length > 0 && (
            <section>
              <h2 className="mb-4 text-sm font-semibold text-zinc-300">
                Engine configuration
              </h2>
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                {tables.map(([title, rows]) => (
                  <div key={title} className="card overflow-hidden">
                    <p className="border-b border-zinc-800/70 px-5 py-3 text-xs font-semibold uppercase tracking-wider text-zinc-500">
                      {title}
                    </p>
                    <dl>
                      {rows.map(([k, v], i) => (
                        <div
                          key={k}
                          className={`flex items-baseline justify-between gap-4 px-5 py-2.5 text-sm ${
                            i % 2 === 1 ? "bg-zinc-950/40" : ""
                          }`}
                        >
                          <dt className="text-zinc-500">{k}</dt>
                          <dd
                            className="truncate font-mono text-xs text-zinc-300"
                            title={v}
                          >
                            {v}
                          </dd>
                        </div>
                      ))}
                    </dl>
                  </div>
                ))}
              </div>
            </section>
          )}

          {stats === null && dist === null && (
            <div className="card px-6 py-16 text-center text-sm text-zinc-500">
              The API returned no admin data yet — index something first.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
