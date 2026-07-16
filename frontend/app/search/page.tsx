"use client";

/**
 * Search page: text queries (with the five acceptance-query chips) or
 * query-by-image with a text refinement, a k selector and rerank toggle,
 * a collapsible "Query understanding" panel rendering the ParsedQuery, and
 * a results grid of explainable ResultCards.
 */

import { useRouter } from "next/navigation";
import {
  useCallback,
  useEffect,
  useRef,
  useState,
  type ChangeEvent,
  type DragEvent,
  type FormEvent,
} from "react";

import Badge from "@/components/Badge";
import ResultCard from "@/components/ResultCard";
import Spinner from "@/components/Spinner";
import { useToast } from "@/components/Toast";
import * as api from "@/lib/api";
import { errorMessage, type ParsedQuery, type SearchResult } from "@/lib/api";
import { useAuth } from "@/lib/auth-context";
import { colorHex } from "@/lib/colors";

const EXAMPLE_QUERIES: readonly string[] = [
  "A person in a bright yellow raincoat",
  "Professional business attire inside a modern office",
  "Someone wearing a blue shirt sitting on a park bench",
  "Casual weekend outfit for a city walk",
  "A red tie and a white shirt in a formal setting",
];

const K_OPTIONS = [5, 10, 20, 30] as const;

type Mode = "text" | "image";

// ── query understanding panel ────────────────────────────────────────────────

function QueryPanel({ parsed }: { parsed: ParsedQuery }) {
  return (
    <div className="card p-4">
      <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-zinc-500">
        Query understanding
      </h2>

      <div className="space-y-3">
        {parsed.garments.length > 0 && (
          <div>
            <p className="mb-1.5 text-[11px] text-zinc-600">Garment terms</p>
            <div className="flex flex-wrap gap-1.5">
              {parsed.garments.map((g, i) => {
                const label = [g.color, g.type].filter(Boolean).join(" ") || "?";
                const hex = g.color ? colorHex(g.color) : null;
                return (
                  <span
                    key={`${label}-${i}`}
                    className="inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium"
                    style={
                      hex
                        ? {
                            borderColor: `${hex}66`,
                            background: `${hex}1a`,
                            color: "#e4e4e7",
                          }
                        : { borderColor: "#3f3f46", color: "#d4d4d8" }
                    }
                  >
                    {hex && (
                      <span
                        className="h-2.5 w-2.5 rounded-full ring-1 ring-white/25"
                        style={{ background: hex }}
                      />
                    )}
                    {label}
                  </span>
                );
              })}
            </div>
          </div>
        )}

        {(parsed.scene || parsed.style) && (
          <div className="flex flex-wrap gap-1.5">
            {parsed.scene && (
              <Badge tone="sky" title="scene">
                scene: {parsed.scene}
              </Badge>
            )}
            {parsed.style && (
              <Badge tone="violet" title="style">
                style: {parsed.style}
              </Badge>
            )}
          </div>
        )}

        {parsed.negations.length > 0 && (
          <div>
            <p className="mb-1.5 text-[11px] text-zinc-600">Excluded</p>
            <div className="flex flex-wrap gap-1.5">
              {parsed.negations.map((n, i) => (
                <Badge key={`${n.term}-${i}`} tone="danger" strike>
                  {n.term}
                </Badge>
              ))}
            </div>
          </div>
        )}

        <div>
          <p className="mb-1.5 text-[11px] text-zinc-600">Parsed query</p>
          <pre className="max-h-80 overflow-auto rounded-xl border border-zinc-800 bg-zinc-950/80 p-3 text-[11px] leading-relaxed text-zinc-400">
            {JSON.stringify(parsed, null, 2)}
          </pre>
        </div>
      </div>
    </div>
  );
}

// ── skeleton loader ──────────────────────────────────────────────────────────

function SkeletonCard() {
  return (
    <div className="card overflow-hidden">
      <div className="skeleton aspect-[4/5] w-full rounded-none" />
      <div className="space-y-2.5 p-4">
        <div className="skeleton h-3 w-2/5" />
        <div className="skeleton h-3 w-4/5" />
        <div className="skeleton h-3 w-3/5" />
        <div className="skeleton h-3 w-1/2" />
      </div>
    </div>
  );
}

// ── page ─────────────────────────────────────────────────────────────────────

export default function SearchPage() {
  const { user, ready } = useAuth();
  const router = useRouter();
  const toast = useToast();

  const [mode, setMode] = useState<Mode>("text");
  const [query, setQuery] = useState("");
  const [k, setK] = useState<number>(10);
  const [useRerank, setUseRerank] = useState(true);

  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [refinement, setRefinement] = useState("");
  const [dragging, setDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<SearchResult[] | null>(null);
  const [parsed, setParsed] = useState<ParsedQuery | null>(null);
  const [panelOpen, setPanelOpen] = useState(true);

  // auth guard
  useEffect(() => {
    if (ready && !user) router.replace("/login");
  }, [ready, user, router]);

  // revoke the object URL on preview change / unmount
  useEffect(() => {
    return () => {
      if (preview) URL.revokeObjectURL(preview);
    };
  }, [preview]);

  const setPickedFile = useCallback((f: File | null) => {
    setFile(f);
    setPreview((old) => {
      if (old) URL.revokeObjectURL(old);
      return f ? URL.createObjectURL(f) : null;
    });
  }, []);

  const runTextSearch = useCallback(
    async (q: string) => {
      const trimmed = q.trim();
      if (!trimmed || loading) return;
      setLoading(true);
      setParsed(null);
      try {
        const res = await api.search(trimmed, k, useRerank);
        setResults(res.results);
        if (res.parsed) {
          setParsed(res.parsed);
        } else {
          // /search may not echo the parse — fetch it for the debug panel
          try {
            setParsed(await api.parse(trimmed));
          } catch {
            /* panel simply stays empty */
          }
        }
      } catch (err) {
        toast.push(errorMessage(err), "error");
      } finally {
        setLoading(false);
      }
    },
    [k, useRerank, loading, toast],
  );

  const runImageSearch = useCallback(async () => {
    if (!file || loading) return;
    setLoading(true);
    setParsed(null);
    try {
      const res = await api.searchImage(file, refinement, k);
      setResults(res.results);
      if (res.parsed) {
        setParsed(res.parsed);
      } else if (refinement.trim()) {
        try {
          setParsed(await api.parse(refinement.trim()));
        } catch {
          /* best-effort */
        }
      }
    } catch (err) {
      toast.push(errorMessage(err), "error");
    } finally {
      setLoading(false);
    }
  }, [file, refinement, k, loading, toast]);

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (mode === "text") void runTextSearch(query);
    else void runImageSearch();
  };

  const handleChip = (q: string) => {
    setMode("text");
    setQuery(q);
    void runTextSearch(q);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setDragging(false);
    const f = e.dataTransfer.files?.[0];
    if (f && f.type.startsWith("image/")) setPickedFile(f);
    else if (f) toast.push("Please drop an image file", "error");
  };

  const handleFileInput = (e: ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0] ?? null;
    if (f) setPickedFile(f);
    e.target.value = ""; // allow re-selecting the same file
  };

  if (!ready || !user) {
    return (
      <div className="flex min-h-[60vh] items-center justify-center">
        <Spinner size={28} />
      </div>
    );
  }

  return (
    <div className="py-8">
      {/* ── header ── */}
      <div className="mb-6">
        <h1 className="text-xl font-semibold tracking-tight text-white">
          Search the fashion index
        </h1>
        <p className="mt-1 text-sm text-zinc-500">
          Compositional queries over garments, colors, scenes and style — with
          region-level explanations.
        </p>
      </div>

      {/* ── query card ── */}
      <form onSubmit={handleSubmit} className="card space-y-4 p-4 sm:p-5">
        {/* controls row */}
        <div className="flex flex-wrap items-center gap-3">
          {/* mode toggle */}
          <div className="grid grid-cols-2 gap-1 rounded-xl bg-zinc-950/70 p-1">
            {(
              [
                ["text", "Text"],
                ["image", "Image + Refine"],
              ] as const
            ).map(([m, label]) => (
              <button
                key={m}
                type="button"
                onClick={() => setMode(m)}
                className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                  mode === m
                    ? "bg-zinc-800 text-white shadow-sm"
                    : "text-zinc-500 hover:text-zinc-300"
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          <div className="ml-auto flex items-center gap-4">
            {/* k selector */}
            <label className="flex items-center gap-2 text-xs text-zinc-500">
              top&nbsp;k
              <select
                value={k}
                onChange={(e) => setK(Number(e.target.value))}
                className="rounded-lg border border-zinc-800 bg-zinc-950/70 px-2 py-1.5 text-xs text-zinc-200 outline-none focus:border-violet-500"
              >
                {K_OPTIONS.map((n) => (
                  <option key={n} value={n}>
                    {n}
                  </option>
                ))}
              </select>
            </label>

            {/* rerank toggle */}
            <label className="flex cursor-pointer items-center gap-2 text-xs text-zinc-500">
              <button
                type="button"
                role="switch"
                aria-checked={useRerank}
                onClick={() => setUseRerank((v) => !v)}
                className={`relative h-5 w-9 rounded-full transition-colors ${
                  useRerank ? "bg-violet-600" : "bg-zinc-700"
                }`}
              >
                <span
                  className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-[left] ${
                    useRerank ? "left-[1.125rem]" : "left-0.5"
                  }`}
                />
              </button>
              rerank
            </label>
          </div>
        </div>

        {/* input row */}
        {mode === "text" ? (
          <div className="flex gap-2">
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder='Try "a red tie and a white shirt in a formal setting"'
              className="input flex-1"
            />
            <button
              type="submit"
              disabled={loading || !query.trim()}
              className="btn-primary shrink-0"
            >
              {loading ? <Spinner size={16} className="text-white" /> : "Search"}
            </button>
          </div>
        ) : (
          <div className="flex flex-col gap-3 sm:flex-row">
            {/* dropzone */}
            <div
              role="button"
              tabIndex={0}
              onClick={() => fileInputRef.current?.click()}
              onKeyDown={(e) => {
                if (e.key === "Enter" || e.key === " ") {
                  e.preventDefault();
                  fileInputRef.current?.click();
                }
              }}
              onDragOver={(e) => {
                e.preventDefault();
                setDragging(true);
              }}
              onDragLeave={() => setDragging(false)}
              onDrop={handleDrop}
              className={`flex min-h-28 flex-1 cursor-pointer items-center justify-center gap-4 rounded-xl border-2 border-dashed px-4 py-4 transition-colors ${
                dragging
                  ? "border-violet-500 bg-violet-500/5"
                  : "border-zinc-800 hover:border-zinc-700"
              }`}
            >
              {preview ? (
                <>
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={preview}
                    alt="Query preview"
                    className="h-24 w-24 rounded-lg object-cover"
                  />
                  <div className="min-w-0 text-sm">
                    <p className="truncate text-zinc-300">{file?.name}</p>
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setPickedFile(null);
                      }}
                      className="mt-1 text-xs text-red-400 hover:text-red-300"
                    >
                      Remove
                    </button>
                  </div>
                </>
              ) : (
                <p className="text-center text-sm text-zinc-500">
                  Drop an image here, or click to browse
                </p>
              )}
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={handleFileInput}
              />
            </div>

            {/* refinement + submit */}
            <div className="flex w-full flex-col justify-center gap-2 sm:w-72">
              <input
                type="text"
                value={refinement}
                onChange={(e) => setRefinement(e.target.value)}
                placeholder='Refinement, e.g. "but in green"'
                className="input"
              />
              <button
                type="submit"
                disabled={loading || !file}
                className="btn-primary"
              >
                {loading ? (
                  <Spinner size={16} className="text-white" />
                ) : (
                  "Search by image"
                )}
              </button>
            </div>
          </div>
        )}

        {/* example chips */}
        {mode === "text" && (
          <div className="flex flex-wrap gap-2">
            {EXAMPLE_QUERIES.map((q) => (
              <button
                key={q}
                type="button"
                onClick={() => handleChip(q)}
                disabled={loading}
                className="rounded-full border border-zinc-800 bg-zinc-950/60 px-3 py-1.5 text-xs text-zinc-400 transition-colors hover:border-violet-500/50 hover:text-violet-300 disabled:opacity-50"
              >
                {q}
              </button>
            ))}
          </div>
        )}
      </form>

      {/* ── results area ── */}
      <div className="mt-8 flex flex-col gap-6 lg:flex-row lg:items-start">
        {/* query understanding panel */}
        {parsed && (
          <aside className="shrink-0 lg:w-80">
            <button
              type="button"
              onClick={() => setPanelOpen((v) => !v)}
              className="mb-2 flex items-center gap-1.5 text-xs font-medium text-zinc-500 transition-colors hover:text-zinc-300"
              aria-expanded={panelOpen}
            >
              <svg
                width="12"
                height="12"
                viewBox="0 0 12 12"
                className={`transition-transform ${panelOpen ? "rotate-90" : ""}`}
              >
                <path
                  d="M4 2l4 4-4 4"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  fill="none"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              Query understanding
            </button>
            {panelOpen && <QueryPanel parsed={parsed} />}
          </aside>
        )}

        {/* grid */}
        <section className="min-w-0 flex-1">
          {loading ? (
            <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 xl:grid-cols-3">
              {Array.from({ length: 6 }, (_, i) => (
                <SkeletonCard key={i} />
              ))}
            </div>
          ) : results === null ? (
            <div className="card flex flex-col items-center justify-center px-6 py-20 text-center">
              <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-2xl bg-violet-500/10">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
                  <circle
                    cx="11"
                    cy="11"
                    r="7"
                    stroke="#a78bfa"
                    strokeWidth="2"
                  />
                  <path
                    d="M20 20l-3.5-3.5"
                    stroke="#a78bfa"
                    strokeWidth="2"
                    strokeLinecap="round"
                  />
                </svg>
              </div>
              <p className="text-sm font-medium text-zinc-300">
                Nothing searched yet
              </p>
              <p className="mt-1 max-w-sm text-sm text-zinc-500">
                Type a query above or pick one of the example chips to see
                region-level explanations for every match.
              </p>
            </div>
          ) : results.length === 0 ? (
            <div className="card flex flex-col items-center justify-center px-6 py-20 text-center">
              <p className="text-sm font-medium text-zinc-300">No matches</p>
              <p className="mt-1 max-w-sm text-sm text-zinc-500">
                The metadata pre-filter may have excluded everything. Try
                softening the query — fewer attributes, or drop a negation.
              </p>
            </div>
          ) : (
            <>
              <p className="mb-3 text-xs text-zinc-600">
                {results.length} result{results.length === 1 ? "" : "s"}
              </p>
              <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 xl:grid-cols-3">
                {results.map((r, i) => (
                  <div
                    key={r.image_id}
                    className="animate-fade-up"
                    style={{ animationDelay: `${Math.min(i, 12) * 40}ms` }}
                  >
                    <ResultCard result={r} />
                  </div>
                ))}
              </div>
            </>
          )}
        </section>
      </div>
    </div>
  );
}
