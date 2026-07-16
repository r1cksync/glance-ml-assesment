"use client";

/**
 * One search result with full explainability:
 *  - bbox overlay: each MatchExplanation is drawn over the image, one stable
 *    hue per query term. Boxes are positioned with PERCENTAGES computed from
 *    the image's natural size (captured onLoad), so they remain pixel-correct
 *    at any rendered size; a ResizeObserver tracks the rendered width to
 *    switch into a compact layout on small cards.
 *  - hover tooltip: "red tie · 0.83 · region label"
 *  - legend chips mapping query term -> hue
 *  - slim score-breakdown bars (garment / scene / attribute / rerank)
 *  - attribute badges (garment type + color swatch, formality, scene type)
 */

import { useEffect, useRef, useState } from "react";

import Badge from "@/components/Badge";
import ScoreBar from "@/components/ScoreBar";
import { imageUrl, type SearchResult } from "@/lib/api";
import { CHANNEL_COLORS, colorHex, termColor } from "@/lib/colors";

interface NaturalSize {
  w: number;
  h: number;
}

const COMPACT_WIDTH_PX = 260;

function pct(fraction: number): string {
  return `${(Math.max(0, Math.min(1, fraction)) * 100).toFixed(2)}%`;
}

export default function ResultCard({ result }: { result: SearchResult }) {
  const [explain, setExplain] = useState(true);
  const [natural, setNatural] = useState<NaturalSize | null>(null);
  const [hoveredKey, setHoveredKey] = useState<string | null>(null);
  const [compact, setCompact] = useState(false);
  const [imgFailed, setImgFailed] = useState(false);
  const frameRef = useRef<HTMLDivElement>(null);

  // Track rendered width (relative-unit overlay needs no per-resize math,
  // but small cards drop secondary detail).
  useEffect(() => {
    const el = frameRef.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        setCompact(entry.contentRect.width < COMPACT_WIDTH_PX);
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const matches = result.matches ?? [];
  const attrs = result.attributes;
  const comps = result.components;

  // legend: one entry per unique query term (best similarity wins)
  const legend = new Map<string, { color: string; similarity: number }>();
  for (const m of matches) {
    const prev = legend.get(m.query_term);
    if (!prev || m.similarity > prev.similarity) {
      legend.set(m.query_term, {
        color: termColor(m.query_term),
        similarity: m.similarity,
      });
    }
  }

  const formalities = attrs
    ? Array.from(
        new Set(
          attrs.garments.map((g) => g.formality).filter((f) => f && f !== "unknown"),
        ),
      )
    : [];

  return (
    <article className="card group flex flex-col overflow-hidden transition-colors hover:border-zinc-700">
      {/* ── image + overlay ── */}
      <div ref={frameRef} className="relative bg-zinc-950">
        {imgFailed ? (
          <div className="flex aspect-[4/5] w-full items-center justify-center text-xs text-zinc-600">
            image unavailable
          </div>
        ) : (
          // Plain <img>: natural aspect ratio preserved (w-full h-auto) so the
          // percentage bbox overlay maps exactly onto displayed pixels.
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={imageUrl(result)}
            alt={result.image_id}
            className="block h-auto w-full select-none"
            loading="lazy"
            draggable={false}
            onLoad={(e) => {
              const el = e.currentTarget;
              if (el.naturalWidth > 0 && el.naturalHeight > 0) {
                setNatural({ w: el.naturalWidth, h: el.naturalHeight });
              }
            }}
            onError={() => setImgFailed(true)}
          />
        )}

        {/* bbox overlay — all geometry in % of the frame, correct at any size */}
        {explain && natural && !imgFailed && (
          <div className="absolute inset-0">
            {matches.map((m) => {
              const [x1, y1, x2, y2] = m.bbox;
              const left = x1 / natural.w;
              const top = y1 / natural.h;
              const width = (x2 - x1) / natural.w;
              const height = (y2 - y1) / natural.h;
              const color = termColor(m.query_term);
              const key = `${m.region_id}:${m.query_term}`;
              const hovered = hoveredKey === key;
              return (
                <div
                  key={key}
                  className="absolute rounded-[4px] border-2 transition-shadow duration-150"
                  style={{
                    left: pct(left),
                    top: pct(top),
                    width: pct(width),
                    height: pct(height),
                    borderColor: color,
                    background: hovered ? `${color}2b` : `${color}12`,
                    boxShadow: hovered
                      ? `0 0 0 2px ${color}55`
                      : "0 0 0 1px rgb(0 0 0 / 0.35)",
                    zIndex: hovered ? 10 : 1,
                  }}
                  onMouseEnter={() => setHoveredKey(key)}
                  onMouseLeave={() =>
                    setHoveredKey((k) => (k === key ? null : k))
                  }
                />
              );
            })}

            {/* tooltip rendered at frame level so it is never trapped inside a small box */}
            {matches.map((m) => {
              const key = `${m.region_id}:${m.query_term}`;
              if (hoveredKey !== key) return null;
              const [x1, y1] = m.bbox;
              const left = x1 / natural.w;
              const top = y1 / natural.h;
              const color = termColor(m.query_term);
              const placeAbove = top > 0.14;
              return (
                <div
                  key={`tip-${key}`}
                  className="pointer-events-none absolute z-20 max-w-[92%] truncate rounded-md border border-zinc-700 bg-zinc-950/95 px-2 py-1 text-[11px] font-medium text-zinc-100 shadow-xl shadow-black/50"
                  style={{
                    left: `min(${(left * 100).toFixed(2)}%, 55%)`,
                    top: placeAbove
                      ? `calc(${(top * 100).toFixed(2)}% - 1.9rem)`
                      : `calc(${(top * 100).toFixed(2)}% + 0.35rem)`,
                  }}
                >
                  <span style={{ color }}>{m.query_term}</span>
                  <span className="text-zinc-500"> · </span>
                  {m.similarity.toFixed(2)}
                  <span className="text-zinc-500"> · </span>
                  <span className="text-zinc-400">{m.label}</span>
                </div>
              );
            })}
          </div>
        )}

        {/* overall score */}
        <span className="absolute right-2 top-2 rounded-full border border-zinc-700/60 bg-zinc-950/80 px-2.5 py-1 text-xs font-semibold tabular-nums text-violet-300 backdrop-blur">
          {result.score.toFixed(3)}
        </span>

        {/* explain toggle */}
        {matches.length > 0 && (
          <button
            type="button"
            onClick={() => setExplain((v) => !v)}
            aria-pressed={explain}
            className={`absolute left-2 top-2 rounded-full border px-2.5 py-1 text-[11px] font-medium backdrop-blur transition-colors ${
              explain
                ? "border-violet-500/60 bg-violet-500/20 text-violet-200"
                : "border-zinc-700 bg-zinc-950/70 text-zinc-400 hover:text-zinc-200"
            }`}
          >
            explain
          </button>
        )}
      </div>

      {/* ── body ── */}
      <div className="flex flex-1 flex-col gap-3 p-4">
        <p
          className="truncate font-mono text-[11px] text-zinc-600"
          title={result.image_id}
        >
          {result.image_id}
        </p>

        {/* legend: query term -> hue */}
        {legend.size > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {Array.from(legend.entries()).map(([term, info]) => (
              <span
                key={term}
                className="inline-flex items-center gap-1.5 rounded-full border border-zinc-800 bg-zinc-950/60 px-2 py-0.5 text-[11px] text-zinc-300"
              >
                <span
                  className="h-2 w-2 shrink-0 rounded-full"
                  style={{ background: info.color }}
                />
                {term}
                {!compact && (
                  <span className="tabular-nums text-zinc-500">
                    {info.similarity.toFixed(2)}
                  </span>
                )}
              </span>
            ))}
          </div>
        )}

        {/* score breakdown */}
        <div className="space-y-1.5">
          <ScoreBar label="garment" value={comps.garment} color={CHANNEL_COLORS.garment} />
          <ScoreBar label="scene" value={comps.scene} color={CHANNEL_COLORS.scene} />
          <ScoreBar label="attribute" value={comps.attribute} color={CHANNEL_COLORS.attribute} />
          {comps.rerank !== null && comps.rerank !== undefined && (
            <ScoreBar label="rerank" value={comps.rerank} color={CHANNEL_COLORS.rerank} />
          )}
        </div>

        {/* attribute badges */}
        {attrs && (
          <div className="mt-auto flex flex-wrap gap-1.5 border-t border-zinc-800/70 pt-3">
            {attrs.garments.slice(0, compact ? 3 : 5).map((g, i) => (
              <Badge
                key={`${g.type}-${g.color}-${i}`}
                swatch={g.color_hex ?? colorHex(g.color)}
                title={g.material ? `material: ${g.material}` : undefined}
              >
                {g.color} {g.type}
              </Badge>
            ))}
            {formalities.map((f) => (
              <Badge key={f} tone="violet">
                {f}
              </Badge>
            ))}
            {attrs.scene_type && attrs.scene_type !== "other" && (
              <Badge tone="sky" title={attrs.scene}>
                {attrs.scene_type}
              </Badge>
            )}
          </div>
        )}
      </div>
    </article>
  );
}
