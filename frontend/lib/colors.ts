/**
 * Color helpers shared by the UI.
 *
 * COLOR_HEX mirrors core/lexicon.py CANONICAL_COLORS so parsed-query chips and
 * attribute swatches render the same anchors the backend canonicalizes into.
 * TERM_PALETTE assigns one stable hue per query term for the bbox overlay —
 * hashing the term keeps a given term the same color on every card.
 */

export const COLOR_HEX: Record<string, string> = {
  black: "#1a1a1a",
  white: "#f5f5f5",
  gray: "#808080",
  red: "#d0312d",
  orange: "#f28c28",
  yellow: "#ffd400",
  green: "#3a7d44",
  blue: "#2a6bd4",
  navy: "#1b2a4a",
  purple: "#7d3cb5",
  pink: "#e75480",
  brown: "#7b4a2d",
  beige: "#d9c7a7",
  cream: "#f4ead5",
  gold: "#c9a227",
  silver: "#c0c0c0",
  teal: "#2a9d9f",
  maroon: "#701c2c",
  olive: "#6b7233",
  khaki: "#b7a878",
  multicolor: "#9ca3af",
};

/** Hex anchor for a canonical color name, with a neutral fallback. */
export function colorHex(
  name: string | null | undefined,
  fallback = "#71717a",
): string {
  if (!name) return fallback;
  return COLOR_HEX[name.trim().toLowerCase()] ?? fallback;
}

/** Distinct, dark-theme-legible hues for query-term bounding boxes. */
export const TERM_PALETTE: readonly string[] = [
  "#f59e0b", // amber
  "#22d3ee", // cyan
  "#34d399", // emerald
  "#fb7185", // rose
  "#a78bfa", // violet-light
  "#a3e635", // lime
  "#e879f9", // fuchsia
  "#60a5fa", // blue
];

/** Stable term → hue mapping (same term, same color, on every card). */
export function termColor(term: string): string {
  let h = 0;
  for (let i = 0; i < term.length; i++) {
    h = (h * 31 + term.charCodeAt(i)) >>> 0;
  }
  return TERM_PALETTE[h % TERM_PALETTE.length];
}

/** Channel colors for the score-breakdown bars. */
export const CHANNEL_COLORS: Record<string, string> = {
  garment: "#8b5cf6",
  scene: "#38bdf8",
  attribute: "#34d399",
  rerank: "#fbbf24",
};
