"use client";

/** Pill badge with optional color swatch; `strike` renders negated terms. */

import type { ReactNode } from "react";

type Tone = "neutral" | "violet" | "danger" | "sky";

const TONES: Record<Tone, string> = {
  neutral: "border-zinc-800 bg-zinc-900/80 text-zinc-300",
  violet: "border-violet-500/40 bg-violet-500/10 text-violet-300",
  danger: "border-red-500/40 bg-red-500/10 text-red-300",
  sky: "border-sky-500/40 bg-sky-500/10 text-sky-300",
};

export default function Badge({
  children,
  swatch,
  tone = "neutral",
  strike = false,
  title,
}: {
  children: ReactNode;
  swatch?: string;
  tone?: Tone;
  strike?: boolean;
  title?: string;
}) {
  return (
    <span
      title={title}
      className={`inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-medium ${TONES[tone]} ${
        strike ? "line-through decoration-red-400/70 opacity-75" : ""
      }`}
    >
      {swatch && (
        <span
          className="h-2 w-2 shrink-0 rounded-full ring-1 ring-white/25"
          style={{ background: swatch }}
        />
      )}
      {children}
    </span>
  );
}
