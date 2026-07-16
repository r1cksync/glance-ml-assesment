"use client";

/** Slim horizontal bar for one fusion-score component (value in [0, 1]). */

export default function ScoreBar({
  label,
  value,
  color = "#8b5cf6",
}: {
  label: string;
  value: number;
  color?: string;
}) {
  const clamped = Math.max(0, Math.min(1, value));
  return (
    <div className="flex items-center gap-2">
      <span className="w-16 shrink-0 text-[10px] font-medium uppercase tracking-wider text-zinc-500">
        {label}
      </span>
      <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-zinc-800/90">
        <div
          className="h-full rounded-full transition-[width] duration-500 ease-out"
          style={{ width: `${clamped * 100}%`, background: color }}
        />
      </div>
      <span className="w-9 shrink-0 text-right text-[11px] tabular-nums text-zinc-400">
        {value.toFixed(2)}
      </span>
    </div>
  );
}
