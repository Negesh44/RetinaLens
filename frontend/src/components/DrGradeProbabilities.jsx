import React, { useState } from 'react';
import { AlertTriangle } from 'lucide-react';

const GRADES = [
  { grade: 0, name: 'No DR', detail: 'No microaneurysms, hemorrhages, or exudates.' },
  { grade: 1, name: 'Mild NPDR', detail: 'Microaneurysms only.' },
  { grade: 2, name: 'Moderate NPDR', detail: 'More than microaneurysms but less than severe NPDR.' },
  { grade: 3, name: 'Severe NPDR', detail: '4-2-1 rule: extensive hemorrhages, venous beading, or IRMA.' },
  { grade: 4, name: 'Proliferative DR', detail: 'Neovascularization or vitreous / preretinal hemorrhage.' },
];
const REFERABLE_FROM = 2;

const pct = (v) => `${(v * 100).toFixed(v > 0 && v < 0.01 ? 1 : 0)}%`;

export default function DrGradeProbabilities({ probabilities = {}, reportedGrade }) {
  const [hovered, setHovered] = useState(null);

  const rows = GRADES.map((g) => ({ ...g, p: Number(probabilities?.[String(g.grade)] ?? 0) }));
  if (rows.every((r) => r.p === 0)) return null;

  const mostLikely = rows.reduce((best, r) => (r.p > best.p ? r : best), rows[0]);
  const referableP = rows.filter((r) => r.grade >= REFERABLE_FROM).reduce((sum, r) => sum + r.p, 0);
  const reportedP = rows[reportedGrade]?.p ?? 0;
  const isInconsistent = reportedGrade != null && mostLikely.grade !== reportedGrade;
  const focus = hovered != null ? rows[hovered] : null;

  return (
    <section className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-xl">
      <header>
        <h4 className="font-heading text-sm font-bold text-white">DR Grade Probabilities</h4>
        <p className="mt-0.5 text-xs text-slate-400">How likely each diabetic retinopathy grade is, per the grading model.</p>
      </header>

      {/* Headline numbers */}
      <div className="grid grid-cols-2 gap-3">
        <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
          <p className="text-[11px] font-medium text-slate-400">Most likely grade</p>
          <p className="mt-1 font-heading text-lg font-bold leading-tight text-white">
            Grade {mostLikely.grade}
            <span className="ml-1.5 font-mono text-sm font-semibold text-slate-300">{pct(mostLikely.p)}</span>
          </p>
          <p className="text-[11px] text-slate-400">{mostLikely.name}</p>
        </div>
        <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
          <p className="text-[11px] font-medium text-slate-400">Chance of referable DR</p>
          <p className="mt-1 font-mono text-lg font-bold leading-tight text-white">{pct(referableP)}</p>
          <p className="text-[11px] text-slate-400">Grade {REFERABLE_FROM} or higher</p>
        </div>
      </div>

      {/* Per-grade bars: one series, the reported grade emphasized */}
      <div role="table" aria-label="Probability per DR grade" className="space-y-1" onMouseLeave={() => setHovered(null)}>
        {rows.map((r) => {
          const isReported = r.grade === reportedGrade;
          return (
            <div
              key={r.grade}
              role="row"
              tabIndex={0}
              onMouseEnter={() => setHovered(r.grade)}
              onFocus={() => setHovered(r.grade)}
              onBlur={() => setHovered(null)}
              className={`grid grid-cols-[7.5rem_1fr_3rem] items-center gap-3 rounded-lg px-2 py-1.5 outline-none transition-colors ${
                hovered === r.grade ? 'bg-slate-800/70' : ''
              } ${r.grade === REFERABLE_FROM ? 'mt-2 border-t border-dashed border-slate-700 pt-2.5' : ''}`}
            >
              <span role="cell" className="flex items-center gap-1.5 truncate text-xs">
                <span className="font-mono text-slate-500">{r.grade}</span>
                <span className={isReported ? 'font-semibold text-white' : 'text-slate-300'}>{r.name}</span>
              </span>
              <span role="cell" className="h-2 overflow-hidden rounded-full bg-slate-800">
                <span
                  className="block h-full rounded-r"
                  style={{
                    width: `${Math.max(r.p > 0 ? 1.5 : 0, r.p * 100)}%`,
                    backgroundColor: isReported ? '#38bdf8' : '#64748b',
                  }}
                />
              </span>
              <span role="cell" className={`text-right font-mono text-xs tabular-nums ${isReported ? 'font-bold text-white' : 'text-slate-300'}`}>
                {pct(r.p)}
              </span>
            </div>
          );
        })}
      </div>

      <div className="flex min-h-[2.5rem] items-start justify-between gap-3 text-[11px] text-slate-400">
        {focus ? (
          <p>
            <span className="font-semibold text-slate-200">Grade {focus.grade} · {focus.name}:</span> {focus.detail}
          </p>
        ) : (
          <p className="flex flex-wrap items-center gap-x-4 gap-y-1">
            <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
              <span className="inline-block h-2 w-3 rounded-sm bg-sky-400" /> Reported grade
            </span>
            <span className="inline-flex items-center gap-1.5 whitespace-nowrap">
              <span className="inline-block h-2 w-3 rounded-sm bg-slate-500" /> Other grades
            </span>
            <span className="whitespace-nowrap text-slate-500">Dashed line = referral cut-off</span>
          </p>
        )}
      </div>

      {isInconsistent && (
        <div className="flex items-start gap-2.5 rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-xs text-amber-200">
          <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0 text-amber-400" />
          <p>
            <span className="font-semibold">Inconsistent model output.</span> The reported grade ({reportedGrade}, {pct(reportedP)}) is
            not the most likely grade ({mostLikely.grade}, {pct(mostLikely.p)}). Treat this DR grade as unreliable and confirm clinically.
          </p>
        </div>
      )}
    </section>
  );
}
