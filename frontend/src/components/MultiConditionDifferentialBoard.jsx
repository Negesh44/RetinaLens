import React, { useState } from 'react';
import { CheckCircle2, ChevronDown, ChevronUp, CircleAlert } from 'lucide-react';
import {
  DISEASE_META,
  getDetectedFindings,
  getDiseaseMeta,
  getSeverityClass,
  getUrgency,
} from '../lib/diseases';

export default function MultiConditionDifferentialBoard({ diseaseFindings = [], cupToDiscRatio = null }) {
  const [showClear, setShowClear] = useState(false);
  const [expandedId, setExpandedId] = useState(null);

  const findings = (Array.isArray(diseaseFindings) ? diseaseFindings : []).filter((f) => f.disease_id !== 'normal');
  const detected = getDetectedFindings(findings);
  const detectedIds = new Set(detected.map((f) => f.disease_id));
  // Only detected conditions get cards; the ones checked and found clear stay in a collapsed list.
  const ordered = detected;
  const clear = findings.filter((f) => !detectedIds.has(f.disease_id)).sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0));

  // When the trained classifier produced the findings, list catalog conditions it cannot assess.
  const fromModel = findings.some((f) => f.model_name);
  const assessed = new Set(findings.map((f) => f.disease_id));
  const uncovered = fromModel
    ? Object.keys(DISEASE_META).filter((id) => !['normal', 'unassessable'].includes(id) && !assessed.has(id))
    : [];

  if (findings.length === 0) return null;

  return (
    <section className="space-y-5 rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-xl sm:p-6">
      <div className="flex flex-col gap-4 lg:flex-row lg:items-end lg:justify-between">
        <div>
          <h3 className="font-heading text-lg font-bold text-white">Detected Conditions</h3>
          <p className="mt-0.5 text-sm text-slate-400">
            <span className="font-semibold text-slate-200">{detected.length}</span> of {findings.length} screened conditions
            found on this scan.
          </p>
        </div>

        <div className="flex flex-col gap-3 sm:flex-row sm:items-center">
          {cupToDiscRatio !== null && cupToDiscRatio !== undefined && (
            <div className="min-w-[200px] rounded-lg border border-slate-800 bg-slate-950/60 px-3 py-2">
              <div className="mb-1 flex justify-between text-[11px]">
                <span className="font-semibold text-slate-400">Cup-to-disc ratio</span>
                <span className={`font-mono font-bold ${cupToDiscRatio >= 0.65 ? 'text-rose-400' : cupToDiscRatio >= 0.5 ? 'text-amber-400' : 'text-emerald-400'}`}>
                  {cupToDiscRatio.toFixed(2)}
                </span>
              </div>
              <div className="relative flex h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
                <div style={{ width: '45%' }} className="h-full bg-emerald-500/60" />
                <div style={{ width: '20%' }} className="h-full bg-amber-500/60" />
                <div style={{ width: '35%' }} className="h-full bg-rose-500/60" />
                <div className="absolute inset-y-0 w-0.5 bg-white" style={{ left: `${Math.min(98, Math.max(2, cupToDiscRatio * 100))}%` }} />
              </div>
            </div>
          )}

        </div>
      </div>

      {ordered.length === 0 ? (
        <p className="rounded-xl border border-slate-800 bg-slate-950/40 p-6 text-center text-sm text-slate-400">
          No conditions detected on this scan.
        </p>
      ) : (
        <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
          {ordered.map((f) => {
            const isDetected = detectedIds.has(f.disease_id);
            const meta = getDiseaseMeta(f.disease_id, f.disease_name);
            const urgency = getUrgency(f.triage_urgency);
            const isExpanded = expandedId === f.disease_id;
            const conf = (f.confidence ?? 0) * 100;

            return (
              <article
                key={f.disease_id}
                className={`rounded-xl border border-l-4 p-4 transition-colors ${
                  isDetected ? 'border-slate-700 bg-slate-950/70' : 'border-slate-800 bg-slate-950/30'
                }`}
                style={{ borderLeftColor: isDetected ? meta.color : '#334155' }}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="min-w-0">
                    <h4 className={`font-heading text-sm font-bold ${isDetected ? 'text-white' : 'text-slate-300'}`}>
                      {meta.name}
                    </h4>
                    <p className="mt-0.5 text-[11px] text-slate-500">
                      {[f.acronym, f.category].filter(Boolean).join(' · ')}
                    </p>
                  </div>
                  {isDetected ? (
                    <span className="inline-flex flex-shrink-0 items-center gap-1 rounded-md bg-rose-500/15 px-2 py-1 text-[11px] font-bold text-rose-300 ring-1 ring-inset ring-rose-500/40">
                      <CircleAlert className="h-3 w-3" /> Detected
                    </span>
                  ) : (
                    <span className="inline-flex flex-shrink-0 items-center gap-1 rounded-md bg-slate-800/70 px-2 py-1 text-[11px] font-semibold text-slate-400">
                      <CheckCircle2 className="h-3 w-3" /> Not detected
                    </span>
                  )}
                </div>

                <div className="mt-3 flex items-center gap-3">
                  <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-slate-800">
                    <div
                      className="h-full rounded-full"
                      style={{ width: `${Math.min(100, Math.max(3, conf))}%`, backgroundColor: isDetected ? meta.color : '#475569' }}
                    />
                  </div>
                  <span className="w-10 text-right font-mono text-xs font-semibold tabular-nums text-slate-300">
                    {conf.toFixed(0)}%
                  </span>
                </div>

                {f.test_sensitivity != null && (
                  <p className="mt-2 text-[11px] text-slate-500" title={`Measured on ${f.test_positives} held-out test eyes with this condition`}>
                    Tested: finds {Math.round(f.test_sensitivity * 100)}% · clears {Math.round((f.test_specificity ?? 0) * 100)}% of healthy
                    {f.test_positives < 15 ? ` · only ${f.test_positives} test cases` : ''}
                  </p>
                )}

                {isDetected && (
                  <div className="mt-3 flex flex-wrap items-center gap-2">
                    {f.severity && f.severity !== 'None' && (
                      <span className={`rounded px-2 py-0.5 text-[11px] font-semibold ring-1 ring-inset ${getSeverityClass(f.severity)}`}>
                        {f.severity}
                      </span>
                    )}
                    <span className={`inline-flex items-center gap-1.5 rounded px-2 py-0.5 text-[11px] font-semibold ring-1 ring-inset ${urgency.className}`}>
                      <span className={`h-1.5 w-1.5 rounded-full ${urgency.dot}`} />
                      {urgency.label}
                    </span>
                  </div>
                )}

                {isDetected && (f.hallmark_findings?.length > 0 || f.clinical_action) && (
                  <>
                    <button
                      type="button"
                      onClick={() => setExpandedId(isExpanded ? null : f.disease_id)}
                      className="mt-3 inline-flex items-center gap-1 text-xs font-semibold text-sky-400 hover:text-sky-300"
                    >
                      {isExpanded ? 'Hide details' : 'Findings & next steps'}
                      {isExpanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                    </button>

                    {isExpanded && (
                      <div className="mt-3 space-y-3 border-t border-slate-800 pt-3 text-xs">
                        {f.hallmark_findings?.length > 0 && (
                          <div>
                            <p className="mb-1 font-semibold uppercase tracking-wider text-slate-500">Hallmark signs</p>
                            <ul className="list-disc space-y-1 pl-4 text-slate-300">
                              {f.hallmark_findings.map((h, i) => <li key={i}>{h}</li>)}
                            </ul>
                          </div>
                        )}
                        {f.clinical_action && (
                          <div>
                            <p className="mb-1 font-semibold uppercase tracking-wider text-slate-500">Recommended action</p>
                            <p className="leading-relaxed text-slate-300">{f.clinical_action}</p>
                          </div>
                        )}
                      </div>
                    )}
                  </>
                )}
              </article>
            );
          })}
        </div>
      )}

      {clear.length > 0 && (
        <div>
          <button
            type="button"
            onClick={() => setShowClear((v) => !v)}
            className="inline-flex items-center gap-1 text-xs font-semibold text-slate-400 hover:text-white"
          >
            {showClear ? 'Hide' : 'Show'} the {clear.length} conditions checked and found clear
            {showClear ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
          </button>
          {showClear && (
            <ul className="mt-2 grid grid-cols-1 gap-x-6 gap-y-1 text-xs sm:grid-cols-2">
              {clear.map((f) => (
                <li key={f.disease_id} className="flex items-center justify-between gap-3 border-b border-slate-800/70 py-1.5">
                  <span className="flex items-center gap-1.5 text-slate-400">
                    <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500/70" />
                    {getDiseaseMeta(f.disease_id, f.disease_name).name}
                  </span>
                  <span className="font-mono text-slate-500">{Math.round((f.confidence ?? 0) * 100)}%</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}

      {uncovered.length > 0 && (
        <p className="text-[11px] text-slate-500">
          <span className="font-semibold text-slate-300">Not assessed by the AI model</span> (no labelled training images):{' '}
          {uncovered.map((id) => getDiseaseMeta(id).name).join(', ')}.
        </p>
      )}
    </section>
  );
}
