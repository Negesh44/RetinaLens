import React from 'react';
import { AlertTriangle, CheckCircle2, ClipboardList, Stethoscope } from 'lucide-react';
import {
  getDetectedFindings,
  getDiseaseMeta,
  getSeverityClass,
  getUrgency,
} from '../lib/diseases';

function Chip({ className, children }) {
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-md px-2.5 py-1 text-xs font-semibold ring-1 ring-inset ${className}`}>
      {children}
    </span>
  );
}

export default function DiagnosisHero({ result }) {
  const findings = Array.isArray(result.disease_findings) ? result.disease_findings : [];
  const detected = getDetectedFindings(findings);
  const primaryId = result.primary_disease || (detected[0]?.disease_id ?? 'normal');
  const isNormal = primaryId === 'normal' && detected.length === 0;

  const fromModel = findings.some((f) => f.model_name);
  const abnormalNoType = primaryId === 'unspecified_abnormality';
  const normalFinding = findings.find((f) => f.disease_id === 'normal');

  // Older screenings stored no per-disease findings; derive the primary card from the DR grade.
  const primaryFinding = findings.find((f) => f.disease_id === primaryId) || (abnormalNoType ? {
    disease_id: primaryId,
    disease_name: 'Signs of retinal disease that the model could not match to one of its conditions',
    category: 'Unclassified',
    confidence: normalFinding ? 1 - normalFinding.confidence : 0,
    severity: 'None',
    triage_urgency: 'Prompt (2-4 wks)',
    clinical_action: 'Refer for a dilated fundus examination by an ophthalmologist to identify the abnormality.',
  } : {
    disease_id: primaryId,
    disease_name: result.primary_disease_name,
    acronym: primaryId === 'diabetic_retinopathy' ? 'DR' : '',
    category: primaryId === 'diabetic_retinopathy' ? 'Retinal Vascular Disease' : 'Physiological Baseline',
    confidence: result.confidence ?? 0,
    severity: isNormal ? 'None' : (result.grade_label?.split(' ')[0] || 'Detected'),
    triage_urgency: isNormal ? 'Routine' : (result.is_referable ? 'Prompt (2-4 wks)' : 'Routine'),
    clinical_action: result.referral_recommendation || '',
    key_biomarkers: [],
  });

  const meta = abnormalNoType ? { name: 'Retinal Abnormality', color: '#f59e0b' } : getDiseaseMeta(primaryId, primaryFinding.disease_name);
  const urgency = getUrgency(primaryFinding.triage_urgency);
  const confidence = primaryFinding.confidence ?? result.confidence ?? 0;
  const others = detected.filter((f) => f.disease_id !== primaryId);
  const screenedCount = findings.filter((f) => f.disease_id !== 'normal').length;
  // The separate ICDR grading model is not validated, so its grade is hidden when the trained classifier ran.
  const showDrGrade = !fromModel && primaryId === 'diabetic_retinopathy' && result.grade != null;

  const title = isNormal ? 'No Abnormality Detected' : meta.name;
  const action = isNormal
    ? 'Retina appears healthy. Routine annual rescreening is recommended.'
    : primaryFinding.clinical_action || result.referral_recommendation || 'Specialist ophthalmic review recommended.';

  return (
    <section
      aria-label="Primary diagnosis"
      className="relative overflow-hidden rounded-2xl border border-slate-800 bg-slate-900 shadow-xl"
    >
      <div className="h-1.5 w-full" style={{ backgroundColor: meta.color }} />

      <div
        className={`grid grid-cols-1 gap-6 p-5 sm:p-7 ${
          result.original_image_url ? 'lg:grid-cols-[auto_minmax(0,1fr)_300px]' : 'lg:grid-cols-[minmax(0,1fr)_300px]'
        }`}
      >
        {/* Scan thumbnail */}
        {result.original_image_url && (
          <div className="hidden sm:block">
            <img
              src={result.original_image_url}
              alt="Fundus scan"
              className="h-36 w-36 rounded-xl object-cover ring-2"
              style={{ '--tw-ring-color': `${meta.color}99` }}
            />
            <p className="mt-2 text-center font-mono text-[11px] text-slate-500">
              {result.eye_side && result.eye_side !== 'Unknown' ? result.eye_side : 'Eye N/A'}
            </p>
          </div>
        )}

        {/* Primary diagnosis */}
        <div className="min-w-0 space-y-4">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.14em] text-slate-400">
              {isNormal ? 'Screening result' : 'Primary diagnosis'}
            </p>
            <h2 className="mt-1.5 flex items-start gap-3 font-heading text-3xl font-extrabold leading-tight text-white sm:text-4xl">
              {isNormal ? (
                <CheckCircle2 className="mt-1 h-8 w-8 flex-shrink-0" style={{ color: meta.color }} />
              ) : (
                <Stethoscope className="mt-1 h-8 w-8 flex-shrink-0" style={{ color: meta.color }} />
              )}
              <span>{title}</span>
            </h2>
            <p className="mt-1.5 text-sm text-slate-400">
              {isNormal
                ? `No signs of disease across ${screenedCount || 10} screened retinal conditions.`
                : [primaryFinding.disease_name, primaryFinding.category].filter(Boolean).join(' · ')}
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {!isNormal && primaryFinding.severity && primaryFinding.severity !== 'None' && (
              <Chip className={getSeverityClass(primaryFinding.severity)}>
                Severity: {primaryFinding.severity}
              </Chip>
            )}
            <Chip className={urgency.className}>
              <span className={`h-1.5 w-1.5 rounded-full ${urgency.dot}`} />
              {urgency.label}
            </Chip>
            {showDrGrade && (
              <Chip className="bg-slate-800 text-slate-200 ring-slate-700">
                ICDR Grade {result.grade}{result.grade_label ? ` · ${result.grade_label}` : ''}
              </Chip>
            )}
          </div>

          {/* Confidence */}
          <div className="max-w-md">
            <div className="mb-1.5 flex items-baseline justify-between text-xs">
              <span className="font-medium text-slate-400">AI confidence</span>
              <span className="font-mono text-sm font-bold tabular-nums text-white">
                {(confidence * 100).toFixed(0)}%
              </span>
            </div>
            <div className="h-2 w-full overflow-hidden rounded-full bg-slate-800">
              <div
                className="h-full rounded-full transition-all"
                style={{ width: `${Math.min(100, Math.max(2, confidence * 100))}%`, backgroundColor: meta.color }}
              />
            </div>
          </div>

          {!isNormal && primaryFinding.test_sensitivity != null && (
            <p className="max-w-xl text-xs leading-relaxed text-slate-400">
              <span className="font-semibold text-slate-300">How reliable is this?</span> On {primaryFinding.test_positives} held-out
              test eyes with this condition, the model found {Math.round(primaryFinding.test_sensitivity * 100)}% of them, and it
              correctly cleared {Math.round((primaryFinding.test_specificity ?? 0) * 100)}% of eyes without it
              {primaryFinding.test_auc != null ? ` (AUC ${primaryFinding.test_auc.toFixed(2)})` : ''}.
            </p>
          )}

          <div className="flex items-start gap-3 rounded-xl border border-slate-800 bg-slate-950/60 p-4">
            <ClipboardList className="mt-0.5 h-5 w-5 flex-shrink-0 text-slate-400" />
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider text-slate-400">Recommended next step</p>
              <p className="mt-1 text-sm leading-relaxed text-slate-200">{action}</p>
            </div>
          </div>

          {/* The uncertainty flag comes from the DR grading model, so it only qualifies a DR diagnosis. */}
          {result.uncertain && primaryId === 'diabetic_retinopathy' && (
            <div className="flex items-start gap-2.5 rounded-xl border border-amber-500/30 bg-amber-500/10 p-3 text-sm text-amber-200">
              <AlertTriangle className="mt-0.5 h-4 w-4 flex-shrink-0 text-amber-400" />
              <span><b className="font-semibold">Grade uncertain (ICDR {result.grade}):</b> {result.uncertainty_note || 'Low grading confidence. Specialist review recommended.'}</span>
            </div>
          )}
        </div>

        {/* Other detected conditions */}
        <aside className="rounded-xl border border-slate-800 bg-slate-950/50 p-4 lg:self-start">
          <div className="mb-3 flex items-center justify-between">
            <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">Also detected</h3>
            <span className="font-mono text-xs text-slate-500">
              {detected.length} of {screenedCount || 10} flagged
            </span>
          </div>

          {others.length === 0 ? (
            <p className="text-sm text-slate-400">
              {isNormal ? 'All screened conditions are clear.' : 'No other conditions flagged.'}
            </p>
          ) : (
            <ul className="space-y-2">
              {others.map((f) => {
                const m = getDiseaseMeta(f.disease_id, f.disease_name);
                const u = getUrgency(f.triage_urgency);
                return (
                  <li key={f.disease_id} className="flex items-center gap-3 rounded-lg bg-slate-900 px-3 py-2.5">
                    <span className="h-2.5 w-2.5 flex-shrink-0 rounded-full" style={{ backgroundColor: m.color }} />
                    <div className="min-w-0 flex-1">
                      <p className="text-sm font-semibold leading-snug text-slate-100">{m.name}</p>
                      <p className="text-[11px] text-slate-500">
                        {f.severity !== 'None' ? `${f.severity} · ` : ''}{u.short}
                      </p>
                    </div>
                    <span className="font-mono text-xs font-semibold tabular-nums text-slate-300">
                      {((f.confidence ?? 0) * 100).toFixed(0)}%
                    </span>
                  </li>
                );
              })}
            </ul>
          )}
        </aside>
      </div>
    </section>
  );
}
