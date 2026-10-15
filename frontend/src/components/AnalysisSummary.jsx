import React from 'react';
import {
  Activity,
  AlertTriangle,
  Award,
  CheckCircle2,
  Cpu,
  Layers,
  Percent,
  ShieldAlert,
  Sparkles,
} from 'lucide-react';
import { LESION_METADATA } from './LesionLegend';

export default function AnalysisSummary({
  result,
  selectedLesion,
}) {
  if (!result) return null;

  const isRejected = result.status === 'rejected_for_quality';
  const grade = result.grade ?? result.retinal_grade ?? 0;
  const lesionCount = result.lesion_count ?? result.lesions?.length ?? 0;
  const overallAreaPct = result.overall_affected_area_percentage;
  const validPixels = result.valid_retinal_pixels;
  const totalPixels = result.total_image_pixels;

  const gradeNames = [
    'No Apparent DR (Grade 0)',
    'Mild NPDR (Grade 1)',
    'Moderate NPDR (Grade 2)',
    'Severe NPDR (Grade 3)',
    'Proliferative DR (Grade 4)',
  ];

  const gradeBadges = [
    'bg-emerald-500/20 text-emerald-300 border-emerald-500/40',
    'bg-blue-500/20 text-blue-300 border-blue-500/40',
    'bg-amber-500/20 text-amber-300 border-amber-500/40',
    'bg-orange-500/20 text-orange-300 border-orange-500/40',
    'bg-rose-500/20 text-rose-300 border-rose-500/40',
  ];

  const findings = result.disease_findings || [];
  const fromModel = findings.some((f) => f.model_name);
  if (fromModel && !isRejected) {
    const normal = findings.find((f) => f.disease_id === 'normal');
    const abnormalP = normal ? 1 - normal.confidence : null;
    const flagged = findings.filter((f) => f.disease_id !== 'normal' && f.is_detected).length;
    const screened = findings.filter((f) => f.disease_id !== 'normal').length;
    const isAbnormal = result.primary_disease && result.primary_disease !== 'normal';
    const tile = 'bg-slate-900 rounded-xl border border-slate-800 p-4 space-y-1.5 shadow-lg';
    const label = 'text-[10px] font-heading font-bold uppercase tracking-wider text-slate-400 flex items-center space-x-1.5';
    return (
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3.5">
        <div className={tile}>
          <span className={label}><Activity className="w-3.5 h-3.5 text-sky-400" /><span>Retina status</span></span>
          <div className={`text-xl sm:text-2xl font-black font-heading ${isAbnormal ? 'text-amber-300' : 'text-emerald-400'}`}>
            {isAbnormal ? 'Abnormal' : 'Normal'}
          </div>
          {abnormalP != null && (
            <span className="text-[11px] text-slate-400 font-mono">{Math.round(abnormalP * 100)}% probability of any disease</span>
          )}
        </div>
        <div className={tile}>
          <span className={label}><ShieldAlert className="w-3.5 h-3.5 text-amber-400" /><span>Conditions flagged</span></span>
          <div className="text-xl sm:text-2xl font-black text-white font-mono tabular-nums">
            {flagged}<span className="text-sm text-slate-500"> / {screened}</span>
          </div>
          <span className="text-[11px] text-slate-400">conditions the AI checks for</span>
        </div>
        <div className={tile}>
          <span className={label}><Layers className="w-3.5 h-3.5 text-violet-400" /><span>AI attention areas</span></span>
          <div className="text-xl sm:text-2xl font-black text-white font-mono tabular-nums">{lesionCount}</div>
          <span className="text-[11px] text-slate-400">{lesionCount ? 'click them in the viewer below' : 'nothing to highlight'}</span>
        </div>
        <div className={tile}>
          <span className={label}><Cpu className="w-3.5 h-3.5 text-emerald-400" /><span>Retinal coverage</span></span>
          <div className="text-xl sm:text-2xl font-black text-emerald-400 font-mono tabular-nums">
            {validPixels ? `${Math.round((validPixels / (totalPixels || 1)) * 100)}%` : '100%'}
          </div>
          <span className="text-[11px] text-slate-400">of the photo is usable retina</span>
        </div>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 sm:grid-cols-4 gap-3.5">
      {/* 1. Overall Retinal Grade */}
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 space-y-1.5 shadow-lg hover:border-slate-700/80 transition-colors">
        <span className="text-[10px] font-heading font-bold uppercase tracking-wider text-slate-400 flex items-center space-x-1.5">
          <Award className="w-3.5 h-3.5 text-sky-400" />
          <span>DR Grade (ICDR)</span>
        </span>
        <div className="flex items-baseline space-x-2">
          <span className="text-xl sm:text-2xl font-black text-white font-heading tracking-tight">
            {isRejected ? 'N/A' : `Grade ${grade}`}
          </span>
          {!isRejected && result.confidence && (
            <span className="text-[11px] font-mono text-slate-400 tabular-nums">
              ({(result.confidence * 100).toFixed(0)}% conf)
            </span>
          )}
        </div>
        <div className="truncate">
          <span className={`inline-block text-[10px] font-semibold font-sans px-2.5 py-0.5 rounded-full border truncate max-w-full ${gradeBadges[Math.min(grade, 4)]}`}>
            {isRejected ? 'Ungradable' : gradeNames[grade]}
          </span>
        </div>
      </div>

      {/* 2. Detected Lesions Count */}
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 space-y-1.5 shadow-lg hover:border-slate-700/80 transition-colors">
        <span className="text-[10px] font-heading font-bold uppercase tracking-wider text-slate-400 flex items-center space-x-1.5">
          <Layers className="w-3.5 h-3.5 text-amber-400" />
          <span>Detected Lesions</span>
        </span>
        <div className="flex items-baseline space-x-2">
          <span className="text-xl sm:text-2xl font-black text-amber-400 font-mono tabular-nums">
            {isRejected ? '0' : lesionCount}
          </span>
          <span className="text-[11px] font-medium text-slate-400 font-sans">focal lesions</span>
        </div>
        <span className="text-[10px] text-slate-400 block truncate font-sans">
          {lesionCount === 0
            ? 'No focal lesions identified'
            : 'Interactive pins active'}
        </span>
      </div>

      {/* 3. Quantified Affected Area % */}
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 space-y-1.5 shadow-lg hover:border-slate-700/80 transition-colors">
        <span className="text-[10px] font-heading font-bold uppercase tracking-wider text-slate-400 flex items-center space-x-1.5">
          <Percent className="w-3.5 h-3.5 text-rose-400" />
          <span>Affected Retinal Area</span>
        </span>
        <div className="flex items-baseline space-x-2">
          <span className="text-xl sm:text-2xl font-black text-rose-400 font-mono tabular-nums">
            {overallAreaPct != null ? `${overallAreaPct.toFixed(2)}%` : 'Not available'}
          </span>
        </div>
        <span className="text-[10px] text-slate-400 block truncate font-sans" title="Union of lesion masks / valid retina">
          Union mask (zero double counting)
        </span>
      </div>

      {/* 4. Valid Retinal Denominator */}
      <div className="bg-slate-900 rounded-xl border border-slate-800 p-4 space-y-1.5 shadow-lg hover:border-slate-700/80 transition-colors">
        <span className="text-[10px] font-heading font-bold uppercase tracking-wider text-slate-400 flex items-center space-x-1.5">
          <Cpu className="w-3.5 h-3.5 text-emerald-400" />
          <span>Retinal Coverage</span>
        </span>
        <div className="flex items-baseline space-x-1.5">
          <span className="text-xl sm:text-2xl font-black text-emerald-400 font-mono tabular-nums">
            {validPixels ? `${Math.round((validPixels / (totalPixels || 1)) * 100)}%` : '100%'}
          </span>
          <span className="text-[10px] text-slate-400 font-sans">valid field</span>
        </div>
        <span className="text-[10px] text-slate-400 block truncate font-mono tabular-nums">
          {validPixels ? `${validPixels.toLocaleString()} disc px` : 'Calibrated'}
        </span>
      </div>
    </div>
  );
}
