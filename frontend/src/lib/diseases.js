// Display metadata for each condition in ml/pathologies.py DISEASE_CATALOG.
// Colors are chosen to stay distinguishable from each other on a dark surface.
export const DISEASE_META = {
  normal:                   { name: 'Normal Retina',                    color: '#10b981' },
  diabetic_retinopathy:     { name: 'Diabetic Retinopathy',             color: '#38bdf8' },
  amd:                      { name: 'Age-Related Macular Degeneration', color: '#f59e0b' },
  glaucoma:                 { name: 'Glaucoma',                         color: '#22d3ee' },
  retinal_detachment:       { name: 'Retinal Detachment',               color: '#f43f5e' },
  macular_hole_pucker:      { name: 'Macular Hole / Epiretinal Membrane', color: '#d946ef' },
  vascular_occlusion:       { name: 'Retinal Vascular Occlusion',       color: '#818cf8' },
  csc:                      { name: 'Central Serous Chorioretinopathy', color: '#2dd4bf' },
  inherited_dystrophy:      { name: 'Inherited Retinal Dystrophy',      color: '#a78bfa' },
  ocular_cancer:            { name: 'Ocular Tumor',                     color: '#ef4444' },
  hypertensive_retinopathy: { name: 'Hypertensive Retinopathy',         color: '#fb923c' },
  pathologic_myopia:        { name: 'Pathologic Myopia',                color: '#0d9488' },
  diabetic_macular_edema:   { name: 'Diabetic Macular Edema (CSME)',    color: '#eab308' },
  papilledema:              { name: 'Papilledema / Disc Edema',         color: '#dc2626' },
  chorioretinitis:          { name: 'Chorioretinitis / Uveitic Scar',   color: '#8b5cf6' },
  unassessable:             { name: 'Not Assessable',                   color: '#94a3b8' },
};

export function getDiseaseMeta(id, fallbackName) {
  return DISEASE_META[id] || { name: fallbackName || 'Unclassified Finding', color: '#94a3b8' };
}

export function isFindingDetected(finding) {
  // The trained classifier applies its own validated thresholds and consistency rule.
  if (finding?.model_name) return Boolean(finding.is_detected);
  return Boolean(finding?.is_detected) || (finding?.confidence ?? 0) >= 0.4;
}

// Maps backend triage_urgency strings to a display tier. Lower rank = more urgent.
export function getUrgency(urgency = 'Routine') {
  const u = urgency.toLowerCase();
  if (u.includes('emergency') || u.includes('same day')) {
    return { rank: 0, label: 'Emergency · Same day', short: 'Emergency', className: 'bg-red-500/15 text-red-300 ring-red-500/40', dot: 'bg-red-500' };
  }
  if (u.includes('urgent') || u.includes('24') || u.includes('48')) {
    return { rank: 1, label: 'Urgent · 24–48 hours', short: 'Urgent', className: 'bg-rose-500/15 text-rose-300 ring-rose-500/40', dot: 'bg-rose-500' };
  }
  if (u.includes('prompt') || u.includes('wk') || u.includes('week')) {
    return { rank: 2, label: 'Prompt · 2–4 weeks', short: 'Prompt', className: 'bg-amber-500/15 text-amber-300 ring-amber-500/40', dot: 'bg-amber-400' };
  }
  return { rank: 3, label: 'Routine follow-up', short: 'Routine', className: 'bg-emerald-500/15 text-emerald-300 ring-emerald-500/40', dot: 'bg-emerald-400' };
}

export function getSeverityClass(severity = 'None') {
  switch (severity) {
    case 'Critical': return 'bg-red-500/15 text-red-300 ring-red-500/40';
    case 'Severe':   return 'bg-orange-500/15 text-orange-300 ring-orange-500/40';
    case 'Moderate': return 'bg-amber-500/15 text-amber-300 ring-amber-500/40';
    case 'Mild':     return 'bg-sky-500/15 text-sky-300 ring-sky-500/40';
    default:         return 'bg-slate-700/40 text-slate-300 ring-slate-600/50';
  }
}

// Non-normal detected findings, most urgent first, then most confident.
export function getDetectedFindings(findings = []) {
  return (Array.isArray(findings) ? findings : [])
    .filter((f) => f.disease_id !== 'normal' && isFindingDetected(f))
    .sort((a, b) => {
      const ur = getUrgency(a.triage_urgency).rank - getUrgency(b.triage_urgency).rank;
      return ur !== 0 ? ur : (b.confidence ?? 0) - (a.confidence ?? 0);
    });
}
