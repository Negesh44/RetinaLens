import React from 'react';
import { Eye, EyeOff, Info, Check, CheckSquare, Square, Layers } from 'lucide-react';

import { getDiseaseMeta } from '../lib/diseases';

export const LESION_METADATA = {
  // Diabetic Retinopathy & Microvascular
  hemorrhage: {
    id: 'hemorrhage',
    name: 'Retinal Hemorrhage',
    shortName: 'Hemorrhage',
    category: 'Vascular / DR',
    color: '#ef4444',
    bgLight: 'bg-red-500/15',
    border: 'border-red-500/40',
    text: 'text-red-400',
    dot: 'bg-red-500',
    description: 'Ruptured capillary walls causing intraretinal pooling of deoxygenated blood.',
  },
  hard_exudate: {
    id: 'hard_exudate',
    name: 'Hard Exudate',
    shortName: 'Hard Exudate',
    category: 'Vascular / DR',
    color: '#eab308',
    bgLight: 'bg-yellow-500/15',
    border: 'border-yellow-500/40',
    text: 'text-yellow-400',
    dot: 'bg-yellow-400',
    description: 'Waxy lipid and lipoprotein leakage from chronically incompetent microvessels.',
  },
  microaneurysm: {
    id: 'microaneurysm',
    name: 'Microaneurysm',
    shortName: 'Microaneurysm',
    category: 'Vascular / DR',
    color: '#3b82f6',
    bgLight: 'bg-blue-500/15',
    border: 'border-blue-500/40',
    text: 'text-blue-400',
    dot: 'bg-blue-500',
    description: 'Focal dilation of retinal capillaries; earliest hallmark sign of DR.',
  },
  cotton_wool_spot: {
    id: 'cotton_wool_spot',
    name: 'Cotton-Wool Spot',
    shortName: 'Cotton-Wool',
    category: 'Vascular / DR',
    color: '#c084fc',
    bgLight: 'bg-purple-500/15',
    border: 'border-purple-500/40',
    text: 'text-purple-400',
    dot: 'bg-purple-400',
    description: 'Localized nerve fiber layer axoplasmic stasis infarct secondary to precapillary occlusion.',
  },

  // Age-Related Macular Degeneration (AMD)
  drusen: {
    id: 'drusen',
    name: 'Macular Drusen',
    shortName: 'Drusen',
    category: 'Macular / AMD',
    color: '#facc15',
    bgLight: 'bg-amber-400/15',
    border: 'border-amber-400/40',
    text: 'text-amber-300',
    dot: 'bg-amber-300',
    description: 'Extracellular lipid and protein deposits beneath the retinal pigment epithelium.',
  },
  geographic_atrophy: {
    id: 'geographic_atrophy',
    name: 'Geographic Atrophy',
    shortName: 'Geo. Atrophy',
    category: 'Macular / AMD',
    color: '#fb923c',
    bgLight: 'bg-orange-500/15',
    border: 'border-orange-500/40',
    text: 'text-orange-400',
    dot: 'bg-orange-400',
    description: 'Well-demarcated thinning and loss of retinal photoreceptors and underlying choriocapillaris.',
  },

  // Glaucoma & Optic Neuropathy
  optic_cup_enlargement: {
    id: 'optic_cup_enlargement',
    name: 'Optic Cup Excavation (CDR > 0.65)',
    shortName: 'Cup Excavation',
    category: 'Glaucoma',
    color: '#f43f5e',
    bgLight: 'bg-rose-500/15',
    border: 'border-rose-500/40',
    text: 'text-rose-400',
    dot: 'bg-rose-500',
    description: 'Enlargement of central physiological cup indicating ganglion cell loss and optic nerve cupping.',
  },
  neuroretinal_rim_thinning: {
    id: 'neuroretinal_rim_thinning',
    name: 'Neuroretinal Rim Thinning',
    shortName: 'Rim Thinning',
    category: 'Glaucoma',
    color: '#e11d48',
    bgLight: 'bg-rose-600/15',
    border: 'border-rose-600/40',
    text: 'text-rose-300',
    dot: 'bg-rose-600',
    description: 'Focal notching or thinning of the inferior or superior neuroretinal disc border (ISNT rule violation).',
  },

  // Vitreoretinal & Detachment
  retinal_detachment_fold: {
    id: 'retinal_detachment_fold',
    name: 'Corrugated Detached Retina',
    shortName: 'Detachment Fold',
    category: 'Vitreoretinal',
    color: '#38bdf8',
    bgLight: 'bg-sky-500/15',
    border: 'border-sky-500/40',
    text: 'text-sky-300',
    dot: 'bg-sky-400',
    description: 'Pale, undulating translucent elevation of the neurosensory retina separated from the RPE.',
  },
  retinal_tear: {
    id: 'retinal_tear',
    name: 'Horseshoe Retinal Tear',
    shortName: 'Retinal Tear',
    category: 'Vitreoretinal',
    color: '#b91c1c',
    bgLight: 'bg-red-800/20',
    border: 'border-red-600/50',
    text: 'text-red-400',
    dot: 'bg-red-700',
    description: 'Full-thickness retinal defect with a vitreoretinal tractional flap exposing underlying dark choroid.',
  },
  macular_hole: {
    id: 'macular_hole',
    name: 'Full-Thickness Macular Hole',
    shortName: 'Macular Hole',
    category: 'Vitreoretinal',
    color: '#ea580c',
    bgLight: 'bg-orange-600/15',
    border: 'border-orange-600/40',
    text: 'text-orange-400',
    dot: 'bg-orange-600',
    description: 'Sharply circumscribed circular foveal aperture surrounded by a gray cuff of subretinal fluid.',
  },
  epiretinal_membrane: {
    id: 'epiretinal_membrane',
    name: 'Epiretinal Membrane Pucker',
    shortName: 'Macular Pucker',
    category: 'Vitreoretinal',
    color: '#94a3b8',
    bgLight: 'bg-slate-400/15',
    border: 'border-slate-400/40',
    text: 'text-slate-300',
    dot: 'bg-slate-400',
    description: 'Avascular semi-translucent fibrocellular sheet causing surface wrinkling and retinal vascular tortuosity.',
  },

  // Vascular Occlusions & Chorioretinopathy
  vascular_occlusion_ischemia: {
    id: 'vascular_occlusion_ischemia',
    name: 'Retinal Vein Occlusion Ischemia',
    shortName: 'Venous Occlusion',
    category: 'Vascular / RVO',
    color: '#dc2626',
    bgLight: 'bg-red-600/15',
    border: 'border-red-600/40',
    text: 'text-red-300',
    dot: 'bg-red-600',
    description: 'Massive sectorial intraretinal flame and blot hemorrhages along an obstructed venous branch.',
  },
  serous_detachment_blister: {
    id: 'serous_detachment_blister',
    name: 'Serous Fluid Blister (CSC)',
    shortName: 'CSC Blister',
    category: 'Chorioretinal / CSC',
    color: '#06b6d4',
    bgLight: 'bg-cyan-500/15',
    border: 'border-cyan-500/40',
    text: 'text-cyan-300',
    dot: 'bg-cyan-400',
    description: 'Dome-shaped blister of transparent subretinal fluid in posterior pole from focal RPE barrier breakdown.',
  },

  // Inherited Dystrophies & Ocular Oncology
  bone_spicule_pigment: {
    id: 'bone_spicule_pigment',
    name: 'Bone-Spicule Pigmentation',
    shortName: 'Bone Spicules',
    category: 'Dystrophy / RP',
    color: '#475569',
    bgLight: 'bg-slate-600/20',
    border: 'border-slate-500/40',
    text: 'text-slate-300',
    dot: 'bg-slate-500',
    description: 'Mid-peripheral perivascular melanin clusters from degenerating RPE migration in Retinitis Pigmentosa.',
  },
  ocular_tumor_mass: {
    id: 'ocular_tumor_mass',
    name: 'Ocular Tumor / Melanoma Mass',
    shortName: 'Tumor Mass',
    category: 'Ocular Oncology',
    color: '#1e293b',
    bgLight: 'bg-slate-800/40',
    border: 'border-slate-600/50',
    text: 'text-slate-200',
    dot: 'bg-slate-700',
    description: 'Elevated slate-gray pigmented choroidal mass with overlying orange lipofuscin deposits.',
  },
  arteriovenous_nicking: {
    id: 'arteriovenous_nicking',
    name: 'Arteriovenous Nicking (HTN)',
    shortName: 'A/V Nicking',
    category: 'Hypertensive',
    color: '#d97706',
    bgLight: 'bg-amber-600/15',
    border: 'border-amber-600/40',
    text: 'text-amber-400',
    dot: 'bg-amber-600',
    description: 'Venous tapering and deflection at crossing points caused by thickened sclerotic arterial wall.',
  },
  copper_wiring: {
    id: 'copper_wiring',
    name: 'Copper / Silver Wiring',
    shortName: 'Copper Wiring',
    category: 'Hypertensive',
    color: '#b45309',
    bgLight: 'bg-amber-700/15',
    border: 'border-amber-700/40',
    text: 'text-amber-300',
    dot: 'bg-amber-700',
    description: 'Bright diffuse burnished light reflex along retinal arterioles from progressive hyaline arteriolosclerosis.',
  },

  // Pathologic Myopia & Degenerative
  myopic_chorioretinal_atrophy: {
    id: 'myopic_chorioretinal_atrophy',
    name: 'Myopic Chorioretinal Atrophy',
    shortName: 'Myopic Atrophy',
    category: 'Myopia',
    color: '#0d9488',
    bgLight: 'bg-teal-500/15',
    border: 'border-teal-500/40',
    text: 'text-teal-300',
    dot: 'bg-teal-500',
    description: 'Patchy bare sclera and choroidal exposure due to mechanical posterior pole elongation and lacquer cracks.',
  },

  // Diabetic Macular Edema (CSME)
  circinate_exudate_ring: {
    id: 'circinate_exudate_ring',
    name: 'Circinate Hard Exudate Ring',
    shortName: 'Circinate Ring',
    category: 'Vascular / DME',
    color: '#f59e0b',
    bgLight: 'bg-amber-500/15',
    border: 'border-amber-500/40',
    text: 'text-amber-300',
    dot: 'bg-amber-500',
    description: 'Circular lipid-rich hard exudate ring encircling leaking microaneurysm clusters threatening central fovea.',
  },

  // Neuro-Ophthalmic & Optic Disc Edema
  disc_edema_margin_blur: {
    id: 'disc_edema_margin_blur',
    name: 'Optic Disc Margin Elevation & Blur',
    shortName: 'Disc Margin Blur',
    category: 'Neuro-Ophthalmic',
    color: '#dc2626',
    bgLight: 'bg-red-600/15',
    border: 'border-red-600/40',
    text: 'text-red-400',
    dot: 'bg-red-600',
    description: 'Circumferential blurring and elevated opacification of the neuroretinal rim obscuring crossing retinal vessels.',
  },

  // Chorioretinitis & Uveitis
  chorioretinal_scar: {
    id: 'chorioretinal_scar',
    name: 'Chorioretinal Inflammatory Scar',
    shortName: 'Chorioretinal Scar',
    category: 'Chorioretinitis',
    color: '#8b5cf6',
    bgLight: 'bg-purple-500/15',
    border: 'border-purple-500/40',
    text: 'text-purple-300',
    dot: 'bg-purple-500',
    description: 'Circumscribed atrophic lesion with hyperpigmented borders indicating resolved or satellite active necrotizing retinitis.',
  },
  // Trained-classifier explainability: where the model looked for a detected condition
  attention_region: {
    id: 'attention_region',
    name: 'AI Attention Area',
    shortName: 'AI Attention',
    category: 'Model explainability',
    color: '#38bdf8',
    bgLight: 'bg-sky-500/15',
    border: 'border-sky-500/40',
    text: 'text-sky-400',
    dot: 'bg-sky-400',
    description: 'Region the trained model relied on most for a detected condition. Not a measured lesion outline.',
  },
};


// Display metadata for a lesion; AI attention areas take the colour of the condition they explain.
export function getLesionMeta(lesion) {
  const meta = LESION_METADATA[lesion?.class_name] || LESION_METADATA.hemorrhage;
  if (lesion?.class_name === 'attention_region' && lesion.associated_disease) {
    return { ...meta, color: getDiseaseMeta(lesion.associated_disease).color };
  }
  return meta;
}

export default function LesionLegend({
  classSummaries = {},
  visibleClasses = {},
  onToggleClass,
  onToggleAll,
  showLabels,
  onToggleLabels,
  showMaskFill,
  onToggleMaskFill,
}) {
  // Only layers that actually have findings on this scan.
  const displayClasses = Object.keys(classSummaries).filter(
    (k) => classSummaries[k]?.count > 0 && LESION_METADATA[k]
  );
  if (displayClasses.length === 0) return null;
  const allVisible = displayClasses.every((c) => visibleClasses[c] !== false);

  return (
    <div className="bg-slate-900/90 backdrop-blur-md rounded-2xl border border-slate-800 p-4 shadow-xl space-y-3.5">
      <div className="flex items-center justify-between pb-2.5 border-b border-slate-800/80">
        <div className="flex items-center space-x-2">
          <div className="w-2.5 h-2.5 rounded-full bg-sky-400 animate-pulse" />
          <h4 className="text-xs font-heading font-bold uppercase tracking-wider text-slate-200">
            Pathological Biomarkers &amp; Detection Layers
          </h4>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => onToggleAll(!allVisible)}
            className="text-[11px] font-heading font-semibold text-sky-400 hover:text-sky-300 transition-colors flex items-center space-x-1"
          >
            {allVisible ? <EyeOff className="w-3.5 h-3.5" /> : <Eye className="w-3.5 h-3.5" />}
            <span>{allVisible ? 'Hide All' : 'Show All'}</span>
          </button>
        </div>
      </div>

      {/* Class Items Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
        {displayClasses.map((clsKey) => {
          const meta = LESION_METADATA[clsKey];
          const summary = classSummaries[clsKey] || { count: 0, affected_area_percentage: 0 };
          const isVisible = visibleClasses[clsKey] !== false;

          return (
            <div
              key={clsKey}
              onClick={() => onToggleClass(clsKey)}
              className={`group flex items-center justify-between p-2.5 rounded-xl border transition-all cursor-pointer select-none ${
                isVisible
                  ? `${meta.bgLight} ${meta.border} shadow-sm`
                  : 'bg-slate-950/40 border-slate-800/60 opacity-50 hover:opacity-80'
              }`}
            >
              <div className="flex items-center space-x-2.5 min-w-0 pr-2">
                <span
                  className="w-3 h-3 rounded-full flex-shrink-0 shadow-sm"
                  style={{ backgroundColor: meta.color }}
                />
                <div className="min-w-0">
                  <div className="flex items-center space-x-1.5">
                    <span className="text-xs font-heading font-bold text-white truncate group-hover:text-sky-300 transition-colors">
                      {meta.name}
                    </span>
                    {meta.category && (
                      <span className="text-[9px] px-1.5 py-0.2 rounded bg-slate-800/80 text-slate-400 font-sans">
                        {meta.category}
                      </span>
                    )}
                  </div>
                  <span className="text-[10px] text-slate-400 block truncate font-sans">
                    {meta.description}
                  </span>
                </div>
              </div>

              <div className="flex items-center space-x-2 flex-shrink-0">
                <div className="text-right">
                  <span className="text-xs font-black font-mono text-white tabular-nums block">
                    {summary.count}
                  </span>
                  <span className="text-[10px] text-slate-400 font-mono tabular-nums">
                    {summary.affected_area_percentage?.toFixed(2) || '0.00'}%
                  </span>
                </div>
                <div
                  className={`w-4 h-4 rounded flex items-center justify-center transition-colors ${
                    isVisible ? 'bg-sky-500 text-white' : 'border border-slate-700 bg-slate-800 text-transparent'
                  }`}
                >
                  <Check className="w-3 h-3 stroke-[3]" />
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Layer Controls: Labels & Fill Opacity */}
      <div className="pt-2 border-t border-slate-800/70 flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex items-center space-x-4">
          <label className="flex items-center space-x-2 cursor-pointer text-slate-300 hover:text-white select-none">
            <input
              type="checkbox"
              checked={showLabels}
              onChange={(e) => onToggleLabels(e.target.checked)}
              className="rounded border-slate-700 bg-slate-800 text-sky-500 focus:ring-sky-500 w-3.5 h-3.5"
            />
            <span className="text-[11px] font-sans font-medium">Show Finding ID Badges</span>
          </label>

          <label className="flex items-center space-x-2 cursor-pointer text-slate-300 hover:text-white select-none">
            <input
              type="checkbox"
              checked={showMaskFill}
              onChange={(e) => onToggleMaskFill(e.target.checked)}
              className="rounded border-slate-700 bg-slate-800 text-sky-500 focus:ring-sky-500 w-3.5 h-3.5"
            />
            <span className="text-[11px] font-sans font-medium">Render Semi-Transparent Masks</span>
          </label>
        </div>

        <div className="text-[10px] text-slate-500 font-mono">
          All coordinates normalized [0.0, 1.0]
        </div>
      </div>
    </div>
  );
}
