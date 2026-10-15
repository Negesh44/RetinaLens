import React from 'react';
import { Eye, Loader2 } from 'lucide-react';

const STEPS = [
  { title: 'Photo quality check', desc: 'Is the photo sharp, bright enough and showing the retina?' },
  { title: 'Retina detection', desc: 'Finds the circular retina area and crops the photo to it' },
  { title: 'Disease screening', desc: 'A trained deep-learning model checks for 10 eye conditions' },
  { title: 'AI attention map', desc: 'Marks the areas the model relied on for each finding' },
  { title: 'Report', desc: 'Builds the result page and the downloadable PDF' },
];

export default function AnalysisLoadingState({ currentStepMessage = 'Analysing the retina photo...' }) {
  return (
    <div className="mx-auto max-w-lg space-y-6 rounded-2xl border border-slate-800 bg-slate-900 p-8 text-center shadow-xl">
      <div className="relative mx-auto h-20 w-20">
        <div className="absolute inset-0 animate-ping rounded-full border-4 border-sky-500/20" />
        <div className="flex h-20 w-20 items-center justify-center rounded-full bg-gradient-to-tr from-sky-600 to-cyan-500 text-white shadow-lg">
          <Eye className="h-10 w-10 animate-pulse" />
        </div>
      </div>

      <div className="space-y-1.5">
        <h3 className="text-lg font-bold text-white">Analysing the retina photo</h3>
        <p className="animate-pulse font-mono text-xs font-medium text-sky-400">{currentStepMessage}</p>
      </div>

      <ol className="space-y-3 rounded-xl border border-slate-800 bg-slate-950/60 p-4 text-left">
        {STEPS.map((s) => (
          <li key={s.title} className="flex items-start gap-2.5 text-xs">
            <Loader2 className="mt-0.5 h-3.5 w-3.5 flex-shrink-0 animate-spin text-sky-400" />
            <div>
              <span className="block font-semibold text-slate-200">{s.title}</span>
              <span className="block leading-tight text-slate-400">{s.desc}</span>
            </div>
          </li>
        ))}
      </ol>
    </div>
  );
}
