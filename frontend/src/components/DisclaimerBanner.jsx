import React from 'react';
import { AlertTriangle } from 'lucide-react';

export default function DisclaimerBanner({ className = "" }) {
  return (
    <aside
      aria-label="Clinical Disclaimer"
      className={`bg-amber-950/40 border-l-4 border-amber-500/90 border-y border-r border-amber-900/40 p-3.5 rounded-r-2xl shadow-lg backdrop-blur-md ${className}`}
    >
      <div className="flex items-start space-x-3">
        <div className="w-6 h-6 rounded-lg bg-amber-500/20 text-amber-400 flex items-center justify-center flex-shrink-0 mt-0.5 border border-amber-500/30">
          <AlertTriangle className="w-3.5 h-3.5" />
        </div>
        <div className="text-xs text-amber-200/90 leading-relaxed font-sans">
          <span className="font-extrabold uppercase tracking-wider text-amber-400 mr-1.5 font-heading text-[11px]">
            Clinical Protocol Notice:
          </span>
          RetinaLens is an investigational screening aid engineered for qualified healthcare professionals.
          It does <b className="text-white font-bold">NOT</b> provide definitive diagnostic conclusions and cannot replace dilated fundus examinations performed by an ophthalmologist or licensed optometrist.
        </div>
      </div>
    </aside>
  );
}

