import React from 'react';
import {
  AlertCircle,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  Crosshair,
  ExternalLink,
  HelpCircle,
  MapPin,
  Maximize2,
  Percent,
  ShieldAlert,
  Sparkles,
  Zap,
} from 'lucide-react';
import { getLesionMeta } from './LesionLegend';

export default function LesionDetailsPanel({
  selectedLesion,
  selectedCoordinate,
  onSelectLesion,
  allLesions = [],
  onClearSelection,
}) {
  // Find current lesion index for prev/next cycling
  const currentIndex = selectedLesion
    ? allLesions.findIndex((l) => l.id === selectedLesion.id)
    : -1;

  const handlePrev = () => {
    if (allLesions.length === 0) return;
    const nextIdx = (currentIndex - 1 + allLesions.length) % allLesions.length;
    onSelectLesion(allLesions[nextIdx]);
  };

  const handleNext = () => {
    if (allLesions.length === 0) return;
    const nextIdx = (currentIndex + 1) % allLesions.length;
    onSelectLesion(allLesions[nextIdx]);
  };

  // 1. STATE: Clicked an area with NO lesion detected
  if (!selectedLesion && selectedCoordinate) {
    const normX = (selectedCoordinate.normalizedX * 100).toFixed(1);
    const normY = (selectedCoordinate.normalizedY * 100).toFixed(1);
    const pxX = Math.round(selectedCoordinate.pixelX || 0);
    const pxY = Math.round(selectedCoordinate.pixelY || 0);

    return (
      <div className="bg-slate-900/95 backdrop-blur-md rounded-2xl border border-slate-800 p-5 shadow-xl space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div className="flex items-center space-x-2">
            <div className="w-8 h-8 rounded-xl bg-emerald-500/20 text-emerald-400 flex items-center justify-center">
              <CheckCircle2 className="w-4 h-4" />
            </div>
            <div>
              <h3 className="text-sm font-extrabold text-white">Retinal Coordinate Inspected</h3>
              <span className="text-[11px] text-slate-400">Interactive Click-to-Inspect</span>
            </div>
          </div>
          <button
            onClick={onClearSelection}
            className="text-xs text-slate-400 hover:text-white px-2 py-1 rounded bg-slate-800/80"
          >
            Clear
          </button>
        </div>

        <div className="bg-emerald-950/30 border border-emerald-500/30 rounded-xl p-4 space-y-2">
          <div className="flex items-center space-x-2 text-emerald-400 text-xs font-bold uppercase tracking-wider">
            <CheckCircle2 className="w-4 h-4" />
            <span>No Abnormality Detected</span>
          </div>
          <p className="text-xs text-slate-300 leading-relaxed">
            No microaneurysm, hemorrhage, exudate, or cotton-wool spot was detected at this location. The underlying retinal parenchyma displays standard physiological reflectance.
          </p>
        </div>

        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="p-2.5 bg-slate-950/60 rounded-xl border border-slate-800">
            <span className="text-[10px] text-slate-400 uppercase font-semibold block">Image Coordinates</span>
            <span className="font-mono font-bold text-slate-200">X: {pxX}px, Y: {pxY}px</span>
          </div>
          <div className="p-2.5 bg-slate-950/60 rounded-xl border border-slate-800">
            <span className="text-[10px] text-slate-400 uppercase font-semibold block">Normalized Field</span>
            <span className="font-mono font-bold text-slate-200">({normX}%, {normY}%)</span>
          </div>
        </div>

        <div className="text-[11px] text-slate-400 italic">
          💡 Click directly on any highlighted lesion boundary or badge to inspect its clinical metrics.
        </div>
      </div>
    );
  }

  // 2. STATE: No lesion and no coordinate selected
  if (!selectedLesion) {
    return (
      <div className="bg-slate-900/90 backdrop-blur-md rounded-2xl border border-slate-800 p-6 shadow-xl text-center space-y-3">
        <div className="w-12 h-12 mx-auto rounded-2xl bg-sky-500/10 text-sky-400 flex items-center justify-center border border-sky-500/20">
          <Crosshair className="w-6 h-6 animate-pulse" />
        </div>
        <div>
          <h3 className="text-sm font-bold text-white">Interactive Lesion Inspector</h3>
          <p className="text-xs text-slate-400 mt-1 max-w-xs mx-auto leading-relaxed">
            Click on any highlighted lesion overlay or lesion badge on the retinal fundus image to view its exact area, confidence score, and clinical explanation.
          </p>
        </div>
        {allLesions.length > 0 && (
          <div className="pt-2">
            <button
              onClick={() => onSelectLesion(allLesions[0])}
              className="px-4 py-2 bg-sky-600 hover:bg-sky-500 text-white rounded-xl text-xs font-bold transition-all shadow-md shadow-sky-600/20 inline-flex items-center space-x-1.5"
            >
              <span>Inspect First Lesion ({allLesions[0].display_name})</span>
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        )}
      </div>
    );
  }

  // 3. STATE: Lesion Selected
  const meta = getLesionMeta(selectedLesion);

  const severityBadgeStyles = {
    Mild: 'bg-blue-500/20 text-blue-300 border-blue-500/40',
    Moderate: 'bg-amber-500/20 text-amber-300 border-amber-500/40',
    Severe: 'bg-rose-500/20 text-rose-300 border-rose-500/40',
  }[selectedLesion.severity || 'Moderate'];

  const lesionNum = selectedLesion.id.replace('lesion-', 'L');

  return (
    <div className="bg-slate-900/95 backdrop-blur-md rounded-2xl border border-slate-800 p-5 shadow-2xl space-y-4">
      {/* Top Header with Navigation */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800">
        <div className="flex items-center space-x-2.5">
          <span
            className="w-3.5 h-3.5 rounded-full flex-shrink-0 shadow-lg"
            style={{ backgroundColor: meta.color }}
          />
          <div>
            <div className="flex items-center space-x-2">
              <h3 className="text-sm font-heading font-extrabold text-white">{selectedLesion.display_name}</h3>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-slate-800 text-sky-300 font-bold border border-slate-700">
                {lesionNum}
              </span>
            </div>
            <div className="flex items-center space-x-2 mt-0.5">
              <span className="text-[11px] text-slate-400 font-sans">{selectedLesion.location}</span>
              {selectedLesion.associated_disease && (
                <span className="text-[9px] font-heading font-bold px-1.5 py-0.5 rounded bg-sky-500/15 border border-sky-500/30 text-sky-300">
                  {selectedLesion.associated_disease.replace(/_/g, ' ').toUpperCase()}
                </span>
              )}
            </div>
          </div>
        </div>

        {/* Previous / Next Cycler */}
        {allLesions.length > 1 && (
          <div className="flex items-center space-x-1">
            <span className="text-[10px] text-slate-400 font-mono mr-1">
              {currentIndex + 1} of {allLesions.length}
            </span>
            <button
              onClick={handlePrev}
              title="Previous Lesion"
              className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
            >
              <ChevronLeft className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={handleNext}
              title="Next Lesion"
              className="p-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
            >
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>
        )}
      </div>

      {/* Primary Metrics Grid */}
      <div className="grid grid-cols-2 gap-2.5 text-xs">
        {/* Affected Area */}
        <div className="p-3 bg-slate-950/70 rounded-xl border border-slate-800/80 space-y-1">
          <span className="text-[10px] uppercase font-heading font-bold text-slate-400 tracking-wider flex items-center space-x-1">
            <Percent className="w-3 h-3 text-sky-400" />
            <span>Affected Area</span>
          </span>
          <div className="flex items-baseline space-x-1.5">
            <span className="text-lg font-black text-sky-400 font-mono tabular-nums">
              {selectedLesion.affected_area_percentage.toFixed(3)}%
            </span>
          </div>
          <span className="text-[10px] text-slate-400 block font-mono tabular-nums">
            {selectedLesion.area_pixels.toLocaleString()} retinal pixels
          </span>
        </div>

        {/* Model Confidence */}
        <div className="p-3 bg-slate-950/70 rounded-xl border border-slate-800/80 space-y-1">
          <span className="text-[10px] uppercase font-heading font-bold text-slate-400 tracking-wider flex items-center space-x-1">
            <Zap className="w-3 h-3 text-emerald-400" />
            <span>AI Confidence</span>
          </span>
          <div className="flex items-baseline space-x-1">
            <span className="text-lg font-black text-emerald-400 font-mono tabular-nums">
              {(selectedLesion.confidence * 100).toFixed(1)}%
            </span>
          </div>
          <span className={`inline-block text-[10px] font-sans font-semibold px-2 py-0.5 rounded border ${severityBadgeStyles}`}>
            {selectedLesion.severity} Severity
          </span>
        </div>
      </div>

      {/* Spatial Anatomical Coordinates */}
      <div className="p-3 bg-slate-950/50 rounded-xl border border-slate-800/80 space-y-2 text-xs">
        <span className="text-[10px] uppercase font-heading font-bold text-slate-400 tracking-wider flex items-center space-x-1">
          <MapPin className="w-3 h-3 text-amber-400" />
          <span>Spatial Coordinates &amp; Bounding Box</span>
        </span>
        <div className="grid grid-cols-2 gap-2 font-mono text-[11px] text-slate-300">
          <div>
            <span className="text-slate-500 block text-[10px]">Centroid:</span>
            <span className="tabular-nums">
              X: {selectedLesion.centroid_pixels?.[0] ?? Math.round(selectedLesion.normalized_centroid[0] * 300)}px,{' '}
              Y: {selectedLesion.centroid_pixels?.[1] ?? Math.round(selectedLesion.normalized_centroid[1] * 300)}px
            </span>
          </div>
          <div>
            <span className="text-slate-500 block text-[10px]">Normalized:</span>
            <span className="tabular-nums">
              ({(selectedLesion.normalized_centroid[0] * 100).toFixed(1)}%,{' '}
              {(selectedLesion.normalized_centroid[1] * 100).toFixed(1)}%)
            </span>
          </div>
        </div>
        {selectedLesion.bbox && (
          <div className="text-[10px] font-mono text-slate-400 pt-1 border-t border-slate-800/60 tabular-nums">
            BBox: [{selectedLesion.bbox.join(', ')}]
          </div>
        )}
      </div>

      {/* Medical Explanation */}
      <div className="p-3.5 bg-slate-950/40 rounded-xl border border-slate-800/80 space-y-1.5 text-xs font-sans">
        <span className="text-[10px] uppercase font-heading font-bold text-slate-300 tracking-wider block">
          Pathophysiological Finding:
        </span>
        <p className="text-slate-300 text-[11px] leading-relaxed">
          {selectedLesion.explanation}
        </p>
      </div>

      {/* Educational Clinical Guidance */}
      <div className="p-3 bg-sky-950/30 border border-sky-800/50 rounded-xl space-y-1 text-xs font-sans">
        <span className="text-[10px] uppercase font-heading font-bold text-sky-300 tracking-wider flex items-center space-x-1">
          <ShieldAlert className="w-3 h-3 text-sky-400" />
          <span>Educational Clinical Recommendation:</span>
        </span>
        <p className="text-slate-300 text-[11px] leading-relaxed">
          {selectedLesion.educational_advice}
        </p>
      </div>
    </div>
  );
}
