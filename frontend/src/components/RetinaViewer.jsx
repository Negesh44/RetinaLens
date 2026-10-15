import React, { useState, useRef, useEffect, useLayoutEffect } from 'react';
import {
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Layers,
  Crosshair,
  Sliders,
  Columns,
  MousePointerClick,
  X,
  CheckCircle2,
  ChevronLeft,
  ChevronRight,
  MapPin,
} from 'lucide-react';
import LesionOverlay from './LesionOverlay';
import { getLesionMeta } from './LesionLegend';
import { getDiseaseMeta } from '../lib/diseases';

const POPOVER_WIDTH = 300;

// Ray-casting point-in-polygon test in normalized image space.
function pointInPolygon(x, y, poly) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

// Finds the lesion at a clicked point: a lesion whose outline contains the point wins
// (smallest first, so a microaneurysm inside a hemorrhage stays clickable); otherwise the
// nearest lesion whose centroid is within a small tolerance, so tiny findings are easy to hit.
function findLesionAt(nx, ny, lesions, aspect) {
  let best = null;
  let bestScore = Infinity;
  for (const lesion of lesions) {
    const [bx1, by1, bx2, by2] = lesion.normalized_bbox || [0, 0, 0, 0];
    const area = (bx2 - bx1) * (by2 - by1);
    if (lesion.normalized_polygon?.length > 2 && pointInPolygon(nx, ny, lesion.normalized_polygon)) {
      if (area - 10 < bestScore) {
        best = lesion;
        bestScore = area - 10; // always beats proximity matches
      }
      continue;
    }
    const [cx, cy] = lesion.normalized_centroid || [(bx1 + bx2) / 2, (by1 + by2) / 2];
    const dist = Math.hypot((nx - cx) * aspect, ny - cy);
    const radius = Math.max(0.025, Math.hypot((bx2 - bx1) * aspect, by2 - by1) / 2 + 0.01);
    if (dist <= radius && dist < bestScore) {
      best = lesion;
      bestScore = dist;
    }
  }
  return best;
}

function nearestLesion(nx, ny, lesions, aspect) {
  let best = null;
  let bestDist = Infinity;
  for (const lesion of lesions) {
    const [cx, cy] = lesion.normalized_centroid || [0.5, 0.5];
    const d = Math.hypot((nx - cx) * aspect, ny - cy);
    if (d < bestDist) {
      best = lesion;
      bestDist = d;
    }
  }
  return best;
}

function describeRegion(nx, ny) {
  const dx = nx - 0.5;
  const dy = ny - 0.5;
  if (Math.hypot(dx, dy) < 0.12) return 'Central retina';
  return `${dy < 0 ? 'Upper' : 'Lower'}-${dx < 0 ? 'left' : 'right'} retina`;
}

function InspectPopover({ anchor, left, width, lesion, coordinate, lesions, aspect, onSelectLesion, onClose }) {
  const style = { width, top: anchor.y, left };
  const shell =
    'absolute z-30 -translate-y-1/2 rounded-xl border border-slate-700 bg-slate-950/95 p-4 text-left shadow-2xl backdrop-blur-md';
  const stop = (e) => e.stopPropagation();

  if (lesion) {
    const meta = getLesionMeta(lesion);
    const disease = lesion.associated_disease ? getDiseaseMeta(lesion.associated_disease) : null;
    const index = lesions.findIndex((l) => l.id === lesion.id);
    const step = (delta) => onSelectLesion(lesions[(index + delta + lesions.length) % lesions.length]);

    return (
      <div className={shell} style={style} onClick={stop} onMouseDown={stop} onWheel={stop} role="dialog" aria-label={lesion.display_name}>
        <div className="flex items-start justify-between gap-2">
          <div className="flex items-center gap-2">
            <span className="h-3 w-3 flex-shrink-0 rounded-full" style={{ backgroundColor: meta.color }} />
            <h4 className="font-heading text-sm font-bold text-white">{lesion.display_name}</h4>
          </div>
          <button type="button" onClick={onClose} className="rounded p-0.5 text-slate-400 hover:bg-slate-800 hover:text-white" aria-label="Close">
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="mt-3 flex items-baseline justify-between">
          <span className="text-xs text-slate-400">Confidence</span>
          <span className="font-mono text-lg font-bold tabular-nums text-white">{Math.round((lesion.confidence ?? 0) * 100)}%</span>
        </div>
        <div className="mt-1 h-1.5 overflow-hidden rounded-full bg-slate-800">
          <div className="h-full rounded-full" style={{ width: `${(lesion.confidence ?? 0) * 100}%`, backgroundColor: meta.color }} />
        </div>

        <dl className="mt-3 space-y-1.5 text-xs">
          {lesion.location && (
            <div className="flex gap-2">
              <dt className="w-16 flex-shrink-0 text-slate-500">Location</dt>
              <dd className="text-slate-200">{lesion.location}</dd>
            </div>
          )}
          {lesion.severity && (
            <div className="flex gap-2">
              <dt className="w-16 flex-shrink-0 text-slate-500">Severity</dt>
              <dd className="text-slate-200">{lesion.severity}</dd>
            </div>
          )}
          {disease && (
            <div className="flex gap-2">
              <dt className="w-16 flex-shrink-0 text-slate-500">Sign of</dt>
              <dd className="flex items-center gap-1.5 text-slate-200">
                <span className="h-2 w-2 rounded-full" style={{ backgroundColor: disease.color }} />
                {disease.name}
              </dd>
            </div>
          )}
          {lesion.affected_area_percentage != null && (
            <div className="flex gap-2">
              <dt className="w-16 flex-shrink-0 text-slate-500">Area</dt>
              <dd className="font-mono text-slate-200">{lesion.affected_area_percentage.toFixed(2)}% of retina</dd>
            </div>
          )}
        </dl>

        {lesion.explanation && <p className="mt-3 text-xs leading-relaxed text-slate-300">{lesion.explanation}</p>}
        {lesion.educational_advice && (
          <p className="mt-2 rounded-lg bg-sky-500/10 p-2.5 text-xs leading-relaxed text-sky-100">
            <span className="font-semibold text-sky-300">Next step: </span>
            {lesion.educational_advice}
          </p>
        )}

        {lesions.length > 1 && (
          <div className="mt-3 flex items-center justify-between border-t border-slate-800 pt-2.5 text-xs text-slate-400">
            <span className="font-mono">Finding {index + 1} of {lesions.length}</span>
            <div className="flex gap-1">
              <button type="button" onClick={() => step(-1)} className="rounded-md bg-slate-800 p-1 hover:bg-slate-700 hover:text-white" aria-label="Previous finding">
                <ChevronLeft className="h-3.5 w-3.5" />
              </button>
              <button type="button" onClick={() => step(1)} className="rounded-md bg-slate-800 p-1 hover:bg-slate-700 hover:text-white" aria-label="Next finding">
                <ChevronRight className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        )}
      </div>
    );
  }

  const nearest = nearestLesion(coordinate.normalizedX, coordinate.normalizedY, lesions, aspect);
  const nearestMeta = nearest ? getLesionMeta(nearest) : null;

  return (
    <div className={shell} style={style} onClick={stop} onMouseDown={stop} onWheel={stop} role="dialog" aria-label="No finding at this spot">
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <CheckCircle2 className="h-4 w-4 text-emerald-400" />
          <h4 className="font-heading text-sm font-bold text-white">No finding at this spot</h4>
        </div>
        <button type="button" onClick={onClose} className="rounded p-0.5 text-slate-400 hover:bg-slate-800 hover:text-white" aria-label="Close">
          <X className="h-4 w-4" />
        </button>
      </div>
      <p className="mt-2 flex items-center gap-1.5 text-xs text-slate-400">
        <MapPin className="h-3.5 w-3.5" /> {describeRegion(coordinate.normalizedX, coordinate.normalizedY)}
      </p>
      <p className="mt-2 text-xs leading-relaxed text-slate-300">
        The AI did not mark any lesion here. Click on a dark-red, yellow, or pale spot to see what was found.
      </p>
      {nearest && (
        <button
          type="button"
          onClick={() => onSelectLesion(nearest)}
          className="mt-3 flex w-full items-center justify-between rounded-lg bg-slate-800 px-3 py-2 text-xs text-slate-200 transition-colors hover:bg-slate-700"
        >
          <span className="flex items-center gap-2">
            <span className="h-2 w-2 rounded-full" style={{ backgroundColor: nearestMeta.color }} />
            Nearest: {nearest.display_name}
          </span>
          <ChevronRight className="h-3.5 w-3.5" />
        </button>
      )}
    </div>
  );
}

const VIEW_MODES = [
  { id: 'inspect', label: 'Click to Inspect', icon: MousePointerClick },
  { id: 'analyzed', label: 'All Findings', icon: Layers },
  { id: 'heatmap', label: 'AI Heatmap', icon: Sliders },
  { id: 'split', label: 'Side-by-Side', icon: Columns },
];

export default function RetinaViewer({
  originalImageUrl,
  heatmapUrl,
  lesions = [],
  selectedLesion = null,
  onSelectLesion,
  selectedCoordinate = null,
  onSelectCoordinate,
  onClearSelection,
  visibleClasses = {},
  showLabels = true,
  showMaskFill = true,
}) {
  const containerRef = useRef(null);
  const dragMovedRef = useRef(false);

  const [viewMode, setViewMode] = useState('inspect');
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [heatmapOpacity, setHeatmapOpacity] = useState(0.55);
  const [hoveredLesion, setHoveredLesion] = useState(null);
  const [mouseCoords, setMouseCoords] = useState(null);
  const [overLesion, setOverLesion] = useState(false);
  const [naturalDimensions, setNaturalDimensions] = useState({ width: 1000, height: 1000 });
  const [box, setBox] = useState({ width: 0, height: 0 });

  const visibleLesions = lesions.filter((l) => visibleClasses[l.class_name] !== false);
  const aspect = naturalDimensions.width / naturalDimensions.height;

  // Track the canvas size so clicks map onto the letterboxed image correctly.
  useLayoutEffect(() => {
    const el = containerRef.current;
    if (!el) return undefined;
    const update = () => setBox({ width: el.clientWidth, height: el.clientHeight });
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, [viewMode]);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') onClearSelection?.();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClearSelection]);

  // The image is drawn with object-contain, so it is letterboxed inside the square canvas.
  const fit = Math.min(box.width / naturalDimensions.width, box.height / naturalDimensions.height) || 0;
  const img = {
    w: naturalDimensions.width * fit,
    h: naturalDimensions.height * fit,
  };
  img.x = (box.width - img.w) / 2;
  img.y = (box.height - img.h) / 2;

  const toNormalized = (clientX, clientY) => {
    const rect = containerRef.current?.getBoundingClientRect();
    if (!rect || !img.w) return null;
    const lx = (clientX - rect.left - pan.x) / zoom;
    const ly = (clientY - rect.top - pan.y) / zoom;
    const nx = (lx - img.x) / img.w;
    const ny = (ly - img.y) / img.h;
    return nx >= 0 && nx <= 1 && ny >= 0 && ny <= 1 ? { nx, ny } : null;
  };

  const toScreen = (nx, ny) => ({
    x: pan.x + (img.x + nx * img.w) * zoom,
    y: pan.y + (img.y + ny * img.h) * zoom,
  });

  const handleImageLoad = (e) => {
    if (e.target.naturalWidth && e.target.naturalHeight) {
      setNaturalDimensions({ width: e.target.naturalWidth, height: e.target.naturalHeight });
    }
  };

  const handleZoomIn = () => setZoom((z) => Math.min(5, Number((z + 0.35).toFixed(2))));
  const handleZoomOut = () => {
    setZoom((z) => {
      const next = Math.max(1, Number((z - 0.35).toFixed(2)));
      if (next === 1) setPan({ x: 0, y: 0 });
      return next;
    });
  };
  const handleResetZoom = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const handleWheel = (e) => {
    const delta = e.deltaY * -0.0015;
    setZoom((prev) => {
      const next = Math.min(Math.max(1, prev + delta), 5);
      if (next === 1) setPan({ x: 0, y: 0 });
      return Number(next.toFixed(2));
    });
  };

  const handleMouseDown = (e) => {
    dragMovedRef.current = false;
    if (zoom > 1 && e.button === 0) {
      setIsDragging(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e) => {
    if (isDragging) {
      dragMovedRef.current = true;
      setPan({ x: e.clientX - dragStart.x, y: e.clientY - dragStart.y });
    }
    const p = toNormalized(e.clientX, e.clientY);
    if (!p) {
      setMouseCoords(null);
      setOverLesion(false);
      return;
    }
    setMouseCoords({
      normX: p.nx,
      normY: p.ny,
      pxX: Math.round(p.nx * naturalDimensions.width),
      pxY: Math.round(p.ny * naturalDimensions.height),
    });
    if (viewMode === 'inspect') setOverLesion(Boolean(findLesionAt(p.nx, p.ny, visibleLesions, aspect)));
  };

  const handleMouseUp = () => setIsDragging(false);

  const handleContainerClick = (e) => {
    if (dragMovedRef.current) return; // the mouse-up ended a pan, not a click
    const p = toNormalized(e.clientX, e.clientY);
    if (!p) {
      onClearSelection?.();
      return;
    }
    const hit = findLesionAt(p.nx, p.ny, visibleLesions, aspect);
    if (hit) {
      onSelectLesion(hit);
    } else {
      onSelectCoordinate({
        normalizedX: p.nx,
        normalizedY: p.ny,
        pixelX: p.nx * naturalDimensions.width,
        pixelY: p.ny * naturalDimensions.height,
      });
    }
  };

  // Popover anchor: the selected lesion's centre, or the clicked spot.
  let anchor = null;
  if (viewMode !== 'split' && img.w) {
    if (selectedLesion) {
      const [bx1, by1, bx2, by2] = selectedLesion.normalized_bbox || [0, 0, 0, 0];
      const [cx, cy] = selectedLesion.normalized_centroid || [(bx1 + bx2) / 2, (by1 + by2) / 2];
      const s = toScreen(cx, cy);
      const r = Math.max(14, (Math.max((bx2 - bx1) * img.w, (by2 - by1) * img.h) / 2) * zoom + 8);
      anchor = { ...s, r };
    } else if (selectedCoordinate) {
      anchor = { ...toScreen(selectedCoordinate.normalizedX, selectedCoordinate.normalizedY), r: 12 };
    }
  }
  // Place the popover beside the ring on whichever side has room. A large finding (an AI attention
  // area can span most of the retina) may leave room on neither side; then clamp it inside the
  // frame, which clips anything that overflows.
  const popoverWidth = Math.min(POPOVER_WIDTH, Math.max(box.width - 16, 0));
  let popoverLeft = 0;
  if (anchor) {
    const rightLeft = anchor.x + anchor.r + 14;
    const leftLeft = anchor.x - anchor.r - 14 - popoverWidth;
    if (rightLeft + popoverWidth <= box.width - 8) popoverLeft = rightLeft;
    else if (leftLeft >= 8) popoverLeft = leftLeft;
    else popoverLeft = anchor.x < box.width / 2 ? rightLeft : leftLeft;
    popoverLeft = Math.min(Math.max(popoverLeft, 8), box.width - popoverWidth - 8);
  }
  const anchorColor = selectedLesion ? getLesionMeta(selectedLesion).color : '#34d399';
  // Keep the popover inside the frame vertically; the ring stays on the true spot.
  const popoverAnchor = anchor && { ...anchor, y: Math.min(Math.max(anchor.y, 150), Math.max(150, box.height - 150)) };

  const cursor = isDragging ? 'cursor-grabbing' : overLesion ? 'cursor-pointer' : zoom > 1 ? 'cursor-grab' : 'cursor-crosshair';

  return (
    <div className="space-y-4 rounded-2xl border border-slate-800 bg-slate-900 p-4 shadow-xl sm:p-5">
      {/* Toolbar */}
      <div className="flex flex-col gap-3 border-b border-slate-800 pb-3 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex flex-wrap items-center gap-1 rounded-lg border border-slate-800 bg-slate-950 p-1 text-xs font-semibold">
          {VIEW_MODES.map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              type="button"
              onClick={() => setViewMode(id)}
              className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 transition-colors ${
                viewMode === id ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <Icon className={`h-3.5 w-3.5 ${viewMode === id ? 'text-sky-400' : ''}`} />
              <span>{label}</span>
            </button>
          ))}
        </div>

        <div className="flex items-center gap-1 rounded-lg border border-slate-800 bg-slate-950 p-1 text-xs">
          <button type="button" onClick={handleZoomOut} disabled={zoom <= 1} title="Zoom out" className="rounded-md p-1.5 text-slate-300 transition-colors hover:bg-slate-800 disabled:opacity-40">
            <ZoomOut className="h-4 w-4" />
          </button>
          <span className="min-w-[3.25rem] text-center font-mono text-xs font-semibold tabular-nums text-slate-200">{Math.round(zoom * 100)}%</span>
          <button type="button" onClick={handleZoomIn} disabled={zoom >= 5} title="Zoom in" className="rounded-md p-1.5 text-slate-300 transition-colors hover:bg-slate-800 disabled:opacity-40">
            <ZoomIn className="h-4 w-4" />
          </button>
          <button type="button" onClick={handleResetZoom} title="Reset view" className="ml-1 rounded-md p-1.5 text-slate-400 transition-colors hover:bg-slate-800 hover:text-slate-200">
            <RotateCcw className="h-3.5 w-3.5" />
          </button>
        </div>
      </div>

      {viewMode === 'split' ? (
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <figure className="space-y-1.5">
            <figcaption className="text-xs font-semibold uppercase tracking-wider text-slate-400">Original photo</figcaption>
            <div className="flex aspect-square items-center justify-center overflow-hidden rounded-xl border border-slate-800 bg-black">
              <img src={originalImageUrl} alt="Original fundus" className="h-full w-full object-contain" />
            </div>
          </figure>
          <figure className="space-y-1.5">
            <figcaption className="text-xs font-semibold uppercase tracking-wider text-slate-400">AI findings</figcaption>
            <div className="relative flex aspect-square items-center justify-center overflow-hidden rounded-xl border border-slate-800 bg-black">
              <img src={originalImageUrl} alt="Fundus with findings" className="h-full w-full object-contain" />
              <LesionOverlay
                lesions={lesions}
                viewBoxWidth={naturalDimensions.width}
                viewBoxHeight={naturalDimensions.height}
                selectedLesionId={selectedLesion?.id}
                hoveredLesionId={hoveredLesion?.id}
                onSelectLesion={onSelectLesion}
                onHoverLesion={setHoveredLesion}
                visibleClasses={visibleClasses}
                showLabels={showLabels}
                showMaskFill={showMaskFill}
              />
            </div>
          </figure>
        </div>
      ) : (
        <div className="relative">
          {viewMode === 'inspect' && (
            <p className="mb-2 flex items-center justify-center gap-1.5 text-xs text-slate-400">
              <MousePointerClick className="h-3.5 w-3.5 text-sky-400" />
              Click any spot on the retina to see what the AI found there.
            </p>
          )}
          <div
            ref={containerRef}
            onWheel={handleWheel}
            onMouseDown={handleMouseDown}
            onMouseMove={handleMouseMove}
            onMouseUp={handleMouseUp}
            onMouseLeave={() => {
              handleMouseUp();
              setMouseCoords(null);
              setOverLesion(false);
            }}
            onClick={handleContainerClick}
            className={`relative mx-auto aspect-square max-w-2xl select-none overflow-hidden rounded-xl border border-slate-800 bg-black ${cursor}`}
          >
            <div
              style={{
                transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
                transformOrigin: 'top left',
                transition: isDragging ? 'none' : 'transform 75ms ease-out',
              }}
              className="relative h-full w-full"
            >
              <img
                src={originalImageUrl}
                alt="Retinal fundus scan"
                onLoad={handleImageLoad}
                className="pointer-events-none absolute inset-0 h-full w-full object-contain"
              />

              {viewMode === 'heatmap' && heatmapUrl && (
                <img
                  src={heatmapUrl}
                  alt="AI attention heatmap"
                  style={{ opacity: heatmapOpacity }}
                  className="pointer-events-none absolute inset-0 h-full w-full object-contain"
                />
              )}

              {viewMode === 'analyzed' && (
                <LesionOverlay
                  lesions={lesions}
                  viewBoxWidth={naturalDimensions.width}
                  viewBoxHeight={naturalDimensions.height}
                  selectedLesionId={selectedLesion?.id}
                  hoveredLesionId={hoveredLesion?.id}
                  onSelectLesion={onSelectLesion}
                  onHoverLesion={setHoveredLesion}
                  visibleClasses={visibleClasses}
                  showLabels={showLabels}
                  showMaskFill={showMaskFill}
                />
              )}
            </div>

            {/* Marker ring around the inspected spot (screen space, so it stays crisp when zoomed) */}
            {anchor && (
              <div
                className="pointer-events-none absolute z-20 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-white"
                style={{
                  left: anchor.x,
                  top: anchor.y,
                  width: anchor.r * 2,
                  height: anchor.r * 2,
                  boxShadow: `0 0 0 2px rgba(0,0,0,0.45), 0 0 14px ${anchorColor}`,
                }}
              />
            )}

            {anchor && (
              <InspectPopover
                anchor={popoverAnchor}
                left={popoverLeft}
                width={popoverWidth}
                lesion={selectedLesion}
                coordinate={selectedCoordinate}
                lesions={visibleLesions.length ? visibleLesions : lesions}
                aspect={aspect}
                onSelectLesion={onSelectLesion}
                onClose={() => onClearSelection?.()}
              />
            )}

            {viewMode === 'analyzed' && hoveredLesion && mouseCoords && !selectedLesion && (
              <div
                style={toScreen(mouseCoords.normX, mouseCoords.normY)}
                className="pointer-events-none absolute z-30 -translate-x-1/2 -translate-y-[130%] rounded-lg border border-slate-700 bg-slate-950/95 px-3 py-1.5 text-xs text-white shadow-xl"
              >
                <span className="font-semibold">{hoveredLesion.display_name}</span>
                <span className="ml-2 font-mono text-slate-400">{Math.round((hoveredLesion.confidence ?? 0) * 100)}%</span>
              </div>
            )}
          </div>

          {viewMode === 'heatmap' && (
            <div className="mt-3 space-y-2 rounded-xl border border-slate-800 bg-slate-950 p-3.5">
              <div className="flex items-center justify-between text-xs font-semibold text-slate-300">
                <span className="flex items-center gap-1.5">
                  <Sliders className="h-4 w-4 text-sky-400" /> Heatmap opacity
                </span>
                <span className="font-mono text-sky-400">{Math.round(heatmapOpacity * 100)}%</span>
              </div>
              <input
                type="range"
                min="0"
                max="1"
                step="0.02"
                value={heatmapOpacity}
                onChange={(e) => setHeatmapOpacity(parseFloat(e.target.value))}
                className="h-2 w-full cursor-pointer appearance-none rounded-lg bg-slate-800 accent-sky-500"
              />
            </div>
          )}
        </div>
      )}

      {/* Status bar */}
      <div className="flex flex-col gap-2 rounded-lg border border-slate-800 bg-slate-950 px-4 py-2.5 text-xs text-slate-400 sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-2">
          <Crosshair className="h-3.5 w-3.5 text-sky-400" />
          {mouseCoords ? (
            <span className="font-mono tabular-nums text-slate-300">
              {describeRegion(mouseCoords.normX, mouseCoords.normY)} · x {mouseCoords.pxX}, y {mouseCoords.pxY}
            </span>
          ) : (
            <span className="text-slate-500">Hover the retina to see the position</span>
          )}
        </div>
        <span className="font-mono">
          {visibleLesions.length} of {lesions.length} findings shown
        </span>
      </div>
    </div>
  );
}
