import React from 'react';
import { getLesionMeta } from './LesionLegend';

export default function LesionOverlay({
  lesions = [],
  viewBoxWidth = 1000,
  viewBoxHeight = 1000,
  selectedLesionId = null,
  hoveredLesionId = null,
  onSelectLesion,
  onHoverLesion,
  visibleClasses = {},
  showLabels = true,
  showMaskFill = true,
}) {
  return (
    <svg
      viewBox={`0 0 ${viewBoxWidth} ${viewBoxHeight}`}
      className="absolute inset-0 w-full h-full pointer-events-auto"
      style={{ overflow: 'visible' }}
    >
      <defs>
        {/* Glow Filters for Selected Lesions */}
        <filter id="glow-selected" x="-30%" y="-30%" width="160%" height="160%">
          <feGaussianBlur stdDeviation="6" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>

        <filter id="glow-hover" x="-20%" y="-20%" width="140%" height="140%">
          <feGaussianBlur stdDeviation="3" result="blur" />
          <feMerge>
            <feMergeNode in="blur" />
            <feMergeNode in="SourceGraphic" />
          </feMerge>
        </filter>
      </defs>

      {/* Render all visible lesions */}
      {lesions.map((lesion) => {
        // Skip if class visibility is toggled off
        if (visibleClasses[lesion.class_name] === false) {
          return null;
        }

        const meta = getLesionMeta(lesion);
        const isSelected = selectedLesionId === lesion.id;
        const isHovered = hoveredLesionId === lesion.id;

        // Convert normalized polygon points to viewBox space
        const pointsString = lesion.normalized_polygon
          ?.map(([nx, ny]) => `${nx * viewBoxWidth},${ny * viewBoxHeight}`)
          .join(' ');

        if (!pointsString) return null;

        // Centroid coordinates
        const [ncx, ncy] = lesion.normalized_centroid || [0.5, 0.5];
        const cx = ncx * viewBoxWidth;
        const cy = ncy * viewBoxHeight;
        const lesionTag = lesion.id.replace('lesion-', 'L');

        // Bounding box in viewBox space
        const [nbx1, nby1, nbx2, nby2] = lesion.normalized_bbox || [0, 0, 0, 0];
        const bx = nbx1 * viewBoxWidth;
        const by = nby1 * viewBoxHeight;
        const bw = (nbx2 - nbx1) * viewBoxWidth;
        const bh = (nby2 - nby1) * viewBoxHeight;

        return (
          <g
            key={lesion.id}
            onClick={(e) => {
              e.stopPropagation();
              onSelectLesion(lesion);
            }}
            onMouseEnter={() => onHoverLesion?.(lesion)}
            onMouseLeave={() => onHoverLesion?.(null)}
            className="cursor-pointer transition-all duration-150"
            style={{ outline: 'none' }}
          >
            {/* 1. Underlying filled polygon mask */}
            <polygon
              points={pointsString}
              fill={showMaskFill ? meta.color : 'transparent'}
              fillOpacity={isSelected ? 0.65 : isHovered ? 0.50 : 0.35}
              stroke={isSelected ? '#ffffff' : meta.color}
              strokeWidth={isSelected ? 3.5 : isHovered ? 2.5 : 1.8}
              strokeLinejoin="round"
              filter={isSelected ? 'url(#glow-selected)' : isHovered ? 'url(#glow-hover)' : undefined}
              className="transition-all duration-100"
            />

            {/* 2. Distinct Animated Pulse Outline on Selected Lesion */}
            {isSelected && (
              <>
                <polygon
                  points={pointsString}
                  fill="none"
                  stroke="#ffffff"
                  strokeWidth={2}
                  strokeDasharray="6,4"
                  className="animate-pulse"
                />
                {/* Bounding box guide */}
                <rect
                  x={bx - 4}
                  y={by - 4}
                  width={bw + 8}
                  height={bh + 8}
                  fill="none"
                  stroke={meta.color}
                  strokeWidth={1.2}
                  strokeDasharray="3,3"
                  opacity={0.8}
                  rx={4}
                />
              </>
            )}

            {/* 3. Lesion ID Badge at Centroid */}
            {showLabels && (
              <g transform={`translate(${cx}, ${cy})`}>
                <circle
                  r={isSelected ? 14 : isHovered ? 12 : 9}
                  fill="#0f172a"
                  stroke={isSelected ? '#ffffff' : meta.color}
                  strokeWidth={isSelected ? 2.5 : 1.5}
                  className="transition-all duration-100 shadow-md"
                />
                <text
                  textAnchor="middle"
                  dy={isSelected ? '4px' : '3.5px'}
                  fill="#ffffff"
                  fontSize={isSelected ? '11px' : isHovered ? '10px' : '8px'}
                  fontWeight="bold"
                  fontFamily="monospace"
                  pointerEvents="none"
                >
                  {lesionTag}
                </text>
              </g>
            )}
          </g>
        );
      })}
    </svg>
  );
}
