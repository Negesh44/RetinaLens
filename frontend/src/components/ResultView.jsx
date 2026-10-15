import React, { useState } from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  ShieldAlert,
  ArrowLeft,
} from 'lucide-react';
import RetinaViewer from './RetinaViewer';
import LesionLegend from './LesionLegend';
import LesionDetailsPanel from './LesionDetailsPanel';
import AnalysisSummary from './AnalysisSummary';
import MultiConditionDifferentialBoard from './MultiConditionDifferentialBoard';
import DiagnosisHero from './DiagnosisHero';
import ReportDownloadMenu from './ReportDownloadMenu';
import DrGradeProbabilities from './DrGradeProbabilities';

export default function ResultView({ result, onReset }) {
  // Interactive lesion selection states
  const [selectedLesion, setSelectedLesion] = useState(null);
  const [selectedCoordinate, setSelectedCoordinate] = useState(null);

  // Class visibility toggles (default all visible)
  const [visibleClasses, setVisibleClasses] = useState({
    hemorrhage: true,
    hard_exudate: true,
    microaneurysm: true,
    cotton_wool_spot: true,
  });

  const [showLabels, setShowLabels] = useState(true);
  const [showMaskFill, setShowMaskFill] = useState(true);

  const handleToggleClass = (classKey) => {
    setVisibleClasses((prev) => ({
      ...prev,
      [classKey]: !prev[classKey],
    }));
  };

  const handleToggleAllClasses = (visible) => {
    setVisibleClasses({
      hemorrhage: visible,
      hard_exudate: visible,
      microaneurysm: visible,
      cotton_wool_spot: visible,
    });
  };

  const handleSelectLesion = (lesion) => {
    setSelectedLesion(lesion);
    setSelectedCoordinate(null);
  };

  const handleSelectCoordinate = (coords) => {
    setSelectedCoordinate(coords);
    setSelectedLesion(null);
  };

  const handleClearSelection = () => {
    setSelectedLesion(null);
    setSelectedCoordinate(null);
  };

  if (!result) return null;

  const isRejected = result.status === 'rejected_for_quality';
  const grade = result.grade ?? result.retinal_grade ?? 0;

  // Grade color tokens
  const gradeConfig = {
    0: {
      color: "#10b981",
      bgColor: "bg-emerald-950/40",
      textColor: "text-emerald-300",
      borderColor: "border-emerald-500/30",
      badgeColor: "bg-emerald-600",
      title: "No Apparent Diabetic Retinopathy",
      action: "Routine annual rescreening recommended.",
      findings: "No microaneurysms, hemorrhages, or exudates detected.",
    },
    1: {
      color: "#3b82f6",
      bgColor: "bg-blue-950/40",
      textColor: "text-blue-300",
      borderColor: "border-blue-500/30",
      badgeColor: "bg-blue-600",
      title: "Mild Non-Proliferative DR (NPDR)",
      action: "Follow-up ophthalmic rescreening in 6-12 months. Optimize glycemic & BP control.",
      findings: "Focal microaneurysms only.",
    },
    2: {
      color: "#f59e0b",
      bgColor: "bg-amber-950/40",
      textColor: "text-amber-300",
      borderColor: "border-amber-500/30",
      badgeColor: "bg-amber-600",
      title: "Moderate Non-Proliferative DR (NPDR)",
      action: "Refer to ophthalmologist within 3-6 months for detailed retinal evaluation.",
      findings: "Microaneurysms, intraretinal blot hemorrhages, and hard exudates.",
    },
    3: {
      color: "#f97316",
      bgColor: "bg-orange-950/40",
      textColor: "text-orange-300",
      borderColor: "border-orange-500/30",
      badgeColor: "bg-orange-600",
      title: "Severe Non-Proliferative DR (NPDR)",
      action: "Prompt specialist referral within 2-4 weeks. High risk of progression.",
      findings: "Any 4-2-1 criteria (>20 hemorrhages in 4 quadrants, venous beading, IRMA).",
    },
    4: {
      color: "#ef4444",
      bgColor: "bg-rose-950/40",
      textColor: "text-rose-300",
      borderColor: "border-rose-500/30",
      badgeColor: "bg-rose-600",
      title: "Proliferative Diabetic Retinopathy (PDR)",
      action: "Urgent ophthalmology referral within 24-48 hours. Risk of acute vision loss.",
      findings: "Neovascularization of the disc/retina, vitreous/preretinal hemorrhage.",
    },
  };

  const currentGrade = gradeConfig[grade] || gradeConfig[0];

  const lesionsList = result.lesions || [];
  // When the trained classifier produced the findings, the legacy ICDR grading and pixel-area cards are hidden.
  const fromModel = (result.disease_findings || []).some((f) => f.model_name);

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 space-y-6">
      {/* Report Header & Actions */}
      <div className="flex flex-col gap-4 rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-xl sm:flex-row sm:items-center sm:justify-between">
        <div className="flex items-center gap-4">
          <button
            onClick={onReset}
            className="rounded-lg border border-slate-700 p-2 text-slate-300 transition-colors hover:bg-slate-800 hover:text-white"
            title="Back to screening"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>
          <div>
            <div className="flex flex-wrap items-center gap-2.5">
              <h1 className="font-heading text-xl font-bold text-white sm:text-2xl">Retinal Screening Report</h1>
              <span className="rounded-md bg-slate-800 px-2 py-0.5 font-mono text-xs text-slate-400">
                #{result.id.slice(0, 8)}
              </span>
            </div>
            <dl className="mt-1.5 flex flex-wrap gap-x-5 gap-y-1 text-xs text-slate-400">
              <div><dt className="inline">Patient </dt><dd className="inline font-mono font-semibold text-slate-200">{result.patient_id || "Anonymous"}</dd></div>
              <div><dt className="inline">Eye </dt><dd className="inline font-mono font-semibold text-slate-200">{result.eye_side || "Unknown"}</dd></div>
              <div><dt className="inline">Age </dt><dd className="inline font-mono font-semibold text-slate-200">{result.patient_age ? `${result.patient_age} yrs` : "N/A"}</dd></div>
              {result.created_at && (
                <div><dt className="inline">Date </dt><dd className="inline font-mono font-semibold text-slate-200">{new Date(result.created_at).toLocaleString([], { dateStyle: 'medium', timeStyle: 'short' })}</dd></div>
              )}
            </dl>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={onReset}
            className="rounded-lg border border-slate-700 px-4 py-2 text-sm font-semibold text-slate-200 transition-colors hover:bg-slate-800 hover:text-white"
          >
            New Screening
          </button>

          {!isRejected && <ReportDownloadMenu reportUrl={result.report_download_url} />}
        </div>
      </div>

      {/* IF REJECTED FOR QUALITY: Prominent Clinical Triage Advice */}
      {isRejected ? (
        <div className="bg-rose-950/40 border-2 border-rose-500/40 rounded-2xl p-6 sm:p-8 space-y-4 shadow-2xl">
          <div className="flex items-start space-x-4">
            <div className="w-12 h-12 rounded-2xl bg-rose-500/20 flex items-center justify-center text-rose-400 flex-shrink-0 border border-rose-500/30">
              <ShieldAlert className="w-7 h-7" />
            </div>
            <div>
              <div className="inline-block px-3 py-1 bg-rose-600 text-white font-extrabold text-xs uppercase tracking-wider rounded-md mb-1">
                Quality Gate: Image Ungradable
              </div>
              <h2 className="text-xl font-bold text-white">
                Retake Image: Capture Does Not Meet Diagnostic Criteria
              </h2>
              <p className="text-sm text-rose-300 mt-1">
                {result.referral_recommendation || "Image quality is insufficient for reliable clinical grading."}
              </p>
            </div>
          </div>

          <div className="bg-slate-900/80 border border-rose-500/20 rounded-2xl p-4 space-y-2">
            <h4 className="text-xs font-bold uppercase tracking-wider text-rose-400">
              Specific Deficiencies Detected:
            </h4>
            <ul className="list-disc list-inside text-xs text-slate-300 space-y-1 font-medium">
              {result.quality_reasons?.map((reason, idx) => (
                <li key={idx}>{reason}</li>
              ))}
            </ul>
          </div>

          <div className="flex items-center space-x-4 text-xs text-rose-300 font-medium">
            <span>Quality Score: <b>{(result.quality_score * 100).toFixed(1)}%</b> (Threshold: 50%)</span>
          </div>

          {result.original_image_url && (
            <div className="mt-4 pt-4 border-t border-rose-500/20 flex flex-col items-center">
              <p className="text-xs font-semibold text-rose-300 mb-2">Original Rejected Fundus Scan:</p>
              <img
                src={result.original_image_url}
                alt="Rejected Scan"
                className="w-64 h-64 object-cover rounded-2xl border border-rose-500/30 shadow-md"
              />
            </div>
          )}
        </div>
      ) : (
        /* IF ACCEPTED: Full Interactive Retina & Lesion Analysis Dashboard */
        <div className="space-y-6">
          {/* Primary Diagnosis */}
          <DiagnosisHero result={result} />

          {/* Key Metrics Row */}
          <AnalysisSummary
            result={result}
            selectedLesion={selectedLesion}
          />

          {/* Main Visualizer & Inspection Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            {/* Left / Center: Interactive Retina Viewer (8 cols) */}
            <div className="lg:col-span-8 space-y-5">
              <RetinaViewer
                originalImageUrl={result.original_image_url}
                heatmapUrl={result.heatmap_url}
                lesions={lesionsList}
                selectedLesion={selectedLesion}
                onSelectLesion={handleSelectLesion}
                selectedCoordinate={selectedCoordinate}
                onSelectCoordinate={handleSelectCoordinate}
                onClearSelection={handleClearSelection}
                visibleClasses={visibleClasses}
                showLabels={showLabels}
                showMaskFill={showMaskFill}
              />

              {/* Lesion Classes Legend & Layer Toggles */}
              <LesionLegend
                classSummaries={result.class_summaries}
                visibleClasses={visibleClasses}
                onToggleClass={handleToggleClass}
                onToggleAll={handleToggleAllClasses}
                showLabels={showLabels}
                onToggleLabels={setShowLabels}
                showMaskFill={showMaskFill}
                onToggleMaskFill={setShowMaskFill}
              />
            </div>

            {/* Right: Interactive Lesion Details Panel & Clinical Triage (4 cols) */}
            <div className="lg:col-span-4 space-y-5">
              {/* Click-to-Inspect Lesion Details Card */}
              <LesionDetailsPanel
                selectedLesion={selectedLesion}
                selectedCoordinate={selectedCoordinate}
                onSelectLesion={handleSelectLesion}
                allLesions={lesionsList}
                onClearSelection={handleClearSelection}
              />

              {!fromModel && (
                <>
              {/* ICDR Severity Card */}
              <div className={`rounded-2xl border ${currentGrade.borderColor} ${currentGrade.bgColor} p-6 shadow-xl space-y-4`}>
                <div className="flex items-center justify-between">
                  <span className={`px-2.5 py-1 rounded-md text-white text-xs font-semibold ${currentGrade.badgeColor}`}>
                    DR Grade {grade} · ICDR
                  </span>
                  <span className="text-xs font-mono font-semibold text-slate-300 tabular-nums">
                    Conf: <b className="text-white font-bold">{(result.confidence * 100).toFixed(1)}%</b>
                  </span>
                </div>

                <div>
                  <h2 className={`text-lg sm:text-xl font-heading font-black tracking-tight ${currentGrade.textColor}`}>
                    {currentGrade.title}
                  </h2>
                  <p className="text-xs text-slate-300 mt-1 font-medium leading-relaxed font-sans">
                    {currentGrade.findings}
                  </p>
                </div>

                {/* Referable Banner */}
                <div className={`p-3 rounded-2xl border flex items-center space-x-2.5 ${
                  result.is_referable
                    ? "bg-rose-950/60 border-rose-500/40 text-rose-300"
                    : "bg-emerald-950/60 border-emerald-500/40 text-emerald-300"
                }`}>
                  <CheckCircle2 className="w-5 h-5 flex-shrink-0" />
                  <span className="text-xs font-heading font-bold uppercase tracking-wider">
                    {result.is_referable ? "Referable DR (Grade >= 2)" : "Non-Referable DR (Grade < 2)"}
                  </span>
                </div>

                {/* Action Window */}
                <div className="bg-slate-950/70 p-3.5 rounded-2xl border border-slate-800 text-xs">
                  <span className="font-heading font-bold text-white block mb-0.5">Triage Recommendation:</span>
                  <p className="text-slate-300 leading-relaxed font-sans">{currentGrade.action}</p>
                </div>

                {/* Uncertainty Flag Banner */}
                {result.uncertain && (
                  <div className="p-3 bg-amber-950/50 border border-amber-500/40 rounded-2xl text-xs text-amber-300 flex items-start space-x-2 font-sans">
                    <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <span className="font-heading font-bold">Uncertainty Flag:</span>
                      <p className="mt-0.5">{result.uncertainty_note || "Confidence below 60%. Specialist clinical review recommended."}</p>
                    </div>
                  </div>
                )}
              </div>

              {/* DR grade probabilities */}
              <DrGradeProbabilities probabilities={result.probabilities} reportedGrade={result.grade} />

              {/* Affected Area Calculation Details Card */}
              <div className="bg-slate-900/90 rounded-2xl border border-slate-800 shadow-xl p-5 space-y-2.5 text-xs">
                <div className="flex items-center justify-between">
                  <span className="font-heading font-bold uppercase tracking-wider text-slate-300">Quantification Method</span>
                  <span className="font-mono text-sky-400 font-bold">Pixel-Level Union</span>
                </div>
                <div className="p-3 bg-slate-950/70 rounded-2xl border border-slate-800/80 space-y-1 font-mono text-[11px] text-slate-300">
                  <div className="text-slate-400 text-[10px]">Mathematical Formula:</div>
                  <div className="text-sky-300 font-bold">
                    Area (%) = (Lesion Union Pixels / Valid Retina Pixels) × 100
                  </div>
                  <div className="pt-1 text-[10px] text-slate-400">
                    Resolution: {result.image_resolution ? `${result.image_resolution[0]} × ${result.image_resolution[1]} px` : 'Calibrated'}
                  </div>
                </div>
                <p className="text-[11px] text-slate-400 leading-relaxed">
                  The valid illuminated retinal field serves as the denominator, excluding camera sensor borders. Overlapping lesions are unified to strictly avoid double-counting.
                </p>
              </div>
                </>
              )}
            </div>
          </div>

          {/* All Screened Conditions */}
          <MultiConditionDifferentialBoard
            diseaseFindings={result.disease_findings || []}
            cupToDiscRatio={result.cup_to_disc_ratio}
          />
        </div>
      )}

    </div>
  );
}
