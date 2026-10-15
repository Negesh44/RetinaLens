import React, { useState, useRef } from 'react';
import { UploadCloud, Sparkles, AlertCircle, CheckCircle2, ArrowRight, RefreshCw } from 'lucide-react';
import ThreeEyeScanner from './ThreeEyeScanner';
import AnalysisLoadingState from './AnalysisLoadingState';
import { RFMID_SAMPLES, QUALITY_SAMPLES, sampleUrl } from '../lib/samples';
import { getDiseaseMeta } from '../lib/diseases';

export default function UploadView({ onScreeningComplete }) {
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [patientId, setPatientId] = useState('');
  const [patientAge, setPatientAge] = useState('');
  const [eyeSide, setEyeSide] = useState('OD');
  const [isLoading, setIsLoading] = useState(false);
  const [stepMessage, setStepMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState(null);

  const fileInputRef = useRef(null);
  const [demoLoading, setDemoLoading] = useState(null);
  const [selectedDemo, setSelectedDemo] = useState(null);
  const submitRef = useRef(null);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0]);
    }
  };

  const handleFileSelection = (file) => {
    setErrorMessage(null);
    setSelectedDemo(null);
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!selectedFile) {
      setErrorMessage("Please select or drop a retinal fundus image first.");
      return;
    }
    runScreening(selectedFile, { patientId, patientAge, eyeSide });
  };

  // One click loads a real demo photo and its details; screening starts with "Begin AI Screening".
  const loadDemo = async (sample) => {
    if (isLoading) return;
    setErrorMessage(null);
    setDemoLoading(sample.file);
    setPatientId(`DEMO-${sample.label.toUpperCase().replace(/[^A-Z0-9]+/g, '-')}`);
    setPatientAge('');
    setEyeSide(sample.eye);
    try {
      const response = await fetch(sampleUrl(sample));
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const blob = await response.blob();
      const file = new File([blob], sample.file, { type: blob.type || 'image/jpeg' });
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(blob));
      setSelectedDemo(sample.file);
      submitRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    } catch {
      setErrorMessage('Could not load the demo image.');
    } finally {
      setDemoLoading(null);
    }
  };

  const runScreening = async (file, details) => {
    setIsLoading(true);
    setErrorMessage(null);

    // Progress messages while the request runs
    setStepMessage("Checking photo quality...");
    const timer1 = setTimeout(() => {
      setStepMessage("Finding the retina in the photo...");
    }, 400);
    const timer2 = setTimeout(() => {
      setStepMessage("Screening for 10 eye conditions...");
    }, 900);
    const timer3 = setTimeout(() => {
      setStepMessage("Mapping where the AI looked...");
    }, 1600);
    const timer4 = setTimeout(() => {
      setStepMessage("Preparing the report...");
    }, 2300);

    const formData = new FormData();
    formData.append("file", file);
    if (details.patientId.trim()) formData.append("patient_id", details.patientId.trim());
    if (details.patientAge.trim()) formData.append("patient_age", details.patientAge.trim());
    formData.append("eye_side", details.eyeSide);

    try {
      const res = await fetch("/api/screen", {
        method: "POST",
        body: formData,
      });

      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      clearTimeout(timer4);

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({ detail: "Screening request failed." }));
        throw new Error(errorData.detail || `Server error (${res.status})`);
      }

      const resultData = await res.json();
      setIsLoading(false);
      onScreeningComplete(resultData);
    } catch (err) {
      clearTimeout(timer1);
      clearTimeout(timer2);
      clearTimeout(timer3);
      clearTimeout(timer4);
      setIsLoading(false);
      setErrorMessage(err.message || "Failed to process retinal scan.");
    }
  };

  return (
    <div className="max-w-5xl mx-auto px-4 py-8 space-y-6">
      {/* Hero Section */}
      <div className="bg-slate-900/90 text-white rounded-3xl border border-slate-800 shadow-2xl p-6 sm:p-8 backdrop-blur-xl relative overflow-hidden">
        {/* Subtle glow accent */}
        <div className="absolute -top-24 -right-24 w-80 h-80 bg-sky-500/15 rounded-full blur-3xl pointer-events-none"></div>

        <div className="relative z-10 flex flex-col gap-5 border-b border-slate-800/80 pb-5 md:flex-row md:items-center md:justify-between">
          <div className="max-w-lg space-y-1.5">
            <span className="text-[11px] font-semibold uppercase tracking-wider text-sky-400">AI eye screening</span>
            <h1 className="font-heading text-xl font-extrabold leading-tight tracking-tight text-white sm:text-2xl">Retinal Disease Screening</h1>
            <p className="text-sm leading-relaxed text-slate-400">
              Upload a retina photo to check it for 10 common eye diseases and see how soon to visit an eye doctor.
            </p>
          </div>

          {/* Interactive 3D Bio-Digital Eye Scanner */}
          <div
            className="theme-purple-island relative flex flex-shrink-0 flex-col items-center justify-center overflow-hidden rounded-2xl border border-slate-800 px-3 pb-3 pt-1"
            style={{ background: 'radial-gradient(circle at 50% 42%, rgba(139,92,246,0.32), rgba(31,20,48,0.96) 68%)' }}
          >
            <ThreeEyeScanner isScanning={isLoading} size={140} />
            <div className="flex items-center gap-2 rounded-full border border-sky-500/20 bg-slate-950/70 px-2.5 py-0.5 font-mono text-[9px] tracking-[0.18em] text-sky-300">
              <span className="relative flex h-1.5 w-1.5">
                <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-sky-400 opacity-60" />
                <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-sky-400" />
              </span>
              {isLoading ? 'SCANNING' : 'READY'}
            </div>
          </div>
        </div>


        {/* One-click demo */}
        <div className="relative z-10 mt-6">
          <div className="mb-2.5 flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-sky-400" />
            <h2 className="text-sm font-semibold text-white">Demo photos</h2>
            <span className="text-xs text-slate-400">Real test photos the AI never trained on. Pick one, then press Begin AI Screening.</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {[...RFMID_SAMPLES, ...QUALITY_SAMPLES].map((sample) => {
              const loading = demoLoading === sample.file;
              const selected = selectedDemo === sample.file;
              const color = sample.expected ? getDiseaseMeta(sample.expected).color : '#94a3b8';
              return (
                <button
                  key={sample.file}
                  type="button"
                  disabled={isLoading}
                  onClick={() => loadDemo(sample)}
                  title={`${sample.desc}
Photo: ${sample.credit}`}
                  className={`inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium transition-colors disabled:cursor-not-allowed ${
                    selected || loading
                      ? 'border-sky-400 bg-sky-500/15 text-white'
                      : 'border-slate-700 bg-slate-950/70 text-slate-200 hover:border-slate-500 hover:bg-slate-800 disabled:opacity-50'
                  }`}
                >
                  {loading ? (
                    <RefreshCw className="h-3.5 w-3.5 animate-spin text-sky-300" />
                  ) : selected ? (
                    <CheckCircle2 className="h-3.5 w-3.5 text-sky-300" />
                  ) : (
                    <span className="h-2 w-2 rounded-full" style={{ backgroundColor: color }} />
                  )}
                  {sample.expected ? sample.label : `Bad photo: ${sample.label.toLowerCase()}`}
                </button>
              );
            })}
          </div>
        </div>

        {/* Upload Form */}
        <form onSubmit={handleSubmit} className="mt-8 space-y-6">
          {/* Patient Metadata Card */}
          <div className="bg-slate-950/60 border border-slate-800/90 rounded-2xl p-4 sm:p-5 shadow-inner">
            <h3 className="text-xs font-heading font-bold uppercase tracking-wider text-slate-300 mb-3.5">
              Patient details (optional)
            </h3>
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5 font-sans">Patient ID</label>
                <input
                  type="text"
                  placeholder="e.g. PAT-98214"
                  value={patientId}
                  onChange={(e) => setPatientId(e.target.value)}
                  className="w-full text-xs font-mono px-3.5 py-2.5 rounded-xl border border-slate-700/80 bg-slate-900/90 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500/80 focus:border-sky-500 transition-all"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5 font-sans">Age (years)</label>
                <input
                  type="number"
                  min="0"
                  max="125"
                  placeholder="e.g. 58"
                  value={patientAge}
                  onChange={(e) => setPatientAge(e.target.value)}
                  className="w-full text-xs font-mono px-3.5 py-2.5 rounded-xl border border-slate-700/80 bg-slate-900/90 text-white placeholder-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500/80 focus:border-sky-500 transition-all"
                />
              </div>

              <div>
                <label className="block text-xs font-semibold text-slate-300 mb-1.5 font-sans">Eye</label>
                <select
                  value={eyeSide}
                  onChange={(e) => setEyeSide(e.target.value)}
                  className="w-full text-xs font-sans font-medium px-3.5 py-2.5 rounded-xl border border-slate-700/80 bg-slate-900/90 text-white focus:outline-none focus:ring-2 focus:ring-sky-500/80 focus:border-sky-500 transition-all cursor-pointer"
                >
                  <option value="OD">Right eye (OD)</option>
                  <option value="OS">Left eye (OS)</option>
                  <option value="Unknown">Not specified</option>
                </select>
              </div>
            </div>
          </div>

          {/* Drag & Drop Area */}
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all duration-200 relative ${
              dragActive
                ? "border-sky-400 bg-sky-950/40 ring-4 ring-sky-500/20"
                : previewUrl
                ? "border-slate-700/90 bg-slate-950/60 hover:border-sky-400/80"
                : "border-slate-700/80 hover:border-sky-400/80 bg-slate-950/50 hover:bg-slate-900/60"
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileInput}
              className="hidden"
            />

            {previewUrl ? (
              <div className="flex flex-col items-center">
                <div className="relative group">
                  <img
                    src={previewUrl}
                    alt="Fundus Preview"
                    className="w-52 h-52 object-cover rounded-2xl shadow-xl border-2 border-slate-700 ring-2 ring-sky-500/30"
                  />
                  <div className="absolute inset-0 bg-slate-950/60 opacity-0 group-hover:opacity-100 transition-opacity rounded-2xl flex items-center justify-center text-white text-xs font-bold backdrop-blur-xs">
                    Click to change image
                  </div>
                </div>
                <div className="mt-3.5 flex items-center space-x-2 text-xs text-slate-300 font-medium font-sans">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                  <span className="font-semibold text-white">{selectedFile?.name || "Selected Image"}</span>
                  <span className="text-slate-400 font-mono text-[11px]">({(selectedFile?.size / 1024).toFixed(1)} KB)</span>
                </div>
              </div>
            ) : (
              <div className="flex flex-col items-center py-4">
                <div className="w-14 h-14 rounded-2xl bg-sky-500/20 text-sky-400 flex items-center justify-center mb-3 shadow-inner border border-sky-500/30">
                  <UploadCloud className="w-7 h-7" />
                </div>
                <h3 className="text-base font-heading font-bold text-white tracking-tight">
                  Drop a retina photo here
                </h3>
                <p className="text-xs text-slate-400 mt-1 font-sans">
                  or click to choose a file · JPEG, PNG, TIFF, BMP or WebP, up to 25 MB
                </p>
                <button
                  type="button"
                  className="mt-4 px-4 py-2 bg-slate-850 hover:bg-slate-800 border border-slate-700 rounded-xl text-xs font-semibold text-slate-200 hover:text-white shadow-sm transition-all"
                >
                  Choose file
                </button>
              </div>
            )}
          </div>

          {/* Error Message Alert */}
          {errorMessage && (
            <div className="p-4 rounded-xl bg-rose-950/50 border border-rose-500/50 text-rose-200 text-sm flex items-start space-x-3 backdrop-blur-md">
              <AlertCircle className="w-5 h-5 flex-shrink-0 mt-0.5 text-rose-400" />
              <div>
                <p className="font-bold text-rose-100">Could not analyse this photo</p>
                <p className="text-xs mt-0.5 text-rose-300">{errorMessage}</p>
              </div>
            </div>
          )}

          {/* Submit / Screening Action */}
          <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-2">
            <div className="text-xs text-slate-400 flex items-center space-x-2 font-sans">
              <Sparkles className="w-4 h-4 text-sky-400" />
              <span>Takes a few seconds. The report can be downloaded as a PDF.</span>
            </div>

            <button
              ref={submitRef}
              type="submit"
              disabled={isLoading || !selectedFile}
              className={`w-full sm:w-auto px-8 py-3.5 rounded-xl font-heading font-bold text-sm shadow-lg flex items-center justify-center space-x-2.5 transition-all ${
                isLoading || !selectedFile
                  ? "bg-slate-800 text-slate-500 cursor-not-allowed border border-slate-700/60 shadow-none"
                  : "bg-gradient-to-r from-sky-500 to-blue-600 hover:from-sky-400 hover:to-blue-500 text-white shadow-sky-500/25 hover:shadow-sky-500/40 active:scale-[0.98]"
              }`}
            >
              {isLoading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin text-white" />
                  <span>Analysing photo...</span>
                </>
              ) : (
                <>
                  <span>Begin AI Screening</span>
                  <ArrowRight className="w-4 h-4" />
                </>
              )}
            </button>
          </div>

          {/* Step Progress Clinical Loading Overlay */}
          {isLoading && (
            <div className="pt-4 border-t border-slate-800">
              <AnalysisLoadingState currentStepMessage={stepMessage} />
            </div>
          )}
        </form>
      </div>
    </div>
  );
}
