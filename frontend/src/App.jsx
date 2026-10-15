import React, { useState } from 'react';
import Navbar from './components/Navbar';
import UploadView from './components/UploadView';
import ResultView from './components/ResultView';
import DashboardView from './components/DashboardView';
import GuideView from './components/GuideView';
import ChatAssistant from './components/ChatAssistant';
import ThreeNeuralBackground from './components/ThreeNeuralBackground';

export default function App() {
  const [activeTab, setActiveTab] = useState('upload'); // 'upload', 'result', 'dashboard', 'guide'
  const [currentResult, setCurrentResult] = useState(null);

  const handleScreeningComplete = (resultData) => {
    setCurrentResult(resultData);
    setActiveTab('result');
  };

  const handleSelectScreeningFromDashboard = (screeningData) => {
    setCurrentResult(screeningData);
    setActiveTab('result');
  };

  const handleResetToUpload = () => {
    setCurrentResult(null);
    setActiveTab('upload');
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans relative selection:bg-sky-500 selection:text-white">
      {/* 3D Neural Particle Ambient Background */}
      <ThreeNeuralBackground />
      {/* Navigation Header */}
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} />

      {/* Main Workspace Area */}
      <main className="flex-1 pb-16">
        {activeTab === 'upload' && (
          <UploadView onScreeningComplete={handleScreeningComplete} />
        )}

        {activeTab === 'result' && (
          <ResultView result={currentResult} onReset={handleResetToUpload} />
        )}

        {activeTab === 'dashboard' && (
          <DashboardView onSelectScreening={handleSelectScreeningFromDashboard} />
        )}

        {activeTab === 'guide' && (
          <GuideView onStart={() => setActiveTab('upload')} />
        )}
      </main>

      {/* Floating report assistant (Google Gemini) */}
      <ChatAssistant result={activeTab === 'result' ? currentResult : null} />

      {/* Persistent Medical Footer */}
      <footer className="bg-slate-900 border-t border-slate-800 text-slate-400 py-8 px-4 text-xs mt-auto">
        <div className="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center space-x-2">
            <div className="w-6 h-6 rounded bg-sky-500/20 text-sky-400 flex items-center justify-center font-bold">
              RL
            </div>
            <span className="font-bold text-white tracking-wide">RetinaLens Clinical AI Platform</span>
            <span className="text-slate-600">|</span>
            <span className="text-slate-500">v1.0.0 (Research Edition)</span>
          </div>

          <div className="text-center sm:text-right text-slate-400 max-w-xl text-[11px] leading-relaxed">
            <b>Regulatory Notice:</b> Investigational screening aid for clinician review, <u>NOT</u> a medical device or definitive diagnosis.
            All findings must be corroborated by a licensed ophthalmic clinician.
          </div>
        </div>
      </footer>
    </div>
  );
}
