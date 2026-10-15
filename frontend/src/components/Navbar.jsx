import React, { useState } from 'react';
import { Eye, Activity, BarChart2, BookOpen, Moon, Sun } from 'lucide-react';

// Light is the default; the choice is saved and applied before first paint by index.html.
function useTheme() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains('dark'));
  const toggle = () => {
    const next = !dark;
    document.documentElement.classList.toggle('dark', next);
    try {
      localStorage.setItem('retinalens-theme', next ? 'dark' : 'light');
    } catch {
      // Storage can be unavailable (private mode); the theme still applies for this visit.
    }
    setDark(next);
  };
  return [dark, toggle];
}

const TABS = [
  { id: 'upload', label: 'Screening', icon: Activity },
  { id: 'dashboard', label: 'Patient Records', icon: BarChart2 },
  { id: 'guide', label: 'How It Works', icon: BookOpen },
];

export default function Navbar({ activeTab, setActiveTab }) {
  const [dark, toggleTheme] = useTheme();
  return (
    <header className="theme-purple-island sticky top-0 z-50 border-b border-slate-800 bg-slate-950/95 text-white shadow-lg backdrop-blur">
      {/* Full width: title pinned to the left edge, navigation and theme toggle to the right edge. */}
      <div className="flex h-16 w-full items-center justify-between gap-4 px-4 sm:px-6 lg:px-8">
        <button type="button" className="flex flex-shrink-0 items-center gap-3 text-left" onClick={() => setActiveTab('upload')}>
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-sky-600">
            <Eye className="h-5 w-5 text-white" />
          </div>
          <div>
            <span className="block font-heading text-lg font-bold leading-tight tracking-tight text-white">
              Retina<span className="text-sky-400">Lens</span>
            </span>
            <span className="hidden text-[11px] text-slate-400 sm:block">Retinal Disease Screening</span>
          </div>
        </button>

        <div className="ml-auto flex items-center gap-2">
          <nav className="flex gap-1">
            {TABS.map(({ id, label, icon: Icon }) => {
              // The result page is part of the screening flow.
              const active = activeTab === id || (id === 'upload' && activeTab === 'result');
              return (
                <button
                  key={id}
                  type="button"
                  onClick={() => setActiveTab(id)}
                  aria-current={active ? 'page' : undefined}
                  title={label}
                  className={`flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-colors ${
                    active ? 'bg-slate-800 text-white' : 'text-slate-400 hover:bg-slate-900 hover:text-white'
                  }`}
                >
                  <Icon className={`h-4 w-4 ${active ? 'text-sky-400' : ''}`} />
                  <span className="hidden sm:inline">{label}</span>
                </button>
              );
            })}
          </nav>
          <button
            type="button"
            onClick={toggleTheme}
            title={dark ? 'Switch to light theme' : 'Switch to dark theme'}
            aria-label={dark ? 'Switch to light theme' : 'Switch to dark theme'}
            className="rounded-lg border border-slate-800 p-2 text-slate-400 transition-colors hover:bg-slate-900 hover:text-white"
          >
            {dark ? <Sun className="h-4 w-4" /> : <Moon className="h-4 w-4" />}
          </button>
        </div>
      </div>
    </header>
  );
}
