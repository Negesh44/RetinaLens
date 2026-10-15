import React, { useEffect, useRef, useState } from 'react';
import { ChevronDown, Download, FileText, Languages } from 'lucide-react';

// Used until /api/report-languages answers (and if it cannot be reached).
const FALLBACK_LANGUAGES = [{ code: 'en', name: 'English', native: 'English' }];

export default function ReportDownloadMenu({ reportUrl }) {
  const [open, setOpen] = useState(false);
  const [languages, setLanguages] = useState(FALLBACK_LANGUAGES);
  const rootRef = useRef(null);

  useEffect(() => {
    let cancelled = false;
    fetch('/api/report-languages')
      .then((res) => (res.ok ? res.json() : null))
      .then((list) => {
        if (!cancelled && Array.isArray(list) && list.length) setLanguages(list);
      })
      .catch(() => {});
    return () => {
      cancelled = true;
    };
  }, []);

  // Close on outside click or Escape.
  useEffect(() => {
    if (!open) return undefined;
    const onClick = (e) => {
      if (!rootRef.current?.contains(e.target)) setOpen(false);
    };
    const onKey = (e) => {
      if (e.key === 'Escape') setOpen(false);
    };
    document.addEventListener('mousedown', onClick);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onClick);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  if (!reportUrl) return null;

  return (
    <div ref={rootRef} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="menu"
        aria-expanded={open}
        className="flex items-center gap-2 rounded-lg bg-sky-600 px-4 py-2 text-sm font-semibold text-white shadow-sm transition-colors hover:bg-sky-500"
      >
        <FileText className="h-4 w-4" />
        <span>Download PDF Report</span>
        <ChevronDown className={`h-4 w-4 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>

      {open && (
        <div
          role="menu"
          className="absolute right-0 z-40 mt-2 w-64 overflow-hidden rounded-xl border border-slate-800 bg-slate-900 shadow-2xl"
        >
          <div className="flex items-center gap-2 border-b border-slate-800 px-4 py-2.5 text-xs font-semibold text-slate-400">
            <Languages className="h-4 w-4 text-sky-400" />
            Choose the report language
          </div>
          <ul className="max-h-80 overflow-y-auto py-1">
            {languages.map((lang) => (
              <li key={lang.code}>
                <a
                  role="menuitem"
                  href={`${reportUrl}?lang=${lang.code}`}
                  download
                  onClick={() => setOpen(false)}
                  className="flex items-center justify-between gap-3 px-4 py-2.5 text-sm text-slate-200 transition-colors hover:bg-slate-800"
                >
                  <span>
                    <span className="font-semibold text-white">{lang.native}</span>
                    {lang.native !== lang.name && <span className="ml-2 text-xs text-slate-400">{lang.name}</span>}
                  </span>
                  <Download className="h-4 w-4 flex-shrink-0 text-slate-400" />
                </a>
              </li>
            ))}
          </ul>
          <p className="border-t border-slate-800 px-4 py-2 text-[11px] leading-snug text-slate-500">
            The patient summary is in the chosen language; the section for the eye doctor stays in English.
          </p>
        </div>
      )}
    </div>
  );
}
