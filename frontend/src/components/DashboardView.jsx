import React, { useEffect, useMemo, useState } from 'react';
import {
  AlertTriangle,
  Camera,
  Clock,
  Download,
  Eye,
  FileText,
  HeartPulse,
  RefreshCw,
  Search,
  ShieldCheck,
  Users,
} from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { getDiseaseMeta, getUrgency } from '../lib/diseases';

const PERIODS = [
  { id: '7', label: '7 days' },
  { id: '30', label: '30 days' },
  { id: '90', label: '90 days' },
  { id: 'all', label: 'All time' },
];

// Outcome colours carry meaning (status): healthy, needs attention, photo unusable, older engine.
const OUTCOMES = {
  normal: { label: 'Normal', color: '#10b981' },
  abnormal: { label: 'Abnormal', color: '#f59e0b' },
  rejected: { label: 'Photo rejected', color: '#94a3b8' },
  legacy: { label: 'Older engine', color: '#cbd5e1' },
};

const URGENCY_LABELS = { emergency: 'Emergency', urgent: 'Urgent', prompt: 'Within 2–4 weeks', routine: 'Routine' };
const URGENCY_COLORS = { emergency: '#dc2626', urgent: '#e11d48', prompt: '#d97706', routine: '#10b981' };

const pct = (v) => (v == null ? '–' : `${Math.round(v)}%`);
const fmtDate = (iso) => new Date(iso).toLocaleDateString([], { day: '2-digit', month: 'short' });
const fmtDateTime = (iso) =>
  new Date(iso).toLocaleString([], { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' });
const daysAgo = (n) => (n === 0 ? 'today' : n === 1 ? 'yesterday' : `${n} days ago`);

function urgencyChip(tier) {
  if (tier === 'retake') return { label: 'Retake photo', className: 'bg-slate-700/40 text-slate-300 ring-slate-600/50' };
  if (!tier) return null;
  const u = getUrgency({ emergency: 'Emergency', urgent: 'Urgent', prompt: 'Prompt', routine: 'Routine' }[tier]);
  return { label: u.label, className: u.className };
}

function Card({ title, subtitle, action, children, className = '' }) {
  return (
    <section className={`rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-lg ${className}`}>
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <h3 className="font-heading text-sm font-bold text-white">{title}</h3>
          {subtitle && <p className="mt-0.5 text-xs text-slate-400">{subtitle}</p>}
        </div>
        {action}
      </div>
      {children}
    </section>
  );
}

function Kpi({ icon: Icon, label, value, note, tone = 'text-white' }) {
  return (
    <div className="rounded-xl border border-slate-800 bg-slate-900 p-4 shadow-sm">
      <p className="flex items-center gap-1.5 text-[11px] font-semibold uppercase tracking-wider text-slate-400">
        <Icon className="h-3.5 w-3.5 text-sky-400" />
        {label}
      </p>
      <p className={`mt-1.5 font-heading text-2xl font-bold tabular-nums ${tone}`}>{value}</p>
      {note && <p className="mt-0.5 text-xs text-slate-400">{note}</p>}
    </div>
  );
}

// A labelled horizontal bar; single accent colour, the dot carries the condition's identity colour.
function BarRow({ label, dot, value, max, right }) {
  return (
    <div className="grid grid-cols-[minmax(0,10rem)_1fr_4.5rem] items-center gap-3 py-1.5 text-xs">
      <span className="flex min-w-0 items-center gap-1.5 text-slate-200">
        {dot && <span className="h-2 w-2 flex-shrink-0 rounded-full" style={{ backgroundColor: dot }} />}
        <span className="truncate" title={label}>{label}</span>
      </span>
      <span className="h-2 overflow-hidden rounded-full bg-slate-800">
        <span className="block h-full rounded-r bg-sky-500" style={{ width: `${max ? Math.max(2, (value / max) * 100) : 0}%` }} />
      </span>
      <span className="text-right font-mono tabular-nums text-slate-300">{right}</span>
    </div>
  );
}

function toCsv(rows) {
  const head = ['Date', 'Patient ID', 'Age', 'Eye', 'Result', 'Main finding', 'Other findings', 'Urgency', 'AI confidence', 'Photo quality'];
  const esc = (v) => `"${String(v ?? '').replace(/"/g, '""')}"`;
  const lines = rows.map((r) => [
    fmtDateTime(r.date), r.patient_id || 'Anonymous', r.age ?? '', r.eye ?? '', OUTCOMES[r.outcome]?.label,
    r.condition_name ?? '', r.other_conditions, r.urgency ? URGENCY_LABELS[r.urgency] : '',
    r.confidence != null ? `${Math.round(r.confidence * 100)}%` : '', r.quality != null ? r.quality.toFixed(2) : '',
  ].map(esc).join(','));
  return [head.map(esc).join(','), ...lines].join('\r\n');
}

export default function DashboardView({ onSelectScreening }) {
  const [period, setPeriod] = useState('30');
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [opening, setOpening] = useState(null);

  const [search, setSearch] = useState('');
  const [outcome, setOutcome] = useState('all');
  const [condition, setCondition] = useState('all');
  const [worklistTab, setWorklistTab] = useState('doctor');
  const [visible, setVisible] = useState(25);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(period === 'all' ? '/api/analytics' : `/api/analytics?days=${period}`);
      if (!res.ok) throw new Error(`Server error (${res.status})`);
      setData(await res.json());
    } catch (e) {
      setError(e.message || 'Could not load analytics.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [period]);

  const openScreening = async (id) => {
    setOpening(id);
    try {
      const res = await fetch(`/api/results/${id}`);
      if (res.ok) onSelectScreening(await res.json());
    } finally {
      setOpening(null);
    }
  };

  const rows = useMemo(() => {
    const q = search.trim().toLowerCase();
    return (data?.screenings || []).filter((r) =>
      (outcome === 'all' || r.outcome === outcome) &&
      (condition === 'all' || r.condition === condition) &&
      (!q || (r.patient_id || 'anonymous').toLowerCase().includes(q)),
    );
  }, [data, search, outcome, condition]);

  const conditionOptions = useMemo(() => {
    const seen = new Map();
    (data?.screenings || []).forEach((r) => r.condition && r.condition !== 'normal' && seen.set(r.condition, r.condition_name));
    return [...seen.entries()].sort((a, b) => a[1].localeCompare(b[1]));
  }, [data]);

  const exportCsv = () => {
    const blob = new Blob([toCsv(rows)], { type: 'text/csv;charset=utf-8' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `retinalens_records_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  const t = data?.totals;
  const worklist = (data?.worklist || []).filter((w) => (worklistTab === 'doctor' ? w.urgency !== 'retake' : w.urgency === 'retake'));
  const urgencyTotal = data ? Object.values(data.urgency).reduce((a, b) => a + b, 0) : 0;
  const maxCondition = Math.max(1, ...(data?.conditions || []).map((c) => c.count));
  const maxIssue = Math.max(1, ...(data?.quality.issues || []).map((i) => i.count));
  const trend = (data?.trend || []).map((d) => ({ ...d, label: fmtDate(d.date) }));

  return (
    <div className="mx-auto max-w-7xl space-y-6 px-4 py-8">
      {/* Header */}
      <div className="flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="font-heading text-2xl font-extrabold tracking-tight text-white">Patient Records</h1>
          <p className="mt-1 text-sm text-slate-400">Who needs follow-up, what is being found, and how screening is going.</p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <div className="inline-flex rounded-lg border border-slate-800 bg-slate-900 p-1 text-xs font-semibold">
            {PERIODS.map((p) => (
              <button key={p.id} type="button" onClick={() => setPeriod(p.id)}
                className={`rounded-md px-3 py-1.5 transition-colors ${period === p.id ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-white'}`}>
                {p.label}
              </button>
            ))}
          </div>
          <button type="button" onClick={load} title="Refresh"
            className="rounded-lg border border-slate-800 bg-slate-900 p-2 text-slate-300 transition-colors hover:text-white">
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 rounded-xl border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-300">
          <AlertTriangle className="h-4 w-4" /> {error}
        </div>
      )}

      {t && (
        <>
          {/* KPIs */}
          <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
            <Kpi icon={Eye} label="Screenings" value={t.screenings} note={`${t.analysed_by_model} analysed by the AI model`} />
            <Kpi icon={Users} label="Patients" value={t.patients} note={`${t.repeat_patients} screened more than once`} />
            <Kpi icon={HeartPulse} label="Abnormal" value={pct(t.abnormal_pct)} tone="text-amber-300"
              note={`${t.abnormal} of ${t.analysed_by_model} analysed scans`} />
            <Kpi icon={ShieldCheck} label="Normal" value={t.normal} tone="text-emerald-400" note="no disease found" />
            <Kpi icon={Clock} label="Need a doctor" value={t.needs_follow_up} tone={t.overdue ? 'text-rose-400' : 'text-white'}
              note={t.overdue ? `${t.overdue} overdue` : 'none overdue'} />
            <Kpi icon={Camera} label="Photos rejected" value={pct(t.rejected_pct)}
              note={`${t.rejected} retakes needed`} />
          </div>

          {/* Follow-up worklist + urgency */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <Card
              className="lg:col-span-2"
              title="Follow-up worklist"
              subtitle="Each patient's latest screening that still needs action, most urgent first."
              action={
                <div className="inline-flex rounded-lg border border-slate-800 bg-slate-950 p-0.5 text-xs font-semibold">
                  {[['doctor', 'See an eye doctor'], ['retake', 'Retake photo']].map(([id, label]) => (
                    <button key={id} type="button" onClick={() => setWorklistTab(id)}
                      className={`rounded-md px-2.5 py-1 ${worklistTab === id ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-white'}`}>
                      {label}
                    </button>
                  ))}
                </div>
              }
            >
              {worklist.length === 0 ? (
                <p className="rounded-xl bg-slate-950/50 p-6 text-center text-sm text-slate-400">Nothing waiting. Every patient is up to date.</p>
              ) : (
                <ul className="max-h-80 divide-y divide-slate-800 overflow-y-auto pr-1">
                  {worklist.map((w) => {
                    const chip = urgencyChip(w.urgency);
                    return (
                      <li key={w.screening_id} className="flex flex-wrap items-center gap-x-3 gap-y-1.5 py-2.5">
                        <span className={`inline-flex w-36 flex-shrink-0 justify-center rounded-md px-2 py-1 text-[11px] font-semibold ring-1 ring-inset ${chip.className}`}>
                          {chip.label}
                        </span>
                        <div className="min-w-0 flex-1">
                          <p className="truncate text-sm font-semibold text-white">{w.patient_id}</p>
                          <p className="truncate text-xs text-slate-400">
                            {w.urgency === 'retake' ? 'Photo could not be read' : w.action}
                            {w.age ? ` · ${w.age} yrs` : ''} · screened {daysAgo(w.days_since)}
                          </p>
                        </div>
                        {w.overdue && (
                          <span className="rounded-md bg-rose-500/15 px-2 py-0.5 text-[11px] font-bold text-rose-300 ring-1 ring-inset ring-rose-500/40">Overdue</span>
                        )}
                        <button type="button" onClick={() => openScreening(w.screening_id)}
                          className="rounded-lg border border-slate-700 px-2.5 py-1 text-xs font-semibold text-slate-200 hover:bg-slate-800">
                          {opening === w.screening_id ? 'Opening…' : 'Review'}
                        </button>
                      </li>
                    );
                  })}
                </ul>
              )}
            </Card>

            <Card title="How urgent are the findings" subtitle="Screenings analysed by the AI model, by when to see a doctor.">
              {urgencyTotal === 0 ? (
                <p className="text-sm text-slate-400">No analysed screenings in this period.</p>
              ) : (
                <>
                  <div className="flex h-3 w-full overflow-hidden rounded-full bg-slate-800">
                    {Object.entries(data.urgency).filter(([, n]) => n > 0).map(([k, n]) => (
                      <span key={k} title={`${URGENCY_LABELS[k]}: ${n}`}
                        style={{ width: `${(n / urgencyTotal) * 100}%`, backgroundColor: URGENCY_COLORS[k] }}
                        className="h-full border-r-2 border-slate-900 last:border-r-0" />
                    ))}
                  </div>
                  <ul className="mt-4 space-y-2">
                    {Object.entries(data.urgency).map(([k, n]) => (
                      <li key={k} className="flex items-center justify-between text-sm">
                        <span className="flex items-center gap-2 text-slate-200">
                          <span className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: URGENCY_COLORS[k] }} />
                          {URGENCY_LABELS[k]}
                        </span>
                        <span className="font-mono tabular-nums text-slate-300">
                          {n} <span className="text-slate-500">({Math.round((n / urgencyTotal) * 100)}%)</span>
                        </span>
                      </li>
                    ))}
                  </ul>
                </>
              )}
            </Card>
          </div>

          {/* Conditions + daily volume */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <Card title="Conditions found" subtitle={`Share of the ${t.analysed_by_model} screenings analysed by the AI model.`}>
              {data.conditions.length === 0 ? (
                <p className="text-sm text-slate-400">No conditions found in this period.</p>
              ) : (
                data.conditions.map((c) => (
                  <BarRow key={c.id} label={c.name} dot={getDiseaseMeta(c.id).color} value={c.count} max={maxCondition}
                    right={`${c.count} · ${pct(c.pct)}`} />
                ))
              )}
            </Card>

            <Card title="Screenings per day" subtitle="By outcome.">
              <div className="h-64 w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={trend} margin={{ top: 4, right: 8, left: -18, bottom: 0 }}>
                    <CartesianGrid vertical={false} stroke="#8884" />
                    <XAxis dataKey="label" tick={{ fontSize: 11, fill: '#94a3b8' }} tickLine={false} axisLine={false} interval="preserveStartEnd" />
                    <YAxis allowDecimals={false} tick={{ fontSize: 11, fill: '#94a3b8' }} tickLine={false} axisLine={false} />
                    <Tooltip cursor={{ fill: '#8882' }}
                      contentStyle={{ backgroundColor: '#1f1430', border: '1px solid #4c3870', borderRadius: 10, fontSize: 12, color: '#f3effa' }}
                      labelStyle={{ color: '#f3effa', fontWeight: 600 }} />
                    <Legend iconType="square" iconSize={10} wrapperStyle={{ fontSize: 12 }} formatter={(value) => <span style={{ color: "rgb(var(--c-slate-300))" }}>{value}</span>} />
                    {['normal', 'abnormal', 'rejected', 'legacy'].map((k, i, arr) => (
                      <Bar key={k} dataKey={k} name={OUTCOMES[k].label} stackId="day" fill={OUTCOMES[k].color}
                        radius={i === arr.length - 1 ? [4, 4, 0, 0] : 0} maxBarSize={36} />
                    ))}
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </Card>
          </div>

          {/* Age groups + photo quality */}
          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            <Card title="Abnormal results by age group" subtitle="Share of analysed screenings in each age group with a finding (patients with a recorded age).">
              {data.age_groups.every((g) => g.screened === 0) ? (
                <p className="text-sm text-slate-400">No analysed screenings with a recorded age yet. Add the patient's age when uploading.</p>
              ) : (
                data.age_groups.map((g) => (
                  <BarRow key={g.group} label={g.group} value={g.abnormal_pct ?? 0} max={100}
                    right={g.screened ? `${pct(g.abnormal_pct)} of ${g.screened}` : 'none'} />
                ))
              )}
            </Card>

            <Card title="Photo quality"
              subtitle={data.quality.mean_score != null ? `Average quality score ${data.quality.mean_score.toFixed(2)} (0–1). Why photos were rejected:` : 'Why photos were rejected:'}>
              {data.quality.issues.length === 0 ? (
                <p className="text-sm text-slate-400">No photos were rejected in this period.</p>
              ) : (
                data.quality.issues.map((i) => (
                  <BarRow key={i.issue} label={i.issue} value={i.count} max={maxIssue} right={String(i.count)} />
                ))
              )}
              <p className="mt-3 text-xs text-slate-500">Tip: most retakes come from blur. Ask the photographer to refocus and keep the patient's eye still.</p>
            </Card>
          </div>

          {/* Records table */}
          <Card
            title="All screenings"
            subtitle={`${rows.length} of ${data.screenings.length} screenings in this period`}
            action={
              <button type="button" onClick={exportCsv} disabled={!rows.length}
                className="inline-flex items-center gap-1.5 rounded-lg border border-slate-700 px-3 py-1.5 text-xs font-semibold text-slate-200 hover:bg-slate-800 disabled:opacity-50">
                <Download className="h-3.5 w-3.5" /> Export CSV
              </button>
            }
          >
            <div className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center">
              <div className="relative flex-1">
                <Search className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                <input value={search} onChange={(e) => { setSearch(e.target.value); setVisible(25); }} placeholder="Search patient ID"
                  className="w-full rounded-lg border border-slate-700 bg-slate-950/50 py-2 pl-9 pr-3 text-sm text-white placeholder-slate-500 focus:border-sky-500 focus:outline-none" />
              </div>
              <select value={outcome} onChange={(e) => { setOutcome(e.target.value); setVisible(25); }}
                className="rounded-lg border border-slate-700 bg-slate-950/50 px-3 py-2 text-sm text-slate-200">
                <option value="all">All results</option>
                {Object.entries(OUTCOMES).map(([k, o]) => <option key={k} value={k}>{o.label}</option>)}
              </select>
              <select value={condition} onChange={(e) => { setCondition(e.target.value); setVisible(25); }}
                className="rounded-lg border border-slate-700 bg-slate-950/50 px-3 py-2 text-sm text-slate-200">
                <option value="all">All conditions</option>
                {conditionOptions.map(([id, name]) => <option key={id} value={id}>{name}</option>)}
              </select>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full min-w-[860px] text-left text-sm">
                <thead className="border-b border-slate-800 text-xs uppercase tracking-wider text-slate-400">
                  <tr>
                    <th className="py-2.5 pr-3 font-semibold">Date</th>
                    <th className="py-2.5 pr-3 font-semibold">Patient</th>
                    <th className="py-2.5 pr-3 font-semibold">Age · Eye</th>
                    <th className="py-2.5 pr-3 font-semibold">Result</th>
                    <th className="py-2.5 pr-3 font-semibold">Urgency</th>
                    <th className="py-2.5 pr-3 text-right font-semibold">Confidence</th>
                    <th className="py-2.5 text-right font-semibold">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/70">
                  {rows.slice(0, visible).map((r) => {
                    const chip = r.outcome === 'rejected' ? urgencyChip('retake') : r.outcome === 'abnormal' ? urgencyChip(r.urgency) : null;
                    const color = r.outcome === 'abnormal' ? getDiseaseMeta(r.condition).color : OUTCOMES[r.outcome].color;
                    const label = r.outcome === 'abnormal' ? r.condition_name : r.outcome === 'normal' ? 'No abnormality' : r.outcome === 'rejected' ? 'Photo rejected' : 'Older engine result';
                    return (
                      <tr key={r.id} className="hover:bg-slate-800/30">
                        <td className="whitespace-nowrap py-2.5 pr-3 font-mono text-xs text-slate-400">{fmtDateTime(r.date)}</td>
                        <td className="py-2.5 pr-3">
                          <span className="font-semibold text-white">{r.patient_id || 'Anonymous'}</span>
                          {r.visits > 1 && <span className="ml-2 rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-400">{r.visits} visits</span>}
                        </td>
                        <td className="whitespace-nowrap py-2.5 pr-3 text-slate-300">{r.age ? `${r.age} yrs` : '–'} · {r.eye && r.eye !== 'Unknown' ? r.eye : '–'}</td>
                        <td className="py-2.5 pr-3">
                          <span className="flex items-center gap-2 text-slate-100">
                            <span className="h-2.5 w-2.5 flex-shrink-0 rounded-full" style={{ backgroundColor: color }} />
                            {label}
                            {r.other_conditions > 0 && <span className="rounded bg-slate-800 px-1.5 py-0.5 font-mono text-[10px] text-slate-400">+{r.other_conditions}</span>}
                          </span>
                        </td>
                        <td className="py-2.5 pr-3">
                          {chip ? <span className={`whitespace-nowrap rounded-md px-2 py-0.5 text-[11px] font-semibold ring-1 ring-inset ${chip.className}`}>{chip.label}</span> : <span className="text-slate-500">–</span>}
                        </td>
                        <td className="py-2.5 pr-3 text-right font-mono tabular-nums text-slate-300">{r.confidence != null ? `${Math.round(r.confidence * 100)}%` : '–'}</td>
                        <td className="whitespace-nowrap py-2.5 text-right">
                          <button type="button" onClick={() => openScreening(r.id)}
                            className="rounded-lg border border-slate-700 px-2.5 py-1 text-xs font-semibold text-slate-200 hover:bg-slate-800">
                            {opening === r.id ? 'Opening…' : 'Review'}
                          </button>
                          {r.outcome !== 'rejected' && (
                            <a href={`/api/report/${r.id}?lang=en`} download title="Download PDF report (English)"
                              className="ml-2 inline-flex items-center rounded-lg border border-slate-700 p-1.5 text-slate-300 hover:bg-slate-800">
                              <FileText className="h-3.5 w-3.5" />
                            </a>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              {rows.length === 0 && <p className="py-8 text-center text-sm text-slate-400">No screenings match these filters.</p>}
            </div>
            {visible < rows.length && (
              <div className="mt-4 text-center">
                <button type="button" onClick={() => setVisible((v) => v + 25)}
                  className="rounded-lg border border-slate-700 px-4 py-2 text-sm font-semibold text-slate-200 hover:bg-slate-800">
                  Show more ({rows.length - visible} left)
                </button>
              </div>
            )}
          </Card>

          {t.legacy > 0 && (
            <p className="text-xs text-slate-500">
              {t.legacy} older screenings were analysed by the previous engine before the AI model was trained. They count towards screening
              volume and photo quality, but not towards condition figures, because their findings are not reliable.
            </p>
          )}
        </>
      )}
    </div>
  );
}
