import React, { useEffect, useState } from 'react';
import { Activity, CheckCircle2, Cpu, Loader2, XCircle } from 'lucide-react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ReferenceDot } from 'recharts';
import { getDiseaseMeta } from '../lib/diseases';

const LIVE = ['starting', 'training', 'evaluating'];
const STATUS = {
  not_started: { label: 'Not started', className: 'bg-slate-700/50 text-slate-300 ring-slate-600' },
  starting: { label: 'Starting', className: 'bg-sky-500/15 text-sky-300 ring-sky-500/40' },
  training: { label: 'Training', className: 'bg-sky-500/15 text-sky-300 ring-sky-500/40' },
  evaluating: { label: 'Testing on held-out images', className: 'bg-amber-500/15 text-amber-300 ring-amber-500/40' },
  completed: { label: 'Completed', className: 'bg-emerald-500/15 text-emerald-300 ring-emerald-500/40' },
  stopped: { label: 'Stopped', className: 'bg-rose-500/15 text-rose-300 ring-rose-500/40' },
  failed: { label: 'Failed', className: 'bg-rose-500/15 text-rose-300 ring-rose-500/40' },
};

const pct = (v) => (v == null ? '–' : `${Math.round(v * 100)}%`);

function formatEta(seconds) {
  if (seconds == null) return '';
  if (seconds < 90) return 'about a minute left';
  const m = Math.round(seconds / 60);
  return m < 60 ? `about ${m} min left` : `about ${Math.floor(m / 60)} h ${m % 60} min left`;
}

function useTrainingProgress() {
  const [data, setData] = useState(null);
  useEffect(() => {
    let timer;
    let cancelled = false;
    const load = async () => {
      try {
        const res = await fetch('/api/training/progress');
        if (res.ok) {
          const json = await res.json();
          if (!cancelled) setData(json);
          if (!cancelled && LIVE.includes(json.status)) timer = setTimeout(load, 10000);
        }
      } catch {
        if (!cancelled) timer = setTimeout(load, 15000);
      }
    };
    load();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, []);
  return data;
}

export default function TrainingProgress({ compact = false }) {
  const data = useTrainingProgress();
  if (!data || data.status === 'not_started') return null;

  const status = STATUS[data.status] || STATUS.training;
  const total = data.total_epochs || 0;
  const done = data.completed_epochs + (data.status === 'training' ? data.current_epoch_fraction : 0);
  const progress = data.status === 'completed' ? 1 : total ? Math.min(0.99, done / total) : 0;
  const live = LIVE.includes(data.status);
  const StatusIcon = data.status === 'completed' ? CheckCircle2 : ['failed', 'stopped'].includes(data.status) ? XCircle : Loader2;

  const headline =
    data.status === 'completed'
      ? 'Training finished and tested on held-out images'
      : data.status === 'evaluating'
        ? 'All rounds done: measuring accuracy on the held-out test set'
        : data.status === 'training'
          ? `Round ${Math.min(data.completed_epochs + 1, total)} of ${total} · ${formatEta(data.eta_seconds)}`
          : status.label;

  if (compact) {
    return (
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1.5 rounded-xl border border-slate-800 bg-slate-950/60 px-4 py-2.5 text-xs">
        <span className="flex items-center gap-1.5 font-semibold text-white">
          <StatusIcon className={`h-3.5 w-3.5 ${live ? 'animate-spin text-sky-400' : 'text-emerald-400'}`} />
          Disease AI model
        </span>
        <span className="text-slate-400">{headline}</span>
        <div className="h-1.5 min-w-[120px] flex-1 overflow-hidden rounded-full bg-slate-800">
          <div className="h-full rounded-full bg-sky-400 transition-all duration-700" style={{ width: `${progress * 100}%` }} />
        </div>
        {data.best_val_auc != null && (
          <span className="font-mono text-slate-300">best AUC {data.best_val_auc.toFixed(3)}</span>
        )}
        <span className="text-slate-500">Details on the Validation tab</span>
      </div>
    );
  }

  const chartData = data.epochs.map((e) => ({ epoch: e.epoch, auc: e.val_auc, loss: e.loss }));
  const testRows = data.test
    ? Object.entries(data.test).sort((a, b) => (a[0] === 'abnormal' ? -1 : b[0] === 'abnormal' ? 1 : b[1].n_pos - a[1].n_pos))
    : [];

  return (
    <section className="space-y-5 rounded-2xl border border-slate-800 bg-slate-900 p-5 shadow-xl sm:p-6">
      <header className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-sky-400" />
            <h2 className="font-heading text-lg font-bold text-white">Disease AI model training</h2>
            <span className={`inline-flex items-center gap-1.5 rounded-md px-2 py-0.5 text-xs font-semibold ring-1 ring-inset ${status.className}`}>
              <StatusIcon className={`h-3 w-3 ${live ? 'animate-spin' : ''}`} />
              {status.label}
            </span>
          </div>
          <p className="mt-1 text-sm text-slate-400">
            Deep-learning models learning 10 eye conditions plus an overall abnormality check from 6,395 real fundus photos (RFMiD + ODIR).
            {live && ' Updates every 10 seconds.'}
          </p>
        </div>
        {data.status === 'completed' && (
          <span className={`text-xs font-semibold ${data.model_loaded ? 'text-emerald-400' : 'text-amber-300'}`}>
            {data.model_loaded ? 'Model is active in screening' : 'Trained; restart the backend to activate it'}
          </span>
        )}
      </header>

      <div>
        <div className="mb-1.5 flex items-baseline justify-between text-sm">
          <span className="font-medium text-slate-200">{headline}</span>
          <span className="font-mono text-xs text-slate-400">{Math.round(progress * 100)}%</span>
        </div>
        <div className="h-2.5 overflow-hidden rounded-full bg-slate-800">
          <div className="h-full rounded-full bg-sky-400 transition-all duration-700" style={{ width: `${progress * 100}%` }} />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
          <p className="text-[11px] text-slate-400">Rounds completed</p>
          <p className="mt-1 font-mono text-xl font-bold text-white">{data.completed_epochs}<span className="text-sm text-slate-500"> / {total}</span></p>
        </div>
        <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
          <p className="text-[11px] text-slate-400">Best validation AUC</p>
          <p className="mt-1 font-mono text-xl font-bold text-white">{data.best_val_auc?.toFixed(3) ?? '–'}</p>
          {data.best_epoch && <p className="text-[11px] text-slate-500">round {data.best_epoch}</p>}
        </div>
        <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
          <p className="text-[11px] text-slate-400">Latest training loss</p>
          <p className="mt-1 font-mono text-xl font-bold text-white">{data.epochs.at(-1)?.loss.toFixed(3) ?? '–'}</p>
          <p className="text-[11px] text-slate-500">lower is better</p>
        </div>
        <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
          <p className="flex items-center gap-1 text-[11px] text-slate-400"><Cpu className="h-3 w-3" /> GPU</p>
          {data.gpu ? (
            <>
              <p className="mt-1 font-mono text-xl font-bold text-white">{data.gpu.utilization}%</p>
              <p className="text-[11px] text-slate-500">
                {data.gpu.temperature_c}°C · {(data.gpu.memory_used_mb / 1024).toFixed(1)}/{(data.gpu.memory_total_mb / 1024).toFixed(1)} GB
              </p>
            </>
          ) : (
            <p className="mt-1 text-sm text-slate-500">idle</p>
          )}
        </div>
      </div>

      {chartData.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-slate-200">Validation AUC after each round</h3>
          <p className="text-xs text-slate-500">
            Measured on 1,594 photos the model does not train on. 1.0 = perfect separation of sick and healthy eyes, 0.5 = guessing.
          </p>
          <div className="mt-3 h-56 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
                <CartesianGrid stroke="#1e293b" vertical={false} />
                <XAxis dataKey="epoch" type="number" domain={[1, total || 'dataMax']} tickCount={Math.min(total, 13)} tick={{ fontSize: 11, fill: '#94a3b8' }} stroke="#334155" />
                <YAxis domain={[(min) => Math.max(0.5, Math.floor((min - 0.02) * 50) / 50), 1]} tickFormatter={(v) => v.toFixed(2)} tick={{ fontSize: 11, fill: '#94a3b8' }} stroke="#334155" width={44} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#020617', border: '1px solid #334155', borderRadius: 8, fontSize: 12 }}
                  labelFormatter={(e) => `Round ${e}`}
                  formatter={(v, name) => [Number(v).toFixed(4), name === 'auc' ? 'Validation AUC' : 'Loss']}
                />
                <Line type="monotone" dataKey="auc" stroke="#38bdf8" strokeWidth={2} dot={{ r: 3, fill: '#38bdf8', strokeWidth: 0 }} activeDot={{ r: 5 }} isAnimationActive={false} />
                {data.best_epoch && (
                  <ReferenceDot x={data.best_epoch} y={data.best_val_auc} r={6} fill="#020617" stroke="#e0f2fe" strokeWidth={2} />
                )}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <p className="mt-1 text-xs text-slate-500">Ringed point = best round, which is the checkpoint kept.</p>
        </div>
      )}

      {testRows.length > 0 && (
        <div>
          <h3 className="text-sm font-semibold text-slate-200">Accuracy on the held-out test set</h3>
          <p className="text-xs text-slate-500">
            {testRows[0]?.[1]?.n?.toLocaleString()} test photos never used for training or tuning. Conditions with few test cases have uncertain scores.
            {data.active_model && <> Model in use: <span className="font-mono text-slate-400">{data.active_model}</span>.</>}
          </p>
          {data.test_summary?.verdict && (
            <div className="mt-3 grid grid-cols-1 gap-3 sm:grid-cols-3">
              <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
                <p className="text-[11px] text-slate-400">Healthy eyes reported fully normal</p>
                <p className="mt-1 font-mono text-xl font-bold text-white">{pct(data.test_summary.verdict.healthy_with_no_condition_reported)}</p>
                <p className="text-[11px] text-slate-500">of {data.test_summary.verdict.healthy_eyes} healthy test eyes</p>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
                <p className="text-[11px] text-slate-400">Diseased eyes flagged as abnormal</p>
                <p className="mt-1 font-mono text-xl font-bold text-white">{pct(data.test_summary.verdict.sensitivity)}</p>
                <p className="text-[11px] text-slate-500">overall verdict sensitivity</p>
              </div>
              <div className="rounded-xl border border-slate-800 bg-slate-950/60 p-3">
                <p className="text-[11px] text-slate-400">Healthy eyes correctly called normal</p>
                <p className="mt-1 font-mono text-xl font-bold text-white">{pct(data.test_summary.verdict.specificity)}</p>
                <p className="text-[11px] text-slate-500">overall verdict specificity</p>
              </div>
            </div>
          )}
          <div className="mt-3 overflow-x-auto">
            <table className="w-full min-w-[560px] text-left text-xs">
              <thead className="border-b border-slate-800 text-slate-400">
                <tr>
                  <th className="py-2 pr-3 font-semibold">Condition</th>
                  <th className="py-2 pr-3 text-right font-semibold">Test cases</th>
                  <th className="py-2 pr-3 text-right font-semibold">AUC</th>
                  <th className="py-2 pr-3 text-right font-semibold">Finds (sensitivity)</th>
                  <th className="py-2 text-right font-semibold">Clears healthy (specificity)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70 font-mono text-slate-200">
                {testRows.map(([id, m]) => (
                  <tr key={id}>
                    <td className="py-2 pr-3 font-sans">
                      <span className="mr-2 inline-block h-2 w-2 rounded-full" style={{ backgroundColor: id === 'abnormal' ? '#94a3b8' : getDiseaseMeta(id).color }} />
                      {id === 'abnormal' ? 'Any abnormality' : getDiseaseMeta(id, data.class_names?.[id]).name}
                    </td>
                    <td className={`py-2 pr-3 text-right ${m.n_pos < 15 ? 'text-amber-300' : ''}`}>{m.n_pos}</td>
                    <td className="py-2 pr-3 text-right">{m.auc != null ? m.auc.toFixed(3) : '–'}</td>
                    <td className="py-2 pr-3 text-right">{pct(m.sensitivity)}</td>
                    <td className="py-2 text-right">{pct(m.specificity)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {data.excluded_classes?.length > 0 && (
            <p className="mt-2 text-xs text-slate-500">
              Not trained (too few labelled photos): {data.excluded_classes.map((id) => getDiseaseMeta(id).name).join(', ')}.
            </p>
          )}
        </div>
      )}
    </section>
  );
}
