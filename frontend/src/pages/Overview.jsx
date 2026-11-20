import React, { useEffect, useState } from 'react';
import { Activity, Cpu, Database, Gauge, RefreshCw, Zap } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../api';

function Metric({ label, value, detail, icon: Icon }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">{label}</p>
          <p className="mt-2 text-2xl font-bold text-slate-900">{value ?? 'Unavailable'}</p>
          {detail && <p className="mt-1 text-xs text-slate-500">{detail}</p>}
        </div>
        {Icon && <div className="rounded-lg bg-blue-50 p-2 text-blue-600"><Icon className="h-5 w-5" /></div>}
      </div>
    </div>
  );
}

export default function Overview({ setActiveTab }) {
  const [overview, setOverview] = useState(null);
  const [models, setModels] = useState(null);
  const [experiments, setExperiments] = useState([]);
  const [selected, setSelected] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const load = async () => {
    setLoading(true);
    setError('');
    try {
      const [overviewData, modelData, runData] = await Promise.all([
        api.getOverview(),
        api.getModelsStatus(),
        api.getExperiments(),
      ]);
      setOverview(overviewData);
      setModels(modelData);
      setExperiments(runData.experiments || []);
      if ((runData.experiments || []).length) {
        const latest = runData.experiments[0];
        const detail = await api.getExperimentDetails(latest.run_id);
        setSelected(detail);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const hw = overview?.hardware || {};
  const strategyRows = selected?.strategies || [];
  const accuracyData = strategyRows.map((row) => ({ name: row.strategy, accuracy: Number(row.accuracy) }));
  const latencyData = strategyRows.map((row) => ({
    name: row.strategy,
    mean: Number(row.mean_latency_ms),
    p95: row.p95_latency_ms == null ? null : Number(row.p95_latency_ms),
  }));
  const modelEntries = Object.values(models?.models || {});

  if (loading) return <div className="p-8 text-sm text-slate-500">Loading system overview...</div>;
  if (error) return (
    <div className="m-6 rounded-xl border border-red-200 bg-red-50 p-5 text-red-700">
      <p className="font-semibold">Overview unavailable</p><p className="mt-1 text-sm">{error}</p>
      <button onClick={load} className="mt-3 rounded-lg bg-white px-3 py-2 text-sm border border-red-200">Retry</button>
    </div>
  );

  return (
    <div className="mx-auto max-w-7xl space-y-6 p-6">
      <section className="rounded-2xl border border-blue-100 bg-white p-6 shadow-sm">
        <div className="flex flex-col justify-between gap-5 md:flex-row md:items-center">
          <div>
            <p className="text-xs font-bold uppercase tracking-wider text-blue-600">Research platform</p>
            <h1 className="mt-2 text-2xl font-bold text-slate-950">ADAPT-EDGE</h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">
              Decision-Selective Computation for Resource-Bounded Edge AI Model Switching
            </p>
          </div>
          <button onClick={() => setActiveTab('livedemo')} className="inline-flex items-center justify-center gap-2 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-700">
            <Zap className="h-4 w-4" /> Open Live Demo
          </button>
        </div>
      </section>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <Metric label="CPU" value={hw.cpu_percent == null ? 'Unavailable' : `${hw.cpu_percent}%`} detail="Live system utilization" icon={Cpu} />
        <Metric label="RAM" value={hw.ram_percent == null ? 'Unavailable' : `${hw.ram_percent}%`} detail={hw.ram_used_mb != null ? `${Math.round(hw.ram_used_mb)} MB used` : 'Not reported'} icon={Activity} />
        <Metric label="GPU" value={hw.gpu_utilization == null ? 'Unavailable' : `${hw.gpu_utilization}%`} detail={hw.gpu_name || 'No GPU detected'} icon={Gauge} />
        <Metric label="Inference device" value={models?.device || hw.device || 'Unavailable'} detail={overview?.backend_online ? 'Backend connected' : 'Backend unavailable'} icon={Database} />
      </div>

      <section className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="font-bold text-slate-900">Model and checkpoint status</h2>
          <div className="mt-4 space-y-2">
            {modelEntries.map((model) => (
              <div key={model.name} className="flex items-center justify-between rounded-lg border border-slate-100 px-3 py-2.5">
                <div><p className="text-sm font-semibold text-slate-800">{model.name}</p><p className="text-xs text-slate-500">{model.checkpoint_path}</p></div>
                <span className={`rounded-full px-2 py-1 text-xs font-semibold ${model.integrity === 'VERIFIED' ? 'bg-green-50 text-green-700' : model.integrity === 'MISMATCH' ? 'bg-amber-50 text-amber-700' : 'bg-red-50 text-red-700'}`}>
                  {model.loaded ? 'Loaded' : 'Not loaded'} · {model.integrity}
                </span>
              </div>
            ))}
            {modelEntries.length === 0 && <p className="text-sm text-slate-500">No model status returned by the backend.</p>}
          </div>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between"><h2 className="font-bold text-slate-900">Historical experiment</h2><RefreshCw onClick={load} className="h-4 w-4 cursor-pointer text-slate-400" /></div>
          {selected ? (
            <div className="mt-4">
              <p className="text-sm font-semibold text-slate-800">{selected.run_id}</p>
              <p className="mt-1 text-xs text-slate-500">Historical run · {selected.total_traces ?? 0} per-image traces</p>
            </div>
          ) : <p className="mt-4 text-sm text-slate-500">No compatible historical experiments found.</p>}
          <p className="mt-4 text-xs text-slate-500">Current-session inference events are displayed in Live Demo history.</p>
        </div>
      </section>

      {accuracyData.length > 0 && (
        <section className="grid gap-6 lg:grid-cols-2">
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="font-bold text-slate-900">Historical accuracy</h2>
            <div className="mt-4 h-72"><ResponsiveContainer width="100%" height="100%"><BarChart data={accuracyData}><CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0"/><XAxis dataKey="name" tick={{fontSize:10}}/><YAxis/><Tooltip/><Bar dataKey="accuracy" fill="#2563eb"/></BarChart></ResponsiveContainer></div>
          </div>
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="font-bold text-slate-900">Historical latency</h2>
            <div className="mt-4 h-72"><ResponsiveContainer width="100%" height="100%"><BarChart data={latencyData}><CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0"/><XAxis dataKey="name" tick={{fontSize:10}}/><YAxis unit=" ms"/><Tooltip/><Bar dataKey="mean" name="Mean" fill="#0891b2"/><Bar dataKey="p95" name="P95" fill="#f59e0b"/></BarChart></ResponsiveContainer></div>
          </div>
        </section>
      )}
    </div>
  );
}
