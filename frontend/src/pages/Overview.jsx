import React, { useEffect, useState } from 'react';
import {
  Activity, BookOpen, BarChart2, Cpu, Database,
  FlaskConical, Gauge, LayoutDashboard, RefreshCw,
  Server, Zap, ChevronRight, CheckCircle2, XCircle, AlertCircle,
} from 'lucide-react';
import {
  Bar, BarChart, CartesianGrid, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis, Legend,
} from 'recharts';
import { api } from '../api';

/* ─── small reusable atoms ────────────────────────────────────────────────── */

function HwMetric({ label, value, detail, icon: Icon, status }) {
  const statusColor = {
    critical: 'border-red-200 bg-red-50',
    warning:  'border-amber-200 bg-amber-50',
    normal:   'border-slate-200 bg-white',
  }[status] ?? 'border-slate-200 bg-white';
  return (
    <div className={`rounded-xl border p-5 shadow-sm ${statusColor}`}>
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

function NavCard({ icon: Icon, label, sub, tab, setActiveTab, color = 'blue' }) {
  const colors = {
    blue:   'from-blue-500 to-indigo-600',
    cyan:   'from-cyan-500 to-blue-500',
    violet: 'from-violet-500 to-purple-600',
    amber:  'from-amber-400 to-orange-500',
    green:  'from-green-500 to-emerald-600',
  };
  return (
    <button
      onClick={() => setActiveTab(tab)}
      className="group flex flex-col gap-3 rounded-xl border border-slate-100 bg-white p-5 shadow-sm hover:shadow-md transition-all text-left"
    >
      <div className={`w-9 h-9 rounded-lg bg-gradient-to-br ${colors[color]} flex items-center justify-center`}>
        <Icon className="h-5 w-5 text-white" />
      </div>
      <div>
        <p className="text-sm font-semibold text-slate-900 group-hover:text-blue-600 transition-colors">{label}</p>
        <p className="mt-0.5 text-xs text-slate-500">{sub}</p>
      </div>
      <ChevronRight className="h-4 w-4 text-slate-300 group-hover:text-blue-500 transition-colors self-end" />
    </button>
  );
}

function IntegrityBadge({ loaded, integrity }) {
  if (integrity === 'VERIFIED')
    return <span className="inline-flex items-center gap-1 rounded-full bg-green-50 px-2 py-1 text-xs font-semibold text-green-700"><CheckCircle2 className="h-3 w-3" />{loaded ? 'Loaded' : 'Ready'}</span>;
  if (integrity === 'MISMATCH')
    return <span className="inline-flex items-center gap-1 rounded-full bg-amber-50 px-2 py-1 text-xs font-semibold text-amber-700"><AlertCircle className="h-3 w-3" />Mismatch</span>;
  return <span className="inline-flex items-center gap-1 rounded-full bg-red-50 px-2 py-1 text-xs font-semibold text-red-700"><XCircle className="h-3 w-3" />Missing</span>;
}

function SectionHeader({ title, action }) {
  return (
    <div className="flex items-center justify-between mb-4">
      <h2 className="font-bold text-slate-900">{title}</h2>
      {action}
    </div>
  );
}

/* ─── main component ──────────────────────────────────────────────────────── */

export default function Overview({ setActiveTab }) {
  const [overview,    setOverview]    = useState(null);
  const [models,      setModels]      = useState(null);
  const [experiments, setExperiments] = useState([]);
  const [selected,    setSelected]    = useState(null);
  const [datasets,    setDatasets]    = useState([]);
  const [training,    setTraining]    = useState(null);
  const [customModels,setCustomModels]= useState([]);
  const [loading,     setLoading]     = useState(true);
  const [error,       setError]       = useState('');

  const load = async () => {
    setLoading(true);
    setError('');
    try {
      const [overviewData, modelData, runData, dsData, trainData, customData] = await Promise.all([
        api.getOverview(),
        api.getModelsStatus(),
        api.getExperiments(),
        api.getDatasets().catch(() => ({ datasets: [] })),
        api.getTrainingStatus().catch(() => ({ telemetry: null })),
        api.getCustomModels().catch(() => ({ models: [] })),
      ]);
      setOverview(overviewData);
      setModels(modelData);
      setExperiments(runData.experiments || []);
      setDatasets(dsData.datasets || []);
      setTraining(trainData.telemetry || null);
      setCustomModels(customData.models || []);
      if ((runData.experiments || []).length) {
        const detail = await api.getExperimentDetails(runData.experiments[0].run_id).catch(() => null);
        setSelected(detail);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  if (loading) return <div className="p-8 text-sm text-slate-500">Loading system overview…</div>;
  if (error) return (
    <div className="m-6 rounded-xl border border-red-200 bg-red-50 p-5 text-red-700">
      <p className="font-semibold">Overview unavailable</p>
      <p className="mt-1 text-sm">{error}</p>
      <button onClick={load} className="mt-3 rounded-lg bg-white px-3 py-2 text-sm border border-red-200">Retry</button>
    </div>
  );

  const hw            = overview?.hardware || {};
  const expMeta       = overview?.experiments || {};
  const strategyRows  = selected?.strategies || [];
  const accuracyData  = strategyRows.map(r => ({ name: r.strategy, accuracy: Number(r.accuracy) }));
  const latencyData   = strategyRows.map(r => ({
    name: r.strategy,
    mean: Number(r.mean_latency_ms),
    p95:  r.p95_latency_ms == null ? null : Number(r.p95_latency_ms),
  }));
  const modelEntries  = Object.values(models?.models || {});
  const trainingHist  = training?.history || [];
  const cifar10       = datasets.find(d => d.id === 'cifar10');
  const uploadedDS    = datasets.filter(d => d.id !== 'cifar10');

  /* architecture catalogue (static, matches models/ implementations) */
  const architectures = [
    { name: 'MobileNetV3-Small', role: 'Model A — Lightweight', params: '~2.5 M', desc: 'Fast edge inference, low VRAM.' },
    { name: 'MobileNetV3-Large', role: 'Model B — Full',        params: '~5.4 M', desc: 'Higher accuracy, deeper features.' },
    { name: 'Probe Block 4',     role: 'Intermediate probe',    params: '—',       desc: 'Early-layer evidence extraction.' },
    { name: 'Probe Block 9',     role: 'Intermediate probe',    params: '—',       desc: 'Mid-layer evidence extraction.' },
    { name: 'Probe Block 13',    role: 'Intermediate probe',    params: '—',       desc: 'Late-layer evidence extraction.' },
  ];

  return (
    <div className="mx-auto max-w-7xl space-y-8 p-6">

      {/* ── Hero banner ───────────────────────────────────────────────── */}
      <section className="rounded-2xl border border-blue-100 bg-white p-6 shadow-sm">
        <div className="flex flex-col justify-between gap-5 md:flex-row md:items-center">
          <div>
            <p className="text-xs font-bold uppercase tracking-wider text-blue-600">Research platform</p>
            <h1 className="mt-2 text-2xl font-bold text-slate-950">ADAPT-EDGE</h1>
            <p className="mt-2 max-w-2xl text-sm leading-6 text-slate-600">
              Decision-Selective Computation for Resource-Bounded Edge AI Model Switching
            </p>
          </div>
          <div className="flex gap-3">
            <button
              onClick={load}
              className="inline-flex items-center gap-2 rounded-lg border border-slate-200 bg-white px-3 py-2.5 text-sm font-medium text-slate-700 hover:bg-slate-50"
            >
              <RefreshCw className="h-4 w-4" /> Refresh
            </button>
            <button
              onClick={() => setActiveTab('livedemo')}
              className="inline-flex items-center gap-2 rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white hover:bg-blue-700"
            >
              <Zap className="h-4 w-4" /> Open Live Demo
            </button>
          </div>
        </div>

        {/* KPI strip */}
        <div className="mt-5 grid grid-cols-2 gap-4 border-t border-slate-100 pt-5 sm:grid-cols-4">
          {[
            { label: 'Total experiments',  value: expMeta.total_runs ?? '—' },
            { label: 'Latest accuracy',    value: expMeta.latest_accuracy_pct != null ? `${expMeta.latest_accuracy_pct.toFixed(1)} %` : '—' },
            { label: 'Latest latency',     value: expMeta.latest_latency_ms  != null ? `${expMeta.latest_latency_ms.toFixed(1)} ms`  : '—' },
            { label: 'Session candidates', value: customModels.length },
          ].map(k => (
            <div key={k.label}>
              <p className="text-xs text-slate-500">{k.label}</p>
              <p className="mt-1 text-xl font-bold text-slate-900">{k.value}</p>
            </div>
          ))}
        </div>
      </section>

      {/* ── Hardware metrics ──────────────────────────────────────────── */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <HwMetric label="CPU"              value={hw.cpu_percent   == null ? 'Unavailable' : `${hw.cpu_percent}%`}    detail="Live system utilization"                                              icon={Cpu}            status="normal" />
        <HwMetric label="RAM"              value={hw.ram_percent   == null ? 'Unavailable' : `${hw.ram_percent}%`}    detail={hw.ram_used_mb != null ? `${Math.round(hw.ram_used_mb)} MB used` : 'Not reported'}  icon={Activity}       status="normal" />
        <HwMetric label="GPU"              value={hw.gpu_utilization == null ? 'Unavailable' : `${hw.gpu_utilization}%`} detail={hw.gpu_name || 'CPU Only'}                                         icon={Gauge}          status={hw.gpu_utilization == null ? 'normal' : 'normal'} />
        <HwMetric label="Inference device" value={models?.device || hw.device || 'cpu'}                                 detail={overview?.backend_online ? 'Backend connected' : 'Backend unavailable'} icon={Server}       status="normal" />
      </div>

      {/* ── Quick navigation ─────────────────────────────────────────── */}
      <section>
        <SectionHeader title="Workbench" />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <NavCard icon={Database}       label="Dataset & Training"   sub="Manage datasets, train candidate models"   tab="datatrain"  setActiveTab={setActiveTab} color="cyan"   />
          <NavCard icon={Zap}            label="Live Demo"            sub="Run ADAPT-EDGE inference on live images"   tab="livedemo"   setActiveTab={setActiveTab} color="blue"   />
          <NavCard icon={BarChart2}      label="Results & Comparison" sub="Compare strategies, view metrics"          tab="results"    setActiveTab={setActiveTab} color="violet" />
          <NavCard icon={LayoutDashboard} label="Settings"            sub="Cache, thresholds, resource policy"        tab="settings"   setActiveTab={setActiveTab} color="amber"  />
        </div>
      </section>

      {/* ── Dataset statistics ───────────────────────────────────────── */}
      <section className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <SectionHeader title="Dataset statistics" action={
            <button onClick={() => setActiveTab('datatrain')} className="text-xs text-blue-600 hover:underline flex items-center gap-1">Manage <ChevronRight className="h-3 w-3" /></button>
          } />
          {cifar10 ? (
            <div className="space-y-3">
              <div className="flex items-center justify-between rounded-lg bg-slate-50 px-3 py-2.5">
                <div>
                  <p className="text-sm font-semibold text-slate-800">CIFAR-10</p>
                  <p className="text-xs text-slate-500">10 classes · 32×32 · {cifar10.total_samples?.toLocaleString() ?? '60,000'} samples</p>
                </div>
                <span className={`rounded-full px-2 py-1 text-xs font-semibold ${cifar10.status === 'ready' ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'}`}>
                  {cifar10.status === 'ready' ? 'Ready' : 'Not found'}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-2">
                {[
                  { label: 'Train', value: '50,000' },
                  { label: 'Test',  value: '10,000' },
                  { label: 'Classes', value: '10' },
                ].map(s => (
                  <div key={s.label} className="rounded-lg bg-blue-50 p-3 text-center">
                    <p className="text-xs text-slate-500">{s.label}</p>
                    <p className="mt-1 text-sm font-bold text-slate-900">{s.value}</p>
                  </div>
                ))}
              </div>
              <div>
                <p className="text-xs font-semibold text-slate-600 mb-1.5">Classes</p>
                <div className="flex flex-wrap gap-1">
                  {(cifar10.classes || []).map(c => (
                    <span key={c} className="rounded bg-slate-100 px-2 py-0.5 text-xs text-slate-600">{c}</span>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <p className="text-sm text-slate-500">CIFAR-10 not found. Download it via Dataset &amp; Training.</p>
          )}
          {uploadedDS.length > 0 && (
            <div className="mt-4 border-t border-slate-100 pt-4 space-y-2">
              <p className="text-xs font-semibold text-slate-600">Uploaded datasets</p>
              {uploadedDS.map(d => (
                <div key={d.id} className="flex items-center justify-between rounded-lg border border-slate-100 px-3 py-2">
                  <div>
                    <p className="text-sm font-semibold text-slate-800">{d.name}</p>
                    <p className="text-xs text-slate-500">{d.classes_count} classes · {d.total_samples} images</p>
                  </div>
                  <span className="rounded-full bg-green-50 px-2 py-0.5 text-xs font-semibold text-green-700">Ready</span>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* ── Training status ───────────────────────────────────────── */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <SectionHeader title="Training status" action={
            <button onClick={() => setActiveTab('datatrain')} className="text-xs text-blue-600 hover:underline flex items-center gap-1">Go to training <ChevronRight className="h-3 w-3" /></button>
          } />
          {training ? (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className={`rounded-full px-3 py-1 text-xs font-semibold ${
                  training.status === 'RUNNING'    ? 'bg-blue-50 text-blue-700'   :
                  training.status === 'COMPLETED'  ? 'bg-green-50 text-green-700' :
                  training.status === 'FAILED'     ? 'bg-red-50 text-red-700'     :
                  training.status === 'STOPPED'    ? 'bg-amber-50 text-amber-700' :
                  'bg-slate-100 text-slate-600'
                }`}>{training.status}</span>
                {training.status === 'RUNNING' && (
                  <span className="text-xs text-slate-500">
                    Epoch {training.current_epoch}/{training.total_epochs} · {training.progress_pct ?? 0}%
                  </span>
                )}
              </div>

              {training.status === 'RUNNING' && (
                <div className="w-full h-2 rounded-full bg-slate-100">
                  <div
                    className="h-2 rounded-full bg-blue-500 transition-all"
                    style={{ width: `${training.progress_pct ?? 0}%` }}
                  />
                </div>
              )}

              {trainingHist.length > 0 ? (
                <div className="h-48">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={trainingHist}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                      <XAxis dataKey="epoch" tick={{ fontSize: 10 }} label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fontSize: 10 }} />
                      <YAxis tick={{ fontSize: 10 }} />
                      <Tooltip />
                      <Legend wrapperStyle={{ fontSize: 10 }} />
                      <Line type="monotone" dataKey="train_loss" name="Train loss" stroke="#2563eb" dot={false} />
                      <Line type="monotone" dataKey="val_loss"   name="Val loss"   stroke="#f59e0b" dot={false} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              ) : (
                <p className="text-sm text-slate-500">
                  {training.status === 'IDLE' ? 'No training job has run yet this session.' : 'No epoch history yet.'}
                </p>
              )}

              {training.candidate_model_id && (
                <div className="rounded-lg bg-green-50 border border-green-100 px-3 py-2">
                  <p className="text-xs font-semibold text-green-800">Candidate model ready</p>
                  <p className="text-xs text-green-700 font-mono mt-0.5 truncate">{training.candidate_model_id}</p>
                </div>
              )}
            </div>
          ) : (
            <p className="text-sm text-slate-500">Training service unavailable.</p>
          )}

          {/* Session candidate models */}
          {customModels.length > 0 && (
            <div className="mt-4 border-t border-slate-100 pt-4 space-y-2">
              <p className="text-xs font-semibold text-slate-600">Session candidate models ({customModels.length})</p>
              {customModels.map(m => (
                <div key={m.run_id} className="flex items-center justify-between rounded-lg border border-slate-100 px-3 py-2">
                  <div>
                    <p className="text-xs font-semibold text-slate-800">{m.architecture}</p>
                    <p className="text-xs text-slate-500">
                      Val acc: {m.metrics?.best_val_accuracy != null ? `${m.metrics.best_val_accuracy.toFixed(1)}%` : '—'} · {m.dataset_id}
                    </p>
                  </div>
                  <span className="rounded-full bg-blue-50 px-2 py-0.5 text-xs font-semibold text-blue-700">In RAM</span>
                </div>
              ))}
            </div>
          )}
        </div>
      </section>

      {/* ── Available architectures ──────────────────────────────────── */}
      <section>
        <SectionHeader title="Available model architectures" />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {architectures.map((arch) => {
            const entry = modelEntries.find(m => m.name === arch.name);
            return (
              <div key={arch.name} className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
                <div className="flex items-start justify-between gap-2">
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-semibold text-slate-900 truncate">{arch.name}</p>
                    <p className="text-xs text-blue-600 font-medium mt-0.5">{arch.role}</p>
                    <p className="text-xs text-slate-500 mt-1">{arch.desc}</p>
                  </div>
                  {entry ? <IntegrityBadge loaded={entry.loaded} integrity={entry.integrity} /> : (
                    <span className="rounded-full bg-slate-100 px-2 py-1 text-xs text-slate-500">—</span>
                  )}
                </div>
                {arch.params !== '—' && (
                  <p className="mt-2 text-xs text-slate-400">Parameters: {arch.params}</p>
                )}
              </div>
            );
          })}
        </div>
      </section>

      {/* ── Recent experiments ───────────────────────────────────────── */}
      <section className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <SectionHeader title="Recent experiments" action={
            <div className="flex items-center gap-3">
              <button onClick={() => setActiveTab('results')} className="text-xs text-blue-600 hover:underline flex items-center gap-1">View all <ChevronRight className="h-3 w-3" /></button>
              <RefreshCw onClick={load} className="h-4 w-4 cursor-pointer text-slate-400 hover:text-slate-600" />
            </div>
          } />
          {experiments.length > 0 ? (
            <div className="space-y-2">
              {experiments.slice(0, 5).map((exp) => (
                <div key={exp.run_id} className="flex items-center justify-between rounded-lg border border-slate-100 px-3 py-2.5 hover:bg-slate-50 transition-colors">
                  <div>
                    <p className="text-sm font-semibold text-slate-800 font-mono truncate max-w-[180px]">{exp.run_id}</p>
                    <p className="text-xs text-slate-500">
                      {exp.total_traces ?? '—'} traces
                      {exp.timestamp ? ` · ${exp.timestamp}` : ''}
                    </p>
                  </div>
                  <button
                    onClick={() => setActiveTab('results')}
                    className="rounded-lg border border-slate-200 px-2 py-1 text-xs text-slate-600 hover:bg-slate-100"
                  >
                    View
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <FlaskConical className="h-8 w-8 text-slate-300 mb-2" />
              <p className="text-sm text-slate-500">No experiments yet.</p>
              <button onClick={() => setActiveTab('livedemo')} className="mt-2 text-xs text-blue-600 hover:underline">Run Live Demo to generate results</button>
            </div>
          )}
        </div>

        {/* ── Model evaluation summaries ─────────────────────────────── */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <SectionHeader title="Model evaluation summary" action={
            selected
              ? <button onClick={() => setActiveTab('results')} className="text-xs text-blue-600 hover:underline flex items-center gap-1">Full report <ChevronRight className="h-3 w-3" /></button>
              : null
          } />
          {selected && strategyRows.length > 0 ? (
            <div className="space-y-3">
              <p className="text-xs text-slate-500 font-mono">{selected.run_id} · {selected.total_traces ?? 0} traces</p>
              <div className="overflow-x-auto">
                <table className="w-full text-xs">
                  <thead>
                    <tr className="border-b border-slate-100">
                      <th className="pb-2 text-left font-semibold text-slate-600">Strategy</th>
                      <th className="pb-2 text-right font-semibold text-slate-600">Acc %</th>
                      <th className="pb-2 text-right font-semibold text-slate-600">Mean ms</th>
                      <th className="pb-2 text-right font-semibold text-slate-600">P95 ms</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-slate-50">
                    {strategyRows.map(row => (
                      <tr key={row.strategy} className="hover:bg-slate-50">
                        <td className="py-2 text-slate-700 font-medium">{row.strategy}</td>
                        <td className="py-2 text-right text-slate-800">{Number(row.accuracy).toFixed(1)}</td>
                        <td className="py-2 text-right text-slate-800">{Number(row.mean_latency_ms).toFixed(1)}</td>
                        <td className="py-2 text-right text-slate-800">{row.p95_latency_ms != null ? Number(row.p95_latency_ms).toFixed(1) : '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-8 text-center">
              <BookOpen className="h-8 w-8 text-slate-300 mb-2" />
              <p className="text-sm text-slate-500">No evaluation data yet.</p>
              <p className="text-xs text-slate-400 mt-1">Run an experiment to see strategy comparisons here.</p>
            </div>
          )}
        </div>
      </section>

      {/* ── Historical charts (only if data exists) ──────────────────── */}
      {accuracyData.length > 0 && (
        <section className="grid gap-6 lg:grid-cols-2">
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="font-bold text-slate-900 mb-4">Accuracy by strategy</h2>
            <div className="h-64"><ResponsiveContainer width="100%" height="100%">
              <BarChart data={accuracyData}><CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" /><XAxis dataKey="name" tick={{ fontSize: 9 }} /><YAxis /><Tooltip /><Bar dataKey="accuracy" fill="#2563eb" radius={[4,4,0,0]} /></BarChart>
            </ResponsiveContainer></div>
          </div>
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
            <h2 className="font-bold text-slate-900 mb-4">Latency by strategy</h2>
            <div className="h-64"><ResponsiveContainer width="100%" height="100%">
              <BarChart data={latencyData}><CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" /><XAxis dataKey="name" tick={{ fontSize: 9 }} /><YAxis unit=" ms" /><Tooltip /><Legend wrapperStyle={{ fontSize: 10 }} /><Bar dataKey="mean" name="Mean" fill="#0891b2" radius={[4,4,0,0]} /><Bar dataKey="p95" name="P95" fill="#f59e0b" radius={[4,4,0,0]} /></BarChart>
            </ResponsiveContainer></div>
          </div>
        </section>
      )}
    </div>
  );
}
