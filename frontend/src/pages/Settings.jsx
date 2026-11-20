import React, { useEffect, useState } from 'react';
import { api } from '../api';

export default function Settings() {
  const [models, setModels] = useState(null);
  const [resources, setResources] = useState(null);
  const [health, setHealth] = useState(null);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');

  const load = async () => {
    setError('');
    try {
      const [modelData, resourceData, healthData] = await Promise.all([api.getModelsStatus(), api.getResources(), api.getHealth()]);
      setModels(modelData); setResources(resourceData); setHealth(healthData);
    } catch (err) { setError(err.message); }
  };
  useEffect(()=>{load()},[]);

  const clear = async (type) => {
    if (!window.confirm(`Clear the ${type === 'ui' ? 'UI sample cache' : 'actual pipeline ContinuationCache'}?`)) return;
    try {
      const data = type === 'ui' ? await api.clearUiCache() : await api.clearContinuationCache();
      setMessage(data.cleared ? `${type === 'ui' ? 'UI cache' : 'Continuation cache'} cleared and confirmed by the backend.` : 'The backend did not confirm clearing.');
    } catch (err) { setError(err.message); }
  };

  const telemetry = resources?.telemetry || {};
  return (
    <div className="mx-auto max-w-7xl space-y-6 p-6">
      <h1 className="text-2xl font-bold text-slate-950">Settings</h1>
      {error && <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div>}
      {message && <div className="rounded-lg border border-green-200 bg-green-50 px-4 py-3 text-sm text-green-700">{message}</div>}

      <section className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="font-bold text-slate-900">Hardware</h2>
          <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
            <div><dt className="text-slate-500">Device</dt><dd className="font-semibold">{models?.device || 'Unavailable'}</dd></div>
            <div><dt className="text-slate-500">GPU</dt><dd className="font-semibold">{telemetry.gpu_name || 'Unavailable'}</dd></div>
            <div><dt className="text-slate-500">CPU</dt><dd className="font-semibold">{telemetry.cpu_percent != null ? `${telemetry.cpu_percent}%` : 'Unavailable'}</dd></div>
            <div><dt className="text-slate-500">RAM</dt><dd className="font-semibold">{telemetry.ram_percent != null ? `${telemetry.ram_percent}%` : 'Unavailable'}</dd></div>
            <div><dt className="text-slate-500">GPU VRAM allocated</dt><dd className="font-semibold">{telemetry.gpu_torch_allocated_mb != null ? `${telemetry.gpu_torch_allocated_mb} MB` : 'Unavailable'}</dd></div>
            <div><dt className="text-slate-500">GPU temperature</dt><dd className="font-semibold">{telemetry.gpu_temperature_c != null ? `${telemetry.gpu_temperature_c} °C` : 'Unavailable'}</dd></div>
          </dl>
          <button onClick={load} className="mt-4 rounded-lg border border-slate-300 px-3 py-2 text-sm font-semibold text-slate-700">Refresh hardware</button>
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="font-bold text-slate-900">Application</h2>
          <dl className="mt-4 space-y-3 text-sm"><div className="flex justify-between gap-3"><dt className="text-slate-500">Backend status</dt><dd className="font-semibold">{health?.status || 'Unavailable'}</dd></div><div className="flex justify-between gap-3"><dt className="text-slate-500">Pipeline ready</dt><dd className="font-semibold">{health?.pipeline_ready ? 'Yes' : 'No'}</dd></div><div className="flex justify-between gap-3"><dt className="text-slate-500">Frontend version</dt><dd className="font-semibold">1.0.0</dd></div><div className="flex justify-between gap-3"><dt className="text-slate-500">API documentation</dt><dd><a className="font-semibold text-blue-600" href="/docs" target="_blank" rel="noreferrer">Open Swagger</a></dd></div></dl>
        </div>
      </section>

      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <h2 className="font-bold text-slate-900">Model checkpoints</h2>
        <div className="mt-4 grid gap-3 lg:grid-cols-2">
          {Object.values(models?.models || {}).map((model)=><div key={model.name} className="rounded-lg border border-slate-200 p-4"><div className="flex items-start justify-between gap-3"><div><p className="text-sm font-semibold text-slate-900">{model.name}</p><p className="mt-1 break-all font-mono text-xs text-slate-500">{model.checkpoint_path}</p></div><span className={`rounded-full px-2 py-1 text-xs font-semibold ${model.integrity==='VERIFIED'?'bg-green-50 text-green-700':model.integrity==='MISMATCH'?'bg-amber-50 text-amber-700':'bg-red-50 text-red-700'}`}>{model.integrity}</span></div><p className="mt-3 text-xs text-slate-600">Available: {model.available?'Yes':'No'} · Loaded: {model.loaded?'Yes':'No'}</p>{model.integrity==='MISMATCH'&&<p className="mt-2 text-xs text-amber-700">Block 13 mismatch requires manual checkpoint verification. The file has not been modified.</p>}</div>)}
        </div>
      </section>

      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <h2 className="font-bold text-slate-900">Cache management</h2>
        <div className="mt-4 grid gap-4 md:grid-cols-2">
          <div className="rounded-lg border border-slate-200 p-4"><p className="text-sm font-semibold">UI sample cache</p><p className="mt-1 text-xs text-slate-500">Thumbnail gallery entries held by the backend ServiceManager.</p><button onClick={()=>clear('ui')} className="mt-3 rounded-lg border border-slate-300 px-3 py-2 text-sm font-semibold text-slate-700">Clear UI cache</button></div>
          <div className="rounded-lg border border-slate-200 p-4"><p className="text-sm font-semibold">Pipeline ContinuationCache</p><p className="mt-1 text-xs text-slate-500">Intermediate Model B feature tensors used by the real ADAPT-EDGE pipeline.</p><button onClick={()=>clear('continuation')} className="mt-3 rounded-lg border border-red-200 px-3 py-2 text-sm font-semibold text-red-700">Clear continuation cache</button></div>
        </div>
      </section>
    </div>
  );
}
