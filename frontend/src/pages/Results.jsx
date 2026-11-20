import React, { useEffect, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Line, LineChart, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { api } from '../api';

export default function Results() {
  const [runs, setRuns] = useState([]);
  const [sessionModels, setSessionModels] = useState([]);
  const [sessionModelId, setSessionModelId] = useState('');
  const [sessionEvaluations, setSessionEvaluations] = useState([]);
  const [selected, setSelected] = useState('');
  const [detail, setDetail] = useState(null);
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getExperiments().then(async (data) => {
      const list = data.experiments || [];
      setRuns(list);
      if (list.length) {
        setSelected(list[0].run_id);
        setDetail(await api.getExperimentDetails(list[0].run_id));
      }
    }).catch((err)=>setError(err.message)).finally(()=>setLoading(false));
    api.getCustomModels().then((data) => {
      const models = data.models || [];
      setSessionModels(models);
      if (models.length) setSessionModelId(models[0].run_id);
    }).catch((err)=>setError(err.message));
    api.getInferenceHistory().then((data) => {
      setSessionEvaluations((data.history || []).filter((row) => row.model_mode === 'custom'));
    }).catch((err)=>setError(err.message));
  }, []);

  const choose = async (runId) => {
    setSelected(runId); setError('');
    try { setDetail(await api.getExperimentDetails(runId)); }
    catch (err) { setError(err.message); setDetail(null); }
  };
  const accuracy = (detail?.strategies || []).map((row)=>({name:row.strategy, value:Number(row.accuracy)}));
  const latency = (detail?.strategies || []).map((row)=>({name:row.strategy, mean:row.mean_latency_ms == null?null:Number(row.mean_latency_ms), p95:row.p95_latency_ms==null?null:Number(row.p95_latency_ms)}));
  const selectedSessionModel = sessionModels.find((model) => model.run_id === sessionModelId);
  const sessionAccuracy = sessionModels
    .filter((model) => model.metrics?.best_val_accuracy != null)
    .map((model) => ({ name: model.run_id, accuracy: Number(model.metrics.best_val_accuracy) }));

  return (
    <div className="mx-auto max-w-7xl space-y-6 p-6">
      <div className="flex flex-wrap items-end justify-between gap-3"><div><h1 className="text-2xl font-bold text-slate-950">Results &amp; Comparison</h1><p className="mt-1 text-sm text-slate-500">Historical experiments and current-session candidate results</p></div>
        <div className="flex gap-2"><a href={`/api/experiments/results/export/${encodeURIComponent(selected)}?format=csv`} className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-700">Export CSV</a><a href={`/api/experiments/results/export/${encodeURIComponent(selected)}?format=json`} className="rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm font-semibold text-slate-700">Export JSON</a></div>
      </div>
      {error && <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div>}
      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <h2 className="font-bold text-slate-900">Current-session candidate results</h2>
            <p className="mt-1 text-sm text-slate-500">These models, training curves, and custom inference results remain available only until the backend restarts.</p>
          </div>
          {sessionModels.length > 0 && <label className="text-sm font-medium text-slate-700">Training curve
            <select value={sessionModelId} onChange={(event)=>setSessionModelId(event.target.value)} className="ml-2 rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm">
              {sessionModels.map((model)=><option key={model.run_id} value={model.run_id}>{model.run_id}</option>)}
            </select>
          </label>}
        </div>
        {sessionModels.length === 0 ? <p className="mt-4 text-sm text-slate-500">No candidate models have been trained in this backend session.</p> : <>
          <div className="mt-5 grid gap-4 lg:grid-cols-2">
            <div className="rounded-lg border border-slate-200 p-4">
              <h3 className="text-sm font-semibold text-slate-800">Candidate validation accuracy comparison</h3>
              {sessionAccuracy.length > 0 ? <div className="mt-3 h-64"><ResponsiveContainer width="100%" height="100%"><BarChart data={sessionAccuracy}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="name" tick={{fontSize:8}}/><YAxis unit="%"/><Tooltip/><Bar dataKey="accuracy" name="Best validation accuracy" fill="#2563eb"/></BarChart></ResponsiveContainer></div> : <p className="mt-3 text-sm text-slate-500">Validation metrics are not available yet.</p>}
            </div>
            <div className="rounded-lg border border-slate-200 p-4">
              <h3 className="text-sm font-semibold text-slate-800">Training and validation loss</h3>
              {selectedSessionModel?.history?.length > 0 ? <div className="mt-3 h-64"><ResponsiveContainer width="100%" height="100%"><LineChart data={selectedSessionModel.history}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="epoch"/><YAxis/><Tooltip/><Legend/><Line type="monotone" dataKey="train_loss" name="Train loss" stroke="#2563eb"/><Line type="monotone" dataKey="val_loss" name="Validation loss" stroke="#f59e0b"/></LineChart></ResponsiveContainer></div> : <p className="mt-3 text-sm text-slate-500">Training metrics are not available yet.</p>}
            </div>
          </div>
          <div className="mt-5 overflow-x-auto">
            <h3 className="font-semibold text-slate-900">Custom-model evaluation results</h3>
            {sessionEvaluations.length === 0 ? <p className="mt-2 text-sm text-slate-500">Run custom-model inference in Live Demo to add results for this session.</p> : <table className="mt-3 w-full text-left text-sm"><thead className="border-b text-xs uppercase text-slate-500"><tr><th className="py-2">Time</th><th>Model</th><th>Image</th><th>Prediction</th><th>Confidence</th><th>Latency</th></tr></thead><tbody>{sessionEvaluations.map((row,index)=><tr key={`${row.timestamp}-${index}`} className="border-b border-slate-100"><td className="py-2 text-slate-500">{row.timestamp}</td><td>{row.model_name}</td><td>{row.image_name}</td><td className="font-semibold">{row.prediction}</td><td>{row.confidence != null ? `${(row.confidence*100).toFixed(1)}%` : '—'}</td><td>{row.latency_ms != null ? `${Number(row.latency_ms).toFixed(2)} ms` : '—'}</td></tr>)}</tbody></table>}
          </div>
        </>}
      </section>
      {loading ? <p className="text-sm text-slate-500">Loading experiment history...</p> : runs.length === 0 ? <div className="rounded-xl border border-slate-200 bg-white p-8 text-center text-sm text-slate-500">No historical experiment runs with readable metrics were found.</div> : (
        <>
          <label className="block max-w-lg text-sm font-medium text-slate-700">Experiment run
            <select value={selected} onChange={(e)=>choose(e.target.value)} className="mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm"><option value="">Select a run</option>{runs.map((run)=><option key={run.run_id} value={run.run_id}>{run.run_id}</option>)}</select>
          </label>
          {detail && <>
            <div className="grid gap-4 lg:grid-cols-2">
              <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm"><h2 className="font-bold">Accuracy comparison</h2><div className="mt-4 h-72"><ResponsiveContainer width="100%" height="100%"><BarChart data={accuracy}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="name" tick={{fontSize:9}}/><YAxis/><Tooltip/><Bar dataKey="value" name="Accuracy" fill="#2563eb"/></BarChart></ResponsiveContainer></div></div>
              <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm"><h2 className="font-bold">Latency comparison</h2><div className="mt-4 h-72"><ResponsiveContainer width="100%" height="100%"><BarChart data={latency}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="name" tick={{fontSize:9}}/><YAxis unit=" ms"/><Tooltip/><Bar dataKey="mean" name="Mean" fill="#0891b2"/><Bar dataKey="p95" name="P95" fill="#f59e0b"/></BarChart></ResponsiveContainer></div></div>
            </div>
            <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
              <h2 className="font-bold">Strategy metrics</h2>
              <div className="mt-4 overflow-x-auto"><table className="w-full text-left text-sm"><thead className="border-b text-xs uppercase text-slate-500"><tr><th className="py-2">Strategy</th><th>Accuracy</th><th>Macro F1</th><th>Mean</th><th>Median</th><th>P95</th><th>P99</th><th>Full B</th></tr></thead><tbody>{(detail.strategies||[]).map((row,i)=><tr key={i} className="border-b border-slate-100"><td className="py-3 font-semibold">{row.strategy}</td><td>{row.accuracy ?? 'N/A'}</td><td>{row.macro_f1 ?? 'N/A'}</td><td>{row.mean_latency_ms != null ? `${row.mean_latency_ms} ms`:'N/A'}</td><td>{row.median_latency_ms != null ? `${row.median_latency_ms} ms`:'N/A'}</td><td>{row.p95_latency_ms != null ? `${row.p95_latency_ms} ms`:'N/A'}</td><td>{row.p99_latency_ms != null ? `${row.p99_latency_ms} ms`:'N/A'}</td><td>{row.full_model_b_pct != null ? `${row.full_model_b_pct}%`:'N/A'}</td></tr>)}</tbody></table></div>
            </div>
            {detail.cache_metrics && Object.keys(detail.cache_metrics).length>0 && <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm"><h2 className="font-bold">Cache metrics</h2><pre className="mt-3 max-h-72 overflow-auto rounded-lg bg-slate-950 p-4 text-xs text-slate-100">{JSON.stringify(detail.cache_metrics,null,2)}</pre></div>}
          </>}
        </>
      )}
    </div>
  );
}
