import React, { useEffect, useState } from 'react';
import { api } from '../api';

export default function LiveDemo() {
  const [samples, setSamples] = useState([]);
  const [selectedIndex, setSelectedIndex] = useState(null);
  const [preview, setPreview] = useState('');
  const [filename, setFilename] = useState('');
  const [dimensions, setDimensions] = useState(null);
  const [uploadFile, setUploadFile] = useState(null);
  const [strategy, setStrategy] = useState('ADAPT_EDGE');
  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [mode, setMode] = useState('cifar10');
  const [customModels, setCustomModels] = useState([]);
  const [customRunId, setCustomRunId] = useState('');
  const [refreshing, setRefreshing] = useState(false);

  const loadHistory = async () => {
    try { setHistory((await api.getInferenceHistory()).history || []); } catch (_) {}
  };
  useEffect(() => {
    api.getSamples(12).then((data) => setSamples(data.samples || [])).catch((err)=>setError(err.message));
    api.getCustomModels().then((data)=> {
      setCustomModels(data.models || []);
      if ((data.models || []).length) setCustomRunId(data.models[0].run_id);
    }).catch(()=>setCustomModels([]));
    loadHistory();
  }, []);

  const selectSample = (sample) => {
    setSelectedIndex(sample.dataset_index ?? sample.index);
    setPreview(sample.image_b64 || sample.image_base64);
    setFilename(`CIFAR-10 sample ${sample.dataset_index ?? sample.index}.png`);
    setDimensions([32,32]);
    setUploadFile(null);
    setResult(null);
  };
  const selectUpload = (file) => {
    if (!file) return;
    setUploadFile(file);
    setFilename(file.name);
    setSelectedIndex(null);
    setPreview(URL.createObjectURL(file));
  };

  const refreshSamples = async () => {
    setRefreshing(true); setError('');
    try {
      const data = await api.refreshSamples(12);
      setSamples(data.samples || []);
    } catch (err) { setError(err.message); }
    finally { setRefreshing(false); }
  };

  const run = async () => {
    setLoading(true); setError('');
    try {
      let data;
      if (uploadFile) {
        const form = new FormData();
        form.append('image', uploadFile);
        data = await api.uploadAndPredict(form, {
          modelMode: mode,
          customRunId: mode === 'custom' ? customRunId : undefined,
          strategy,
        });
      } else {
        data = await api.predict({ sample_index: selectedIndex ?? 0, strategy: strategy === 'ALL' ? 'ALL' : strategy });
      }
      setResult(data);
      await loadHistory();
    } catch (err) { setError(err.message); }
    finally { setLoading(false); }
  };

  const clear = () => {
    setUploadFile(null); setSelectedIndex(null); setPreview(''); setFilename(''); setDimensions(null); setResult(null); setError('');
  };

  const adapt = result?.adapt_edge_cached || result?.primary;
  const trace = adapt?.trace;
  const action = adapt?.decision || 'Unavailable';

  return (
    <div className="mx-auto max-w-7xl space-y-6 p-6">
      <h1 className="text-2xl font-bold text-slate-950">Live Demo</h1>
      {error && <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div>}
      <section className="grid gap-6 lg:grid-cols-[380px_1fr]">
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="font-bold text-slate-900">Input</h2>
          <div className="mt-4 rounded-lg border border-slate-200 p-3">
            <label className="block text-xs font-semibold uppercase text-slate-500">Model mode</label>
            <select value={mode} onChange={(event)=>setMode(event.target.value)} className="mt-2 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm">
              <option value="cifar10">CIFAR-10 research pipeline</option>
              <option value="custom" disabled={customModels.length===0}>Custom-trained model</option>
            </select>
            {mode === 'cifar10' && <p className="mt-2 text-xs text-slate-500">Closed-set CIFAR-10 classifier. External photographs may receive unreliable predictions.</p>}
            {mode === 'custom' && (
              <select value={customRunId} onChange={(event)=>setCustomRunId(event.target.value)} className="mt-2 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm">
                {customModels.map((model)=><option key={model.run_id} value={model.run_id}>{model.run_id} · {model.architecture}</option>)}
              </select>
            )}
          </div>
          <label className="mt-4 flex h-28 cursor-pointer items-center justify-center rounded-xl border-2 border-dashed border-slate-300 bg-slate-50 text-center text-sm text-slate-600">
            <input type="file" accept=".jpg,.jpeg,.png,.bmp,.webp" className="hidden" onChange={(e)=>selectUpload(e.target.files[0])}/>
            Upload or drop an image
          </label>
          {preview && <img src={preview} alt="Inference input" className="mt-4 max-h-64 w-full rounded-lg border border-slate-200 object-contain"/>}
          {filename && <div className="mt-3 text-xs text-slate-500 break-all"><strong className="text-slate-700">{filename}</strong>{dimensions && ` · ${dimensions[0]} x ${dimensions[1]}`}</div>}
          <div className="mt-5 flex items-center justify-between">
            <h3 className="text-sm font-semibold text-slate-800">CIFAR-10 samples</h3>
            <button onClick={refreshSamples} disabled={refreshing} className="rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-semibold text-slate-700 disabled:opacity-50">{refreshing?'Refreshing...':'Refresh Samples'}</button>
          </div>
          <div className="mt-3 grid grid-cols-4 gap-2 max-h-52 overflow-auto">
            {samples.map((sample)=><button key={sample.index} onClick={()=>selectSample(sample)} className={`overflow-hidden rounded-lg border ${selectedIndex === sample.index ? 'border-blue-600 ring-2 ring-blue-100' : 'border-slate-200'}`}><img src={sample.image_b64} alt={sample.class_name} className="aspect-square w-full object-cover"/></button>)}
          </div>
          {mode === 'cifar10' && <label className="mt-5 block text-sm font-medium text-slate-700">Strategy
            <select value={strategy} onChange={(e)=>setStrategy(e.target.value)} className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-2 text-sm">
              <option value="ADAPT_EDGE">ADAPT-EDGE</option><option value="MODEL_A">Model A only</option><option value="MODEL_B">Model B only</option><option value="CONVENTIONAL">Conventional switching</option><option value="ALL">Comparison mode</option>
            </select>
          </label>}
          <div className="mt-4 flex gap-3">
            <button onClick={run} disabled={loading || (!preview)} className="flex-1 rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white disabled:bg-slate-300">{loading?'Running...':'Run inference'}</button>
            <button onClick={clear} className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-semibold text-slate-700">Clear</button>
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="font-bold text-slate-900">Inference result</h2>
          {!result ? <div className="mt-5 rounded-lg border border-slate-200 bg-slate-50 p-8 text-center text-sm text-slate-500">Select an input and run the real pipeline.</div> : (
            <div className="mt-4 space-y-4">
              {result.model_mode === 'custom' && adapt && <div className="rounded-lg border border-blue-200 bg-blue-50 p-4">
                <p className="text-xs font-semibold uppercase text-blue-700">Custom-trained model</p>
                <p className="mt-2 text-2xl font-bold text-slate-950">{adapt.prediction_class}</p>
                <p className="text-sm text-slate-600">Confidence: {(adapt.confidence*100).toFixed(1)}% · {adapt.latency_ms?.toFixed(2)} ms</p>
                <p className="mt-2 text-xs text-slate-500">This is single-model custom inference, not the ADAPT-EDGE RETAIN/PROBE/SWITCH pipeline.</p>
              </div>}
              {result.ood && <div className={`rounded-lg border p-4 ${result.ood.status==='POTENTIAL_OOD'?'border-amber-200 bg-amber-50 text-amber-800':'border-slate-200 bg-slate-50 text-slate-700'}`}>
                <p className="text-xs font-bold uppercase">OOD awareness: {result.ood.status}</p>
                <p className="mt-1 text-sm">{result.ood.message}</p>
              </div>}
              {strategy === 'ALL' && result.comparison && <div className="grid gap-3 md:grid-cols-2">{result.comparison.map((item)=><div key={item.strategy_key} className="rounded-lg border border-slate-200 p-4"><p className="text-xs font-semibold uppercase text-slate-500">{item.strategy_name}</p><p className="mt-2 text-lg font-bold text-slate-900">{item.prediction_class}</p><p className="text-sm text-slate-600">Confidence {item.confidence != null ? `${(item.confidence*100).toFixed(1)}%` : 'Unavailable'} · {item.latency_ms?.toFixed(2)} ms</p><p className="mt-2 text-xs text-slate-500">Accuracy: {result.true_class ? `${item.prediction_class === result.true_class ? 'Correct' : 'Incorrect'} (${result.true_class})` : 'Unavailable'}</p></div>)}</div>}
              {adapt && <div className="grid gap-4 md:grid-cols-2">
                <div className="rounded-lg border border-slate-200 p-4"><p className="text-xs font-semibold uppercase text-slate-500">Initial Model A</p><p className="mt-2 text-lg font-bold text-slate-900">{trace?.model_a_pred_class || 'Unavailable'}</p><p className="text-sm text-slate-600">{trace?.model_a_conf != null ? `${(trace.model_a_conf*100).toFixed(1)}% confidence` : 'Unavailable'}</p></div>
                <div className="rounded-lg border border-slate-200 p-4"><p className="text-xs font-semibold uppercase text-slate-500">Controller action</p><p className="mt-2 text-lg font-bold text-blue-700">{action}</p><p className="mt-1 text-sm text-slate-600">{trace?.decision_reason || adapt.status || 'No reason returned by pipeline.'}</p></div>
              </div>}
              {adapt && <div className="rounded-lg border border-slate-200 p-4">
                <p className="text-xs font-semibold uppercase text-slate-500">Final result</p>
                <div className="mt-2 flex flex-wrap items-end justify-between gap-3"><div><p className="text-2xl font-bold text-slate-950">{adapt.prediction_class}</p><p className="text-sm text-slate-600">{adapt.confidence != null ? `${(adapt.confidence*100).toFixed(1)}% confidence` : 'Confidence unavailable'}</p></div><p className="font-mono text-sm text-slate-700">{adapt.latency_ms?.toFixed(2)} ms</p></div>
                {trace?.probes_executed?.length > 0 && <p className="mt-3 text-sm text-slate-600">Executed probe blocks: {trace.probes_executed.map((probe)=>probe.block_id).join(', ')}</p>}
                {adapt.status && <p className="mt-2 text-xs text-slate-500">Status: {adapt.status} · Cache: {adapt.cache_status}</p>}
              </div>}
            </div>
          )}
        </div>
      </section>

      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <h2 className="font-bold text-slate-900">Current-session history</h2>
        {history.length === 0 ? <p className="mt-3 text-sm text-slate-500">No inference events yet.</p> : <div className="mt-4 overflow-x-auto"><table className="w-full text-left text-sm"><thead className="border-b text-xs uppercase text-slate-500"><tr><th className="py-2">Time</th><th>Image</th><th>Mode</th><th>Model</th><th>Prediction</th><th>Confidence</th><th>Action</th><th>OOD</th><th>Latency</th></tr></thead><tbody>{history.map((row,i)=><tr key={i} className="border-b border-slate-100"><td className="py-2 text-slate-500">{row.timestamp}</td><td>{row.image_name}</td><td>{row.model_mode || 'cifar10'}</td><td>{row.model_name || '—'}</td><td className="font-semibold">{row.prediction}</td><td>{row.confidence != null ? `${(row.confidence*100).toFixed(1)}%` : '—'}</td><td>{row.action || '—'}</td><td>{row.ood_status || '—'}</td><td>{row.latency_ms != null ? `${row.latency_ms.toFixed?.(2) ?? row.latency_ms} ms` : 'Unavailable'}</td></tr>)}</tbody></table></div>}
      </section>
    </div>
  );
}
