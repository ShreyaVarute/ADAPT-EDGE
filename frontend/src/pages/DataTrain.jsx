import React, { useEffect, useRef, useState } from 'react';
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { UploadCloud } from 'lucide-react';
import { api } from '../api';

const fieldClass = "mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none focus:border-blue-500";

export default function DataTrain() {
  const [datasets, setDatasets] = useState([]);
  const [selected, setSelected] = useState('cifar10');
  const [info, setInfo] = useState(null);
  const [recommendations, setRecommendations] = useState(null);
  const [telemetry, setTelemetry] = useState(null);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState('');
  const [dragging, setDragging] = useState(false);
  const [customModels, setCustomModels] = useState([]);
  const pollRef = useRef(null);
  const [config, setConfig] = useState({
    model_type: 'small', dataset_id: 'cifar10', epochs: 3, batch_size: 4,
    learning_rate: 0.001, optimizer_type: 'adam', device_selection: 'auto',
    val_split: 0.15, train_limit: 500,
  });

  const loadDatasets = async () => {
    try {
      const data = await api.getDatasets();
      setDatasets(data.datasets || []);
    } catch (err) { setError(err.message); }
  };
  const inspect = async (id) => {
    try {
      setError('');
      const data = await api.inspectDataset(id);
      setInfo(data);
      setSelected(id);
      setConfig((old) => ({ ...old, dataset_id: id }));
    } catch (err) { setError(err.message); setInfo(null); }
  };
  const refreshTraining = async () => {
    try {
      const data = await api.getTrainingStatus();
      setTelemetry(data.telemetry);
      if (['COMPLETED','STOPPED','FAILED'].includes(data.telemetry.status)) {
        api.getCustomModels().then((models)=>setCustomModels(models.models || [])).catch(()=>{});
      }
      if (!['RUNNING', 'PAUSED'].includes(data.telemetry.status)) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    } catch (err) { setError(err.message); }
  };

  useEffect(() => {
    loadDatasets();
    api.getCustomModels().then((data)=>setCustomModels(data.models || [])).catch(()=>setCustomModels([]));
    api.getTrainingRecommendations().then(setRecommendations).catch((err) => setError(err.message));
    refreshTraining();
    return () => clearInterval(pollRef.current);
  }, []);
  useEffect(() => { inspect(selected); }, [selected]);

  const upload = (file) => {
    if (!file) return;
    setError('');
    setProgress(1);
    const form = new FormData();
    form.append('archive', file);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/datasets/upload');
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) setProgress(Math.round(event.loaded / event.total * 100));
    };
    xhr.onload = async () => {
      setProgress(0);
      let body;
      try { body = JSON.parse(xhr.responseText); } catch { body = { detail: 'Invalid server response' }; }
      if (xhr.status >= 200 && xhr.status < 300 && body.success) {
        await loadDatasets();
        await inspect(body.dataset_id);
      } else {
        setError(body.detail || body.message || `Upload failed (${xhr.status})`);
      }
    };
    xhr.onerror = () => { setProgress(0); setError('Network error during upload'); };
    xhr.send(form);
  };

  const startPolling = () => {
    if (!pollRef.current) pollRef.current = setInterval(refreshTraining, 2000);
  };
  const startTraining = async () => {
    setError('');
    try {
      await api.startTraining(config);
      await refreshTraining();
      startPolling();
    } catch (err) { setError(err.message); }
  };
  const stopTraining = async () => {
    try { await api.stopTraining(); await refreshTraining(); }
    catch (err) { setError(err.message); }
  };

  const chartData = info?.class_distribution ? Object.entries(info.class_distribution).map(([name, count]) => ({ name, count })) : [];

  return (
    <div className="mx-auto max-w-7xl space-y-6 p-6">
      <h1 className="text-2xl font-bold text-slate-950">Dataset &amp; Training</h1>
      {error && <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div>}
      <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
        Candidate models and their training metrics are kept temporarily in backend memory. They are cleared when the backend restarts, so retrain candidates you want to use again.
      </div>

      <section className="grid gap-6 lg:grid-cols-[1.1fr_.9fr]">
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="font-bold text-slate-900">Dataset</h2>
          <label className="mt-4 block text-sm font-medium text-slate-700">Selected dataset
            <select className={fieldClass} value={selected} onChange={(event) => setSelected(event.target.value)}>
              {datasets.map((dataset) => <option key={dataset.id} value={dataset.id}>{dataset.name} · {dataset.status}</option>)}
            </select>
          </label>
          <div onDragOver={(event) => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); upload(event.dataTransfer.files[0]); }}
            className={`mt-4 flex min-h-36 cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed px-4 text-center ${dragging ? 'border-blue-500 bg-blue-50' : 'border-slate-300 bg-slate-50'}`}
          >
            <input id="dataset-upload" type="file" accept=".zip" className="hidden" onChange={(event) => upload(event.target.files[0])}/>
            <label htmlFor="dataset-upload" className="cursor-pointer">
              <UploadCloud className="mx-auto h-8 w-8 text-blue-500"/>
              <p className="mt-2 text-sm font-semibold text-slate-800">Drop a class-folder ZIP or choose a file</p>
              <p className="mt-1 text-xs text-slate-500">Maximum 200 MB · JPG, PNG, BMP, or WebP</p>
            </label>
          </div>
          {progress > 0 && <div className="mt-3 h-2 rounded-full bg-slate-200"><div className="h-2 rounded-full bg-blue-600 transition-all" style={{width: `${progress}%`}} /></div>}
        </div>

        {recommendations && <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="font-bold text-slate-900">Hardware recommendation</h2>
          <p className="mt-3 text-sm text-slate-600">{recommendations.hardware_advisory}</p>
          <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
            <div><dt className="text-slate-500">Device</dt><dd className="font-semibold">{recommendations.recommended_device}</dd></div>
            <div><dt className="text-slate-500">Batch size</dt><dd className="font-semibold">{recommendations.recommended_batch_size}</dd></div>
            <div><dt className="text-slate-500">Workers</dt><dd className="font-semibold">{recommendations.recommended_num_workers}</dd></div>
            <div><dt className="text-slate-500">VRAM</dt><dd className="font-semibold">{recommendations.total_vram_mb ? `${Math.round(recommendations.total_vram_mb)} MB` : 'Unavailable'}</dd></div>
          </dl>
        </div>}
      </section>

      {info && <section className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="font-bold text-slate-900">{info.name}</h2>
          <dl className="mt-4 grid grid-cols-2 gap-4 text-sm">
            <div><dt className="text-slate-500">Images</dt><dd className="text-lg font-bold">{info.total_images ?? info.total_test_images}</dd></div>
            <div><dt className="text-slate-500">Classes</dt><dd className="text-lg font-bold">{info.classes?.length ?? 0}</dd></div>
            <div><dt className="text-slate-500">Corrupted</dt><dd className="font-semibold">{info.corrupted_images ?? 0}</dd></div>
            <div><dt className="text-slate-500">Dimensions</dt><dd className="font-semibold">{Array.isArray(info.image_dimensions) ? info.image_dimensions.join(' x ') : 'Varies'}</dd></div>
          </dl>
          {chartData.length > 0 && <div className="mt-5 h-64"><ResponsiveContainer width="100%" height="100%"><BarChart data={chartData}><CartesianGrid strokeDasharray="3 3"/><XAxis dataKey="name" tick={{fontSize:10}}/><YAxis/><Tooltip/><Bar dataKey="count" fill="#2563eb"/></BarChart></ResponsiveContainer></div>}
        </div>
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <h2 className="font-bold text-slate-900">Sample preview</h2>
          <div className="mt-4 grid grid-cols-3 gap-3 sm:grid-cols-4">
            {(info.samples_preview || []).slice(0,12).map((sample, index) => {
              const src = typeof sample === 'string' ? sample : sample.image_b64;
              return src ? <img key={index} src={src} alt={`Dataset sample ${index + 1}`} className="aspect-square w-full rounded-lg border border-slate-200 object-cover"/> : null;
            })}
          </div>
        </div>
      </section>}

      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <h2 className="font-bold text-slate-900">Training configuration</h2>
        <div className="mt-4 grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <label className="text-sm font-medium">Model<select className={fieldClass} value={config.model_type} onChange={(e)=>setConfig({...config,model_type:e.target.value})}><option value="small">MobileNetV3-Small</option><option value="large">MobileNetV3-Large</option></select></label>
          <label className="text-sm font-medium">Epochs<input type="number" min="1" max="100" className={fieldClass} value={config.epochs} onChange={(e)=>setConfig({...config,epochs:Number(e.target.value)})}/></label>
          <label className="text-sm font-medium">Batch size<input type="number" min="1" max="32" className={fieldClass} value={config.batch_size} onChange={(e)=>setConfig({...config,batch_size:Number(e.target.value)})}/></label>
          <label className="text-sm font-medium">Learning rate<input type="number" step="0.0001" className={fieldClass} value={config.learning_rate} onChange={(e)=>setConfig({...config,learning_rate:Number(e.target.value)})}/></label>
          <label className="text-sm font-medium">Optimizer<select className={fieldClass} value={config.optimizer_type} onChange={(e)=>setConfig({...config,optimizer_type:e.target.value})}><option value="adam">Adam</option><option value="adamw">AdamW</option><option value="sgd">SGD</option></select></label>
          <label className="text-sm font-medium">Device<select className={fieldClass} value={config.device_selection} onChange={(e)=>setConfig({...config,device_selection:e.target.value})}><option value="auto">Auto</option><option value="cuda">CUDA</option><option value="cpu">CPU</option></select></label>
          <label className="text-sm font-medium">Validation split<input type="number" min="0.05" max="0.89" step="0.05" className={fieldClass} value={config.val_split} onChange={(e)=>setConfig({...config,val_split:Number(e.target.value)})}/></label>
          <label className="text-sm font-medium">Training sample limit<input type="number" min="20" className={fieldClass} value={config.train_limit || ''} onChange={(e)=>setConfig({...config,train_limit:e.target.value ? Number(e.target.value):null})}/></label>
        </div>
        <div className="mt-5 flex gap-3">
          <button onClick={startTraining} disabled={['RUNNING','PAUSED'].includes(telemetry?.status)} className="rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white disabled:cursor-not-allowed disabled:bg-slate-300">Start training</button>
          <button onClick={stopTraining} disabled={!['RUNNING','PAUSED'].includes(telemetry?.status)} className="rounded-lg border border-red-200 bg-white px-4 py-2 text-sm font-semibold text-red-700 disabled:cursor-not-allowed disabled:opacity-50">Stop training</button>
        </div>
      </section>

      {telemetry && <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex items-center justify-between"><h2 className="font-bold text-slate-900">Live training</h2><span className="rounded-full bg-slate-100 px-2 py-1 text-xs font-semibold">{telemetry.status}</span></div>
        <div className="mt-4 h-2 rounded-full bg-slate-200"><div className="h-2 rounded-full bg-blue-600" style={{width:`${telemetry.progress_pct || 0}%`}}/></div>
        <div className="mt-4 grid grid-cols-2 gap-3 text-sm md:grid-cols-6">
          <p><span className="text-slate-500">Epoch</span><br/><strong>{telemetry.current_epoch}/{telemetry.total_epochs}</strong></p>
          <p><span className="text-slate-500">Train loss</span><br/><strong>{telemetry.train_loss ?? '—'}</strong></p>
          <p><span className="text-slate-500">Val loss</span><br/><strong>{telemetry.val_loss ?? '—'}</strong></p>
          <p><span className="text-slate-500">Train accuracy</span><br/><strong>{telemetry.train_acc != null ? `${telemetry.train_acc}%` : '—'}</strong></p>
          <p><span className="text-slate-500">Val accuracy</span><br/><strong>{telemetry.val_acc != null ? `${telemetry.val_acc}%` : '—'}</strong></p>
          <p><span className="text-slate-500">Elapsed</span><br/><strong>{telemetry.elapsed_seconds}s</strong></p>
        </div>
        {telemetry.error_message && <p className="mt-4 text-sm text-red-700">{telemetry.error_message}</p>}
        {telemetry.candidate_model_id && <p className="mt-4 break-all text-xs text-slate-500">Session model in backend memory: {telemetry.candidate_model_id}</p>}
        {telemetry.logs?.length > 0 && <div className="mt-4 max-h-44 overflow-auto rounded-lg bg-slate-950 p-3 font-mono text-xs text-slate-100">{telemetry.logs.map((line,i)=><div key={i}>{line}</div>)}</div>}
      </section>}
      {customModels.length > 0 && <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <h2 className="font-bold text-slate-900">Current-session candidate models</h2>
        <div className="mt-4 grid gap-3 md:grid-cols-2">
          {customModels.map((model)=><div key={model.run_id} className="rounded-lg border border-slate-200 p-4">
            <p className="text-sm font-semibold text-slate-900">{model.run_id}</p>
            <p className="mt-1 text-xs text-slate-500">{model.architecture} · {model.class_names?.join(', ')}</p>
            <p className="mt-2 text-sm text-slate-700">Best validation: {model.metrics?.best_val_accuracy != null ? `${model.metrics.best_val_accuracy}%` : 'Unavailable'}</p>
          </div>)}
        </div>
      </section>}
    </div>
  );
}
