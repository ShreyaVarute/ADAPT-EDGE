import React, { useEffect, useState } from 'react';
import {
  Zap, Sliders, ShieldAlert, CheckCircle2, AlertTriangle,
  Clock, Gauge, Cpu, RefreshCw, Upload, Image as ImageIcon,
  Layers, ArrowRight, Activity, Filter, Info
} from 'lucide-react';
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
  const [refreshing, setRefreshing] = useState(false);

  // Model A / Model B Pair Configuration State
  const [pairConfig, setPairConfig] = useState({
    model_a_id: 'mobilenet_v3_small_baseline',
    model_b_id: 'mobilenet_v3_large_baseline',
    threshold_a: 0.85,
    threshold_probe: 0.80,
    latency_budget_ms: 25.0,
    enforce_resource_budget: true,
  });
  const [activePair, setActivePair] = useState(null);
  const [availableCandidates, setAvailableCandidates] = useState([]);
  const [pairSavedMsg, setPairSavedMsg] = useState('');
  const [savingPair, setSavingPair] = useState(false);

  const loadHistory = async () => {
    try {
      const data = await api.getInferenceHistory();
      setHistory(data.history || []);
    } catch (_) {}
  };

  const loadPair = async () => {
    try {
      const data = await api.getModelPair();
      if (data.active_pair) {
        setActivePair(data.active_pair);
        setPairConfig({
          model_a_id: data.active_pair.model_a_id,
          model_b_id: data.active_pair.model_b_id,
          threshold_a: data.active_pair.threshold_a,
          threshold_probe: data.active_pair.threshold_probe,
          latency_budget_ms: data.active_pair.latency_budget_ms,
          enforce_resource_budget: data.active_pair.enforce_resource_budget,
        });
      }
      setAvailableCandidates(data.available_candidates || []);
    } catch (e) {
      console.warn('Failed to load active pair:', e);
    }
  };

  useEffect(() => {
    api.getSamples(12).then((data) => setSamples(data.samples || [])).catch((err) => setError(err.message));
    loadPair();
    loadHistory();
  }, []);

  const savePairConfig = async () => {
    setSavingPair(true);
    setPairSavedMsg('');
    setError('');
    try {
      const res = await api.configureModelPair(pairConfig);
      if (res.active_pair) {
        setActivePair(res.active_pair);
        setPairSavedMsg('Model pairing & thresholds updated successfully!');
        setTimeout(() => setPairSavedMsg(''), 4000);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setSavingPair(false);
    }
  };

  const selectSample = (sample) => {
    setSelectedIndex(sample.dataset_index ?? sample.index);
    setPreview(sample.image_b64 || sample.image_base64);
    setFilename(`CIFAR-10 sample ${sample.dataset_index ?? sample.index}.png`);
    setDimensions([32, 32]);
    setUploadFile(null);
    setResult(null);
  };

  const selectUpload = (file) => {
    if (!file) return;
    setUploadFile(file);
    setFilename(file.name);
    setSelectedIndex(null);
    setPreview(URL.createObjectURL(file));
    setResult(null);
  };

  const refreshSamples = async () => {
    setRefreshing(true);
    setError('');
    try {
      const data = await api.refreshSamples(12);
      setSamples(data.samples || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setRefreshing(false);
    }
  };

  const run = async () => {
    setLoading(true);
    setError('');
    try {
      let data;
      if (uploadFile) {
        const form = new FormData();
        form.append('image', uploadFile);
        data = await api.uploadAndPredict(form, {
          strategy,
        });
      } else {
        data = await api.predict({
          sample_index: selectedIndex ?? 0,
          strategy: strategy === 'ALL' ? 'ALL' : strategy,
          threshold_a: pairConfig.threshold_a,
          threshold_probe: pairConfig.threshold_probe,
          latency_budget_ms: pairConfig.latency_budget_ms,
          enforce_resource_budget: pairConfig.enforce_resource_budget,
        });
      }
      setResult(data);
      await loadHistory();
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const clear = () => {
    setUploadFile(null);
    setSelectedIndex(null);
    setPreview('');
    setFilename('');
    setDimensions(null);
    setResult(null);
    setError('');
  };

  const adapt = result?.adapt_edge_cached || result?.primary;
  const trace = adapt?.trace;
  const action = adapt?.decision || 'Unavailable';

  const probesCompatible = activePair?.probes_compatible ?? true;

  return (
    <div className="mx-auto max-w-7xl space-y-6 p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold text-slate-950">Live Adaptive Demo &amp; Model Pairing</h1>
          <p className="mt-1 text-sm text-slate-500">
            Configure Model A, Model B, confidence thresholds, and test real ADAPT-EDGE inference
          </p>
        </div>
      </div>

      {error && <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</div>}
      {pairSavedMsg && <div className="rounded-lg border border-green-200 bg-green-50 px-4 py-3 text-sm text-green-700">{pairSavedMsg}</div>}

      {/* ── SECTION 1: MODEL A / MODEL B PAIR CONFIGURATION WORKBENCH ── */}
      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2">
            <Sliders className="w-5 h-5 text-blue-600" />
            <div>
              <h2 className="font-bold text-sm text-slate-900">Select Model A &amp; Model B Pair</h2>
              <p className="text-xs text-slate-500">
                Choose lightweight Model A and larger Model B. Configure confidence &amp; resource thresholds.
              </p>
            </div>
          </div>
          <button
            onClick={savePairConfig}
            disabled={savingPair}
            className="rounded-lg bg-blue-600 px-4 py-1.5 text-xs font-semibold text-white hover:bg-blue-700 disabled:opacity-50 shadow-xs"
          >
            {savingPair ? 'Saving...' : 'Apply Pair Settings'}
          </button>
        </div>

        {/* Selected Pair Display Banner */}
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4 bg-slate-50 border border-slate-200 rounded-xl p-4">
          <div>
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Active Model A (Lightweight)</span>
            <p className="text-sm font-bold text-slate-900 mt-0.5">{activePair?.model_a_name || 'MobileNetV3-Small'}</p>
            <span className="text-[11px] font-mono text-slate-500">Threshold: {pairConfig.threshold_a}</span>
          </div>

          <div>
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Active Model B (Larger)</span>
            <p className="text-sm font-bold text-slate-900 mt-0.5">{activePair?.model_b_name || 'MobileNetV3-Large'}</p>
            <span className="text-[11px] font-mono text-slate-500">Threshold: {pairConfig.threshold_probe}</span>
          </div>

          <div>
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Intermediate Probing Status</span>
            <div className="mt-1 flex items-center gap-1.5">
              {probesCompatible ? (
                <span className="inline-flex items-center gap-1 text-xs font-semibold text-green-700 bg-green-100/70 px-2.5 py-0.5 rounded-full">
                  <CheckCircle2 className="w-3.5 h-3.5" /> PROBE Enabled (Blocks 4, 9, 13)
                </span>
              ) : (
                <span className="inline-flex items-center gap-1 text-xs font-semibold text-amber-800 bg-amber-100/80 px-2.5 py-0.5 rounded-full" title="Existing probe checkpoints are calibrated for MobileNetV3-Large only">
                  <ShieldAlert className="w-3.5 h-3.5" /> PROBE Disabled (Architecture Incompatible)
                </span>
              )}
            </div>
          </div>

          <div>
            <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-500">Budgets Enforced</span>
            <p className="text-xs font-mono font-medium text-slate-800 mt-1">
              Latency: {pairConfig.latency_budget_ms} ms · Hardware Budget: {pairConfig.enforce_resource_budget ? 'Active' : 'Off'}
            </p>
          </div>
        </div>

        {/* Configuration Selectors & Sliders */}
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4 pt-2">
          {/* Model A Selector */}
          <div>
            <label className="block text-xs font-semibold text-slate-700">
              Model A (Lightweight Candidate)
            </label>
            <select
              value={pairConfig.model_a_id}
              onChange={(e) => setPairConfig({ ...pairConfig, model_a_id: e.target.value })}
              className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-800 outline-none focus:border-blue-500"
            >
              <option value="mobilenet_v3_small_baseline">MobileNetV3-Small (Audited Baseline)</option>
              {availableCandidates.map((c) => (
                <option key={c.id} value={c.id}>{c.name}</option>
              ))}
            </select>
            <p className="text-[10px] text-slate-400 mt-1">Fast primary inference model</p>
          </div>

          {/* Model B Selector */}
          <div>
            <label className="block text-xs font-semibold text-slate-700">
              Model B (Larger Candidate)
            </label>
            <select
              value={pairConfig.model_b_id}
              onChange={(e) => setPairConfig({ ...pairConfig, model_b_id: e.target.value })}
              className="mt-1.5 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-800 outline-none focus:border-blue-500"
            >
              <option value="mobilenet_v3_large_baseline">MobileNetV3-Large (Audited Baseline)</option>
              {availableCandidates.map((c) => (
                <option key={c.id} value={c.id}>{c.name}</option>
              ))}
            </select>
            <p className="text-[10px] text-slate-400 mt-1">Full model fallback for difficult inputs</p>
          </div>

          {/* Model A Threshold */}
          <div>
            <div className="flex justify-between text-xs font-semibold text-slate-700">
              <span>Model A Retain Threshold (τ_A)</span>
              <span className="font-mono text-blue-600">{pairConfig.threshold_a}</span>
            </div>
            <input
              type="range"
              min="0.50"
              max="0.99"
              step="0.01"
              value={pairConfig.threshold_a}
              onChange={(e) => setPairConfig({ ...pairConfig, threshold_a: parseFloat(e.target.value) })}
              className="w-full mt-2 accent-blue-600"
            />
            <p className="text-[10px] text-slate-400 mt-1">If Model A confidence ≥ τ_A → RETAIN</p>
          </div>

          {/* Probe Threshold */}
          <div>
            <div className="flex justify-between text-xs font-semibold text-slate-700">
              <span>Probe Retain Threshold (τ_probe)</span>
              <span className="font-mono text-blue-600">{pairConfig.threshold_probe}</span>
            </div>
            <input
              type="range"
              min="0.50"
              max="0.99"
              step="0.01"
              value={pairConfig.threshold_probe}
              onChange={(e) => setPairConfig({ ...pairConfig, threshold_probe: parseFloat(e.target.value) })}
              className="w-full mt-2 accent-blue-600"
            />
            <p className="text-[10px] text-slate-400 mt-1">If probe confidence ≥ τ_probe → RETAIN</p>
          </div>
        </div>

        {/* Latency & Resource Budget Options */}
        <div className="grid gap-4 md:grid-cols-2 pt-2 border-t border-slate-100">
          <div>
            <div className="flex justify-between text-xs font-semibold text-slate-700">
              <span>Max Latency Budget (ms)</span>
              <span className="font-mono text-blue-600">{pairConfig.latency_budget_ms} ms</span>
            </div>
            <input
              type="range"
              min="5.0"
              max="50.0"
              step="1.0"
              value={pairConfig.latency_budget_ms}
              onChange={(e) => setPairConfig({ ...pairConfig, latency_budget_ms: parseFloat(e.target.value) })}
              className="w-full mt-2 accent-blue-600"
            />
          </div>

          <div className="flex items-center gap-3 pt-3">
            <input
              type="checkbox"
              id="enforce_res"
              checked={pairConfig.enforce_resource_budget}
              onChange={(e) => setPairConfig({ ...pairConfig, enforce_resource_budget: e.target.checked })}
              className="rounded text-blue-600 accent-blue-600"
            />
            <label htmlFor="enforce_res" className="text-xs text-slate-700 cursor-pointer">
              <strong>Enforce Hardware Resource Guardrails</strong> (auto-reverts to RETAIN if memory/thermal thresholds are exceeded)
            </label>
          </div>
        </div>
      </section>

      {/* ── SECTION 2: LIVE INFERENCE PIPELINE RUNNER ── */}
      <section className="grid gap-6 lg:grid-cols-[380px_1fr]">
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-4">
          <h2 className="font-bold text-slate-900">Inference Input</h2>

          <label className="flex h-24 cursor-pointer items-center justify-center rounded-xl border-2 border-dashed border-slate-300 bg-slate-50 text-center text-xs text-slate-600 hover:bg-slate-100">
            <input type="file" accept=".jpg,.jpeg,.png,.bmp,.webp" className="hidden" onChange={(e) => selectUpload(e.target.files[0])} />
            <div className="flex flex-col items-center gap-1">
              <Upload className="w-4 h-4 text-slate-400" />
              <span>Upload or drop image</span>
            </div>
          </label>

          {preview && (
            <div className="relative rounded-lg overflow-hidden border border-slate-200 bg-black/5">
              <img src={preview} alt="Inference input" className="max-h-52 w-full object-contain mx-auto" />
              {filename && (
                <div className="p-2 text-center text-[10px] text-slate-500 truncate bg-white/90 border-t border-slate-100">
                  {filename} {dimensions && `(${dimensions[0]}×${dimensions[1]})`}
                </div>
              )}
            </div>
          )}

          <div>
            <div className="flex items-center justify-between mb-2">
              <h3 className="text-xs font-semibold text-slate-700">Or Select CIFAR-10 Sample</h3>
              <button
                onClick={refreshSamples}
                disabled={refreshing}
                className="rounded border border-slate-300 px-2 py-0.5 text-[11px] font-semibold text-slate-600 disabled:opacity-50"
              >
                {refreshing ? 'Refreshing...' : 'Refresh'}
              </button>
            </div>
            <div className="grid grid-cols-4 gap-2 max-h-40 overflow-auto p-1 bg-slate-50 rounded-lg border border-slate-200">
              {samples.map((sample) => (
                <button
                  key={sample.index}
                  onClick={() => selectSample(sample)}
                  className={`overflow-hidden rounded-md border transition-all ${
                    selectedIndex === sample.index ? 'border-blue-600 ring-2 ring-blue-300' : 'border-slate-200 hover:border-slate-400'
                  }`}
                >
                  <img src={sample.image_b64} alt={sample.class_name} className="aspect-square w-full object-cover" />
                </button>
              ))}
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-700">Execution Strategy</label>
            <select
              value={strategy}
              onChange={(e) => setStrategy(e.target.value)}
              className="mt-1 w-full rounded-lg border border-slate-300 px-3 py-1.5 text-xs font-medium text-slate-800"
            >
              <option value="ADAPT_EDGE">ADAPT-EDGE (Decision-Selective with Cache)</option>
              <option value="MODEL_A">Model A Only</option>
              <option value="MODEL_B">Model B Only</option>
              <option value="CONVENTIONAL">Conventional Hard Switching</option>
              <option value="ALL">Comparison Mode (Run All 4)</option>
            </select>
          </div>

          <div className="flex gap-2 pt-2">
            <button
              onClick={run}
              disabled={loading || !preview}
              className="flex-1 rounded-lg bg-blue-600 px-4 py-2 text-xs font-semibold text-white hover:bg-blue-700 disabled:bg-slate-300 flex items-center justify-center gap-1.5 shadow-xs"
            >
              <Zap className="w-3.5 h-3.5" />
              {loading ? 'Executing...' : 'Run Pipeline'}
            </button>
            <button onClick={clear} className="rounded-lg border border-slate-300 px-3 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-50">
              Clear
            </button>
          </div>
        </div>

        {/* Inference Decision Output Display */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-4">
          <h2 className="font-bold text-slate-900">Adaptive Decision Trace</h2>

          {!result ? (
            <div className="rounded-lg border border-slate-200 bg-slate-50 py-16 text-center text-xs text-slate-500">
              Select an image or sample and click <strong>"Run Pipeline"</strong> to execute live model switching.
            </div>
          ) : (
            <div className="space-y-4">
              {/* Decision Action Banner */}
              {adapt && (
                <div className={`p-4 rounded-xl border flex items-center justify-between ${
                  action === 'RETAIN' ? 'bg-emerald-50 border-emerald-200 text-emerald-950' :
                  action === 'PROBE' ? 'bg-blue-50 border-blue-200 text-blue-950' :
                  'bg-indigo-50 border-indigo-200 text-indigo-950'
                }`}>
                  <div>
                    <span className="text-[10px] font-bold uppercase tracking-wider opacity-75">Decision Action</span>
                    <p className="text-2xl font-black">{action}</p>
                    <p className="text-xs opacity-90 mt-0.5">{trace?.decision_reason || adapt.status}</p>
                  </div>

                  <div className="text-right">
                    <span className="text-[10px] uppercase font-semibold opacity-75">Final Predicted Class</span>
                    <p className="text-xl font-bold font-mono">{adapt.prediction_class}</p>
                    <p className="text-xs font-mono opacity-80">
                      {adapt.confidence != null ? `${(adapt.confidence * 100).toFixed(1)}% conf` : ''} · {adapt.latency_ms?.toFixed(2)} ms
                    </p>
                  </div>
                </div>
              )}

              {/* Step-by-Step Decision Stages */}
              {adapt && (
                <div className="grid gap-3 md:grid-cols-3">
                  <div className="rounded-lg border border-slate-200 p-3 bg-slate-50/50">
                    <span className="text-[10px] font-semibold text-slate-400 uppercase">Step 1: Model A</span>
                    <p className="text-sm font-bold text-slate-900 mt-1">{trace?.model_a_pred_class || 'Executed'}</p>
                    <p className="text-xs text-slate-500">{trace?.model_a_conf != null ? `${(trace.model_a_conf * 100).toFixed(1)}% conf` : '—'}</p>
                  </div>

                  <div className="rounded-lg border border-slate-200 p-3 bg-slate-50/50">
                    <span className="text-[10px] font-semibold text-slate-400 uppercase">Step 2: Probing</span>
                    <p className="text-sm font-bold text-slate-900 mt-1">
                      {trace?.probes_executed?.length > 0
                        ? `Block ${trace.probes_executed.map((p) => p.block_id).join(', ')}`
                        : probesCompatible ? 'No Probing Needed' : 'Disabled (Incompatible)'}
                    </p>
                    <p className="text-xs text-slate-500">
                      {trace?.probes_executed?.length > 0 ? 'Evidence collected' : 'Skipped'}
                    </p>
                  </div>

                  <div className="rounded-lg border border-slate-200 p-3 bg-slate-50/50">
                    <span className="text-[10px] font-semibold text-slate-400 uppercase">Step 3: Outcome</span>
                    <p className="text-sm font-bold text-slate-900 mt-1 font-mono">{adapt.status || 'Success'}</p>
                    <p className="text-xs text-slate-500">Continuation Cache: {adapt.cache_status || 'N/A'}</p>
                  </div>
                </div>
              )}

              {/* Comparison Mode Table */}
              {strategy === 'ALL' && result.comparison && (
                <div className="overflow-x-auto rounded-lg border border-slate-200">
                  <table className="w-full text-left text-xs">
                    <thead className="bg-slate-50 text-[10px] uppercase text-slate-400 border-b">
                      <tr>
                        <th className="py-2 px-3">Strategy</th>
                        <th>Prediction</th>
                        <th>Confidence</th>
                        <th>Latency</th>
                        <th>Match GT</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-100">
                      {result.comparison.map((item) => (
                        <tr key={item.strategy_key} className="hover:bg-slate-50">
                          <td className="py-2 px-3 font-semibold text-slate-800">{item.strategy_name}</td>
                          <td className="font-bold">{item.prediction_class}</td>
                          <td className="font-mono">{item.confidence != null ? `${(item.confidence * 100).toFixed(1)}%` : '—'}</td>
                          <td className="font-mono text-slate-600">{item.latency_ms?.toFixed(2)} ms</td>
                          <td>
                            {result.true_class ? (
                              <span className={`text-[10px] px-2 py-0.5 rounded font-bold ${
                                item.prediction_class === result.true_class ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                              }`}>
                                {item.prediction_class === result.true_class ? 'MATCH' : 'MISMATCH'}
                              </span>
                            ) : '—'}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}
        </div>
      </section>

      {/* ── SECTION 3: RECENT INFERENCE EVENTS ── */}
      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-3">
        <h2 className="font-bold text-sm text-slate-900">Current Session Inference History</h2>
        {history.length === 0 ? (
          <p className="text-xs text-slate-400">No inference requests executed in this session yet.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-slate-100 text-[10px] uppercase text-slate-400">
                <tr>
                  <th className="py-2">Time</th>
                  <th>Image Source</th>
                  <th>Prediction</th>
                  <th>Confidence</th>
                  <th>Decision Action</th>
                  <th>Latency</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-50 font-mono">
                {history.slice(0, 10).map((row, i) => (
                  <tr key={i} className="hover:bg-slate-50/70">
                    <td className="py-2 text-slate-400 text-[11px]">{row.timestamp}</td>
                    <td className="text-slate-800 font-sans">{row.image_name}</td>
                    <td className="font-bold text-slate-900 font-sans">{row.prediction}</td>
                    <td className="text-slate-600">{row.confidence != null ? `${(row.confidence * 100).toFixed(1)}%` : '—'}</td>
                    <td>
                      <span className={`text-[10px] px-2 py-0.5 rounded font-bold font-sans ${
                        row.action === 'RETAIN' ? 'bg-emerald-100 text-emerald-800' :
                        row.action === 'PROBE' ? 'bg-blue-100 text-blue-800' :
                        row.action === 'SWITCH' ? 'bg-indigo-100 text-indigo-800' :
                        'bg-slate-100 text-slate-600'
                      }`}>
                        {row.action || 'INFERENCE'}
                      </span>
                    </td>
                    <td className="text-slate-600">{row.latency_ms != null ? `${Number(row.latency_ms).toFixed(2)} ms` : '—'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
