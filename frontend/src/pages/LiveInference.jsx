import React, { useState, useEffect } from 'react';
import {
  Zap,
  Play,
  RotateCcw,
  CheckCircle2,
  AlertCircle,
  HelpCircle,
  ArrowRight,
  Cpu,
  Layers,
  Sparkles,
  Database,
  Sliders,
  ChevronDown,
  Info
} from 'lucide-react';
import { api } from '../api';
import StatusBadge from '../components/StatusBadge';

export default function LiveInference() {
  const [samples, setSamples] = useState([]);
  const [selectedSample, setSelectedSample] = useState(null);
  const [strategy, setStrategy] = useState('ALL');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  // Controller override knobs
  const [tauHigh, setTauHigh] = useState(0.85);
  const [tauLow, setTauLow] = useState(0.50);
  const [tauProbe, setTauProbe] = useState(0.80);
  const [latencyBudget, setLatencyBudget] = useState(25.0);

  useEffect(() => {
    loadSamples();
  }, []);

  const loadSamples = async () => {
    try {
      const data = await api.getSamples(12);
      if (data && data.samples && data.samples.length > 0) {
        setSamples(data.samples);
        setSelectedSample(data.samples[0]);
      }
    } catch (e) {
      console.error('Failed to load sample gallery:', e);
    }
  };

  const runInference = async () => {
    if (!selectedSample) return;
    setLoading(true);
    setError(null);
    try {
      const payload = {
        sample_index: selectedSample.index,
        strategy: strategy,
        tau_high: parseFloat(tauHigh),
        tau_low: parseFloat(tauLow),
        tau_probe: parseFloat(tauProbe),
        latency_budget_ms: parseFloat(latencyBudget)
      };
      const res = await api.predict(payload);
      setResult(res);
    } catch (e) {
      setError(e.message || 'Inference execution failed');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Top Controls Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Sample Selection Panel */}
        <div className="glass-panel p-5 rounded-xl border border-gray-800 space-y-4 lg:col-span-2">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-sm font-bold text-white tracking-wide flex items-center gap-2">
                <Database className="w-4 h-4 text-cyan-400" />
                Select CIFAR-10 Test Sample
              </h2>
              <p className="text-xs text-gray-400">Click a test sample with known ground truth label</p>
            </div>
            <button
              onClick={loadSamples}
              className="text-xs text-cyan-400 hover:text-cyan-300 font-mono flex items-center gap-1"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              New Samples
            </button>
          </div>

          {/* Sample Thumbnails Grid */}
          <div className="grid grid-cols-4 sm:grid-cols-6 gap-2.5">
            {samples.map((s) => {
              const isSelected = selectedSample?.index === s.index;
              return (
                <button
                  key={s.index}
                  onClick={() => setSelectedSample(s)}
                  className={`p-2 rounded-lg border flex flex-col items-center gap-1.5 transition-all text-left ${
                    isSelected
                      ? 'bg-cyan-500/15 border-cyan-400 ring-2 ring-cyan-400/30'
                      : 'bg-gray-900/60 border-gray-800 hover:border-gray-700'
                  }`}
                >
                  <img
                    src={s.image_base64}
                    alt={s.class_name}
                    className="w-12 h-12 object-cover rounded bg-black"
                  />
                  <div className="text-center w-full">
                    <span className="text-[10px] font-mono text-cyan-300 truncate block">
                      #{s.index}
                    </span>
                    <span className="text-[11px] font-medium text-white truncate block">
                      {s.class_name}
                    </span>
                  </div>
                </button>
              );
            })}
          </div>

          {/* Selected Sample Details Card */}
          {selectedSample && (
            <div className="flex items-center justify-between p-3 rounded-lg bg-gray-900/80 border border-gray-800 text-xs font-mono">
              <div className="flex items-center gap-3">
                <img
                  src={selectedSample.image_base64}
                  alt={selectedSample.class_name}
                  className="w-8 h-8 rounded border border-gray-700"
                />
                <div>
                  <span className="text-gray-400">Active Test Sample: </span>
                  <span className="text-white font-bold">Index #{selectedSample.index}</span>
                  <span className="text-cyan-400 ml-2">({selectedSample.class_name})</span>
                </div>
              </div>
              <span className="text-gray-400 text-[11px]">Class ID: {selectedSample.label}</span>
            </div>
          )}
        </div>

        {/* Strategy and Trigger Panel */}
        <div className="glass-panel p-5 rounded-xl border border-gray-800 space-y-4 flex flex-col justify-between">
          <div className="space-y-4">
            <div>
              <h2 className="text-sm font-bold text-white tracking-wide flex items-center gap-2">
                <Sliders className="w-4 h-4 text-cyan-400" />
                Execution Mode
              </h2>
              <p className="text-xs text-gray-400">Choose single strategy or compare all simultaneously</p>
            </div>

            <div className="space-y-1.5">
              <label className="text-xs text-gray-300 font-medium">Inference Strategy</label>
              <select
                value={strategy}
                onChange={(e) => setStrategy(e.target.value)}
                className="w-full bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-xs text-white focus:outline-none focus:border-cyan-400"
              >
                <option value="ALL">Compare All 4 Strategies (Recommended)</option>
                <option value="ADAPT_EDGE_CACHED">ADAPT-EDGE (Cached)</option>
                <option value="ADAPT_EDGE_UNCACHED">ADAPT-EDGE (Uncached)</option>
                <option value="MODEL_A">Model A Baseline (MobileNetV3-S)</option>
                <option value="MODEL_B">Model B Baseline (MobileNetV3-L)</option>
                <option value="CONVENTIONAL">Conventional Hard Switching</option>
              </select>
            </div>

            {/* Threshold fine-tuners */}
            <div className="p-3 rounded-lg bg-gray-900/60 border border-gray-800/80 space-y-2.5 text-xs font-mono">
              <div className="text-[11px] text-gray-400 uppercase tracking-wider font-semibold">Controller Thresholds</div>
              <div className="flex items-center justify-between text-gray-300">
                <span>Tau High (Retain):</span>
                <span className="text-cyan-300 font-bold">{tauHigh}</span>
              </div>
              <input
                type="range"
                min="0.5"
                max="0.99"
                step="0.01"
                value={tauHigh}
                onChange={(e) => setTauHigh(e.target.value)}
                className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-400"
              />

              <div className="flex items-center justify-between text-gray-300">
                <span>Tau Low (Switch):</span>
                <span className="text-cyan-300 font-bold">{tauLow}</span>
              </div>
              <input
                type="range"
                min="0.2"
                max="0.8"
                step="0.01"
                value={tauLow}
                onChange={(e) => setTauLow(e.target.value)}
                className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-400"
              />
            </div>
          </div>

          <button
            onClick={runInference}
            disabled={loading || !selectedSample}
            className="w-full py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 disabled:opacity-50 text-white font-bold text-xs flex items-center justify-center gap-2 shadow-lg shadow-cyan-500/20 transition-all cursor-pointer"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                Executing Neural Pipeline...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-white" />
                Run Real-Time Inference
              </>
            )}
          </button>
        </div>
      </div>

      {/* Error Notice */}
      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Results Display */}
      {result && (
        <div className="space-y-6">
          {/* If Strategy was ALL: Show Side-by-Side Comparison */}
          {result.comparison && (
            <div className="glass-panel p-6 rounded-xl border border-gray-800 space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-sm font-bold text-white tracking-wide">Multi-Strategy Comparison Matrix</h3>
                  <p className="text-xs text-gray-400">Exact side-by-side execution metrics on Sample #{result.sample_index} (Ground Truth: {result.ground_truth})</p>
                </div>
                <span className="text-xs font-mono text-cyan-400 bg-cyan-500/10 px-2.5 py-1 rounded-full border border-cyan-500/30">
                  GT Class: {result.ground_truth}
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                {result.comparison.map((c, idx) => {
                  const isCorrect = c.prediction_class === result.ground_truth;
                  const isAdaptEdge = c.strategy_key === 'ADAPT_EDGE_CACHED';
                  return (
                    <div
                      key={idx}
                      className={`p-4 rounded-xl border transition-all ${
                        isAdaptEdge
                          ? 'bg-gradient-to-b from-cyan-950/40 to-gray-900 border-cyan-500/50 ring-1 ring-cyan-500/30'
                          : 'bg-gray-900/60 border-gray-800'
                      }`}
                    >
                      <div className="flex items-center justify-between text-xs font-mono text-gray-400">
                        <span className="font-semibold truncate">{c.strategy_name}</span>
                        {isAdaptEdge && (
                          <span className="text-[10px] text-cyan-300 px-1.5 py-0.5 rounded bg-cyan-500/20">Ours</span>
                        )}
                      </div>

                      <div className="mt-3 flex items-baseline justify-between">
                        <span className="text-lg font-bold text-white">{c.prediction_class}</span>
                        <StatusBadge
                          status={isCorrect ? 'PASS' : 'FAIL'}
                          label={isCorrect ? 'CORRECT' : 'WRONG'}
                          size="xs"
                        />
                      </div>

                      <div className="mt-3 pt-3 border-t border-gray-800/80 space-y-1.5 text-xs font-mono">
                        <div className="flex justify-between text-gray-400">
                          <span>Confidence:</span>
                          <span className="text-white font-medium">{(c.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <div className="flex justify-between text-gray-400">
                          <span>Inference Latency:</span>
                          <span className={`${isAdaptEdge ? 'text-cyan-300 font-bold' : 'text-gray-300'}`}>
                            {c.latency_ms.toFixed(2)} ms
                          </span>
                        </div>
                        {c.cache_status && (
                          <div className="flex justify-between text-gray-400">
                            <span>Cache Reuse:</span>
                            <span className="text-indigo-300">{c.cache_status}</span>
                          </div>
                        )}
                        {c.decision && (
                          <div className="flex justify-between text-gray-400">
                            <span>Final Decision:</span>
                            <StatusBadge status={c.decision} size="xs" />
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Deep Execution Trace & Decision Rationale for ADAPT-EDGE */}
          {result.adapt_edge_cached && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Left 2 Cols: Step-by-Step Decision Timeline */}
              <div className="glass-panel p-6 rounded-xl border border-gray-800 space-y-4 lg:col-span-2">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-bold text-white tracking-wide flex items-center gap-2">
                    <Sparkles className="w-4 h-4 text-cyan-400" />
                    ADAPT-EDGE Step-by-Step Execution Trace
                  </h3>
                  <StatusBadge status={result.adapt_edge_cached.decision} />
                </div>

                {/* Timeline Component */}
                <div className="relative pl-6 space-y-5 before:absolute before:left-2 before:top-2 before:bottom-2 before:w-0.5 before:bg-gray-800">
                  {/* Step 1: Model A Forward Pass */}
                  <div className="relative">
                    <div className="absolute -left-6 top-0.5 w-4 h-4 rounded-full bg-cyan-500/20 border-2 border-cyan-400"></div>
                    <div className="p-3.5 rounded-lg bg-gray-900/80 border border-gray-800 space-y-1">
                      <div className="flex items-center justify-between text-xs font-mono">
                        <span className="text-cyan-300 font-bold">Step 1: Model A Forward Pass (MobileNetV3-S)</span>
                        <span className="text-gray-400">{result.adapt_edge_cached.trace?.model_a_latency_ms?.toFixed(2)} ms</span>
                      </div>
                      <p className="text-xs text-gray-300">
                        Evaluated lightweight model. Output prediction: <strong className="text-white">{result.adapt_edge_cached.trace?.model_a_pred_class}</strong> with{' '}
                        <strong className="text-cyan-300">{((result.adapt_edge_cached.trace?.model_a_conf || 0) * 100).toFixed(1)}% confidence</strong>.
                      </p>
                    </div>
                  </div>

                  {/* Step 2: Decision Controller Evaluation */}
                  <div className="relative">
                    <div className="absolute -left-6 top-0.5 w-4 h-4 rounded-full bg-amber-500/20 border-2 border-amber-400"></div>
                    <div className="p-3.5 rounded-lg bg-gray-900/80 border border-gray-800 space-y-1">
                      <div className="flex items-center justify-between text-xs font-mono">
                        <span className="text-amber-300 font-bold">Step 2: Decision Controller Policy Assessment</span>
                        <StatusBadge status={result.adapt_edge_cached.trace?.decision} size="xs" />
                      </div>
                      <p className="text-xs text-gray-300">
                        Confidence was evaluated against <span className="font-mono text-cyan-300">tau_high ({tauHigh})</span> and{' '}
                        <span className="font-mono text-cyan-300">tau_low ({tauLow})</span>.
                        Policy Action: <strong className="text-amber-300">{result.adapt_edge_cached.trace?.decision}</strong>.
                      </p>
                    </div>
                  </div>

                  {/* Step 3: Probe Execution if any */}
                  {result.adapt_edge_cached.trace?.probes_executed && result.adapt_edge_cached.trace.probes_executed.length > 0 && (
                    <div className="relative">
                      <div className="absolute -left-6 top-0.5 w-4 h-4 rounded-full bg-indigo-500/20 border-2 border-indigo-400"></div>
                      <div className="p-3.5 rounded-lg bg-gray-900/80 border border-gray-800 space-y-2">
                        <div className="flex items-center justify-between text-xs font-mono">
                          <span className="text-indigo-300 font-bold">Step 3: Selective Probe Execution</span>
                          <span className="text-indigo-400">Intermediate Exits</span>
                        </div>
                        <div className="space-y-1.5">
                          {result.adapt_edge_cached.trace.probes_executed.map((p, pIdx) => (
                            <div key={pIdx} className="p-2 rounded bg-indigo-950/30 border border-indigo-500/20 text-xs flex justify-between items-center">
                              <span className="font-mono text-white">Block {p.block_id} Probe</span>
                              <span className="text-gray-300">Pred: <strong className="text-indigo-300">{p.pred_class}</strong></span>
                              <span className="font-mono text-cyan-300">{(p.confidence * 100).toFixed(1)}%</span>
                              <StatusBadge status={p.resolved ? 'RESOLVED' : 'INCONCLUSIVE'} size="xs" />
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}

                  {/* Step 4: Continuation Cache & Final Exit */}
                  <div className="relative">
                    <div className="absolute -left-6 top-0.5 w-4 h-4 rounded-full bg-emerald-500/20 border-2 border-emerald-400"></div>
                    <div className="p-3.5 rounded-lg bg-emerald-950/20 border border-emerald-500/30 space-y-1">
                      <div className="flex items-center justify-between text-xs font-mono">
                        <span className="text-emerald-300 font-bold">Step 4: Final Output & Computation Savings</span>
                        <span className="text-emerald-400 font-bold">{result.adapt_edge_cached.latency_ms?.toFixed(2)} ms Total</span>
                      </div>
                      <p className="text-xs text-gray-300">
                        Final Model Selected: <strong className="text-white">{result.adapt_edge_cached.trace?.final_model}</strong>.
                        Cache Status: <span className="text-indigo-300 font-mono">{result.adapt_edge_cached.trace?.continuation_cache_status}</span>.
                        Avoided redundant early layers of Model B!
                      </p>
                    </div>
                  </div>
                </div>
              </div>

              {/* Right Col: Transparent "Why did ADAPT-EDGE decide this?" Card */}
              <div className="glass-panel p-6 rounded-xl border border-gray-800 space-y-4">
                <h3 className="text-sm font-bold text-white tracking-wide flex items-center gap-2">
                  <Info className="w-4 h-4 text-cyan-400" />
                  Decision Transparency
                </h3>

                <div className="p-4 rounded-xl bg-gray-900/90 border border-gray-800 space-y-3 text-xs">
                  <div className="text-[11px] font-mono uppercase text-gray-400 font-semibold">Mathematical Rationale</div>
                  <p className="text-gray-300 leading-relaxed">
                    {result.adapt_edge_cached.trace?.decision_reason ||
                      'Confidence and probe information gain thresholds were evaluated dynamically against available latency budget.'}
                  </p>

                  <div className="pt-2 border-t border-gray-800 space-y-2 font-mono">
                    <div className="flex justify-between">
                      <span className="text-gray-400">Sample Index:</span>
                      <span className="text-cyan-300">#{result.sample_index}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">True Label:</span>
                      <span className="text-white font-bold">{result.ground_truth}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Predicted Label:</span>
                      <span className="text-emerald-400 font-bold">{result.adapt_edge_cached.prediction_class}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">Top-1 Confidence:</span>
                      <span className="text-white">{((result.adapt_edge_cached.confidence || 0) * 100).toFixed(1)}%</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-400">FLOPs Cost Saved:</span>
                      <span className="text-emerald-400 font-bold">~35.7%</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

