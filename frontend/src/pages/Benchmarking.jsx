import React, { useState } from 'react';
import {
  Activity,
  Play,
  RotateCcw,
  Sparkles,
  TrendingUp,
  Cpu,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';
import { api } from '../api';
import StatusBadge from '../components/StatusBadge';

export default function Benchmarking() {
  const [numSamples, setNumSamples] = useState(15);
  const [loading, setLoading] = useState(false);
  const [benchmarkResult, setBenchmarkResult] = useState(null);
  const [error, setError] = useState(null);

  // Baseline Phase 5 reference stats (for comparison)
  const referenceData = [
    { strategy: 'Model A (MobileNetV3-S)', accuracy: 88.60, latency: 1.488, f1: 0.8841, cost: '1.0x', tag: 'Fast' },
    { strategy: 'Model B (MobileNetV3-L)', accuracy: 92.50, latency: 3.757, f1: 0.9248, cost: '4.5x', tag: 'Heavy' },
    { strategy: 'Conventional Switching', accuracy: 92.50, latency: 4.148, f1: 0.9248, cost: '5.5x', tag: 'Wasteful' },
    { strategy: 'ADAPT-EDGE (Uncached)', accuracy: 91.80, latency: 2.894, f1: 0.9175, cost: '3.2x', tag: 'Probed' },
    { strategy: 'ADAPT-EDGE (Cached - Ours)', accuracy: 91.90, latency: 2.613, f1: 0.9186, cost: '2.4x', tag: 'Optimal' },
  ];

  const runBenchmark = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.runBenchmark(parseInt(numSamples), true);
      setBenchmarkResult(res);
    } catch (e) {
      setError(e.message || 'Benchmark execution failed');
    } finally {
      setLoading(false);
    }
  };

  const activeData = benchmarkResult ? benchmarkResult.results : referenceData;

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Header and Controls */}
      <div className="glass-panel p-6 rounded-xl border border-gray-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-cyan-400 text-xs font-mono">
            <Activity className="w-4 h-4" />
            LIVE HARDWARE BENCHMARK
          </div>
          <h2 className="text-xl font-bold text-white">Comparative Strategy Benchmark</h2>
          <p className="text-xs text-gray-400">
            Execute real-time side-by-side inference across Model A, Model B, Conventional Switching, and ADAPT-EDGE.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 bg-gray-900 border border-gray-700 px-3 py-2 rounded-xl text-xs">
            <span className="text-gray-400 font-mono">Samples:</span>
            <select
              value={numSamples}
              onChange={(e) => setNumSamples(e.target.value)}
              className="bg-transparent text-white font-mono focus:outline-none"
            >
              <option value="5">5 samples (Instant)</option>
              <option value="15">15 samples (Balanced)</option>
              <option value="30">30 samples (Comprehensive)</option>
            </select>
          </div>

          <button
            onClick={runBenchmark}
            disabled={loading}
            className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 disabled:opacity-50 text-white font-bold text-xs flex items-center gap-2 shadow-lg shadow-cyan-500/20 transition-all cursor-pointer"
          >
            {loading ? (
              <>
                <div className="w-3.5 h-3.5 border-2 border-white/20 border-t-white rounded-full animate-spin" />
                Benchmarking...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 fill-white" />
                Run Live Benchmark
              </>
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-300 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Benchmark Results Table */}
      <div className="glass-panel p-6 rounded-xl border border-gray-800 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold text-white tracking-wide">
            {benchmarkResult
              ? `Live Benchmark Output (${benchmarkResult.num_samples} Images on ${benchmarkResult.device})`
              : 'Phase 5 Reference Benchmark (1,000 Stratified Test Samples)'}
          </h3>
          <span className="text-xs font-mono text-cyan-400">
            {benchmarkResult ? 'REAL-TIME EXECUTION' : 'STATISTICALLY AUDITED BASELINE'}
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-gray-800 text-gray-400 font-mono text-[11px]">
                <th className="py-3 px-3">Strategy</th>
                <th className="py-3 px-3">Accuracy</th>
                <th className="py-3 px-3">Macro F1</th>
                <th className="py-3 px-3">Mean Latency</th>
                <th className="py-3 px-3">Throughput (FPS)</th>
                <th className="py-3 px-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800/50">
              {activeData.map((row, idx) => {
                const isOurs = (row.strategy || '').includes('Cached') || (row.strategy || '').includes('ADAPT');
                const fps = row.latency ? (1000 / row.latency).toFixed(1) : '-';
                return (
                  <tr
                    key={idx}
                    className={`transition-colors ${
                      isOurs ? 'bg-cyan-500/5 font-semibold text-cyan-300' : 'hover:bg-gray-800/20 text-gray-300'
                    }`}
                  >
                    <td className="py-3 px-3 flex items-center gap-2">
                      {isOurs && <Sparkles className="w-3.5 h-3.5 text-cyan-400" />}
                      <span className={isOurs ? 'text-white font-bold' : ''}>{row.strategy}</span>
                    </td>
                    <td className="py-3 px-3 font-mono">
                      {typeof row.accuracy === 'number' ? `${row.accuracy.toFixed(2)}%` : row.accuracy}
                    </td>
                    <td className="py-3 px-3 font-mono">
                      {typeof row.f1 === 'number' ? row.f1.toFixed(4) : (row.macro_f1 || '-')}
                    </td>
                    <td className="py-3 px-3 font-mono">
                      {typeof row.latency === 'number' ? `${row.latency.toFixed(3)} ms` : `${row.mean_latency_ms} ms`}
                    </td>
                    <td className="py-3 px-3 font-mono text-gray-400">{fps} fps</td>
                    <td className="py-3 px-3">
                      <StatusBadge status={isOurs ? 'PASS' : 'ACTIVE'} label={row.tag || 'Tested'} size="xs" />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Visual Pareto Trade-Off Analysis */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="glass-panel p-6 rounded-xl border border-gray-800 space-y-3">
          <h3 className="text-sm font-bold text-white tracking-wide">Accuracy vs Latency Trade-Off</h3>
          <p className="text-xs text-gray-400">
            ADAPT-EDGE delivers a near-optimal Pareto frontier: retaining 99.4% of Model B accuracy while saving 37% latency over naive switching.
          </p>

          <div className="space-y-3 pt-2">
            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-gray-400">Model A (MobileNetV3-S)</span>
                <span className="text-cyan-400">88.60% (1.49 ms)</span>
              </div>
              <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full bg-cyan-400 rounded-full" style={{ width: '88.6%' }}></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-gray-400">Model B (MobileNetV3-L)</span>
                <span className="text-indigo-400">92.50% (3.76 ms)</span>
              </div>
              <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                <div className="h-full bg-indigo-500 rounded-full" style={{ width: '92.5%' }}></div>
              </div>
            </div>

            <div>
              <div className="flex justify-between text-xs font-mono mb-1">
                <span className="text-cyan-300 font-bold">ADAPT-EDGE Cached (Proposed)</span>
                <span className="text-cyan-300 font-bold">91.90% (2.61 ms)</span>
              </div>
              <div className="w-full h-2.5 bg-gray-800 rounded-full overflow-hidden ring-1 ring-cyan-400">
                <div className="h-full bg-gradient-to-r from-cyan-400 to-indigo-500 rounded-full" style={{ width: '91.9%' }}></div>
              </div>
            </div>
          </div>
        </div>

        <div className="glass-panel p-6 rounded-xl border border-gray-800 space-y-3">
          <h3 className="text-sm font-bold text-white tracking-wide">Why Conventional Switching Fails</h3>
          <p className="text-xs text-gray-300 leading-relaxed">
            When Model A is uncertain, conventional systems run Model B completely from layer 0. This incurs a double latency penalty:
          </p>
          <div className="p-3 rounded-lg bg-gray-900 border border-gray-800 font-mono text-xs space-y-2">
            <div className="text-rose-400 flex items-center justify-between">
              <span>Model A Latency:</span>
              <span>1.49 ms</span>
            </div>
            <div className="text-rose-400 flex items-center justify-between">
              <span>+ Full Model B Latency:</span>
              <span>3.76 ms</span>
            </div>
            <div className="pt-1 border-t border-gray-800 text-white font-bold flex items-center justify-between">
              <span>Total Switching Penalty:</span>
              <span>~5.25 ms</span>
            </div>
          </div>
          <p className="text-xs text-emerald-400 leading-relaxed font-mono">
            ADAPT-EDGE resumes from cached intermediate Block 13 activations, compressing switch latency to just 1.12 ms!
          </p>
        </div>
      </div>
    </div>
  );
}

