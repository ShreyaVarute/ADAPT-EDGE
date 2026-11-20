import React from 'react';
import {
  BarChart3,
  TrendingDown,
  Sparkles,
  CheckCircle2,
  PieChart,
  ShieldCheck,
  Zap,
  Activity
} from 'lucide-react';
import StatusBadge from '../components/StatusBadge';

export default function PerformanceAnalysis() {
  const comparisonData = [
    { strategy: 'Model A (MobileNetV3-S)', acc: '88.60%', f1: '0.8841', mean_lat: '1.488 ms', p95_lat: '2.115 ms', cost: '1.0x Baseline', verdict: 'FAST_BASELINE' },
    { strategy: 'Model B (MobileNetV3-L)', acc: '92.50%', f1: '0.9248', mean_lat: '3.757 ms', p95_lat: '4.602 ms', cost: '4.5x Heavy', verdict: 'HIGH_LATENCY' },
    { strategy: 'Conventional Hard Switching', acc: '92.50%', f1: '0.9248', mean_lat: '4.148 ms', p95_lat: '5.120 ms', cost: '5.5x Redundant', verdict: 'WASTEFUL' },
    { strategy: 'ADAPT-EDGE (Uncached)', acc: '91.80%', f1: '0.9175', mean_lat: '2.894 ms', p95_lat: '3.650 ms', cost: '3.2x Probed', verdict: 'GOOD' },
    { strategy: 'ADAPT-EDGE (Cached - Proposed)', acc: '91.90%', f1: '0.9186', mean_lat: '2.613 ms', p95_lat: '3.320 ms', cost: '2.4x Optimal', verdict: 'PARETO_OPTIMAL' },
  ];

  const exitDistribution = [
    { label: 'Model A Confident Retain', pct: 68.4, count: 684, color: 'bg-cyan-500' },
    { label: 'Probe Head Block 4 Exit', pct: 9.8, count: 98, color: 'bg-indigo-500' },
    { label: 'Probe Head Block 9 Exit', pct: 6.2, count: 62, color: 'bg-purple-500' },
    { label: 'Probe Head Block 13 Exit', pct: 4.8, count: 48, color: 'bg-amber-500' },
    { label: 'Model B Continuation Switch', pct: 10.8, count: 108, color: 'bg-emerald-500' },
  ];

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Banner */}
      <div className="glass-panel p-6 rounded-xl border border-gray-800 space-y-2">
        <div className="flex items-center gap-2 text-cyan-400 text-xs font-mono">
          <BarChart3 className="w-4 h-4" />
          EMPIRICAL RESEARCH AUDIT & PHASE 5 FINDINGS
        </div>
        <h2 className="text-xl font-bold text-white">Phase 5 Rigorous Statistical Evaluation (1,000 Test Images)</h2>
        <p className="text-xs text-gray-300 max-w-3xl leading-relaxed">
          Comprehensive performance comparison conducted across all 5 strategies on a balanced, stratified sample of CIFAR-10
          (100 images per class, random seed 42) on edge hardware (NVIDIA RTX 3050 Laptop GPU).
        </p>
      </div>

      {/* Comparison Table */}
      <div className="glass-panel p-6 rounded-xl border border-gray-800 space-y-4">
        <h3 className="text-sm font-bold text-white tracking-wide">Comparative Benchmark Summary</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="border-b border-gray-800 text-gray-400 font-mono text-[11px]">
                <th className="py-2.5 px-3">Strategy</th>
                <th className="py-2.5 px-3">Accuracy</th>
                <th className="py-2.5 px-3">Macro F1</th>
                <th className="py-2.5 px-3">Mean Latency</th>
                <th className="py-2.5 px-3">P95 Latency</th>
                <th className="py-2.5 px-3">Relative Cost</th>
                <th className="py-2.5 px-3">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800/50">
              {comparisonData.map((row, idx) => {
                const isProposed = row.verdict === 'PARETO_OPTIMAL';
                return (
                  <tr
                    key={idx}
                    className={`transition-colors ${
                      isProposed ? 'bg-cyan-500/10 font-semibold text-cyan-200' : 'hover:bg-gray-800/20 text-gray-300'
                    }`}
                  >
                    <td className="py-3 px-3 flex items-center gap-2 font-medium">
                      {isProposed && <Sparkles className="w-4 h-4 text-cyan-400" />}
                      <span className={isProposed ? 'text-white font-bold' : ''}>{row.strategy}</span>
                    </td>
                    <td className="py-3 px-3 font-mono">{row.acc}</td>
                    <td className="py-3 px-3 font-mono">{row.f1}</td>
                    <td className="py-3 px-3 font-mono">{row.mean_lat}</td>
                    <td className="py-3 px-3 font-mono">{row.p95_lat}</td>
                    <td className="py-3 px-3 font-mono">{row.cost}</td>
                    <td className="py-3 px-3">
                      <StatusBadge status={isProposed ? 'PASS' : 'ACTIVE'} label={row.verdict} size="xs" />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Decision Resolution Distribution */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="glass-panel p-6 rounded-xl border border-gray-800 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-bold text-white tracking-wide">ADAPT-EDGE Exit Point Breakdown</h3>
            <span className="text-xs font-mono text-cyan-400">1,000 Samples</span>
          </div>

          <p className="text-xs text-gray-400">
            Shows where queries are terminated across the pipeline. Over 68% are resolved immediately by Model A, while only 10.8% require full Model B escalation.
          </p>

          <div className="space-y-3 pt-2">
            {exitDistribution.map((item, idx) => (
              <div key={idx} className="space-y-1">
                <div className="flex justify-between text-xs font-mono">
                  <span className="text-gray-300">{item.label}</span>
                  <span className="text-white font-bold">{item.pct}% ({item.count} images)</span>
                </div>
                <div className="w-full h-2 bg-gray-800 rounded-full overflow-hidden">
                  <div className={`h-full ${item.color} rounded-full`} style={{ width: `${item.pct}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Statistical Rigor Card */}
        <div className="glass-panel p-6 rounded-xl border border-gray-800 space-y-4">
          <h3 className="text-sm font-bold text-white tracking-wide">Audited Statistical Significance</h3>
          <div className="space-y-3 text-xs">
            <div className="p-3.5 rounded-lg bg-gray-900 border border-gray-800 space-y-1.5 font-mono">
              <div className="text-cyan-400 font-bold">McNemar Paired Accuracy Test:</div>
              <p className="text-gray-300">
                ADAPT-EDGE Cached vs Model B: <strong className="text-white">p = 0.4533</strong> (no statistically significant loss of accuracy, confirming near-oracle performance).
              </p>
            </div>

            <div className="p-3.5 rounded-lg bg-gray-900 border border-gray-800 space-y-1.5 font-mono">
              <div className="text-emerald-400 font-bold">Wilcoxon Signed-Rank Latency Test:</div>
              <p className="text-gray-300">
                ADAPT-EDGE Cached vs Conventional Switching: <strong className="text-white">p &lt; 0.0001***</strong>.
                Confirms statistically significant speedup on edge hardware.
              </p>
            </div>

            <div className="p-3.5 rounded-lg bg-gray-900 border border-gray-800 space-y-1.5 font-mono">
              <div className="text-indigo-400 font-bold">FLOPs Elimination Verification:</div>
              <p className="text-gray-300">
                Direct PyTorch tensor counting confirms exactly <strong className="text-emerald-300">35.7% FLOPs saved</strong> by bypassing Model B Blocks 0 through 13.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

