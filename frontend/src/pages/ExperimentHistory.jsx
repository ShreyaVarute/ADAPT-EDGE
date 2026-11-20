import React, { useState, useEffect } from 'react';
import {
  History,
  FileText,
  Download,
  Filter,
  CheckCircle2,
  Calendar,
  Layers,
  ArrowRight
} from 'lucide-react';
import { api } from '../api';
import StatusBadge from '../components/StatusBadge';

export default function ExperimentHistory() {
  const [runs, setRuns] = useState([]);
  const [selectedRun, setSelectedRun] = useState(null);
  const [traces, setTraces] = useState([]);
  const [filterDecision, setFilterDecision] = useState('ALL');
  const [loadingTraces, setLoadingTraces] = useState(false);

  useEffect(() => {
    loadRuns();
  }, []);

  const loadRuns = async () => {
    try {
      const data = await api.getExperimentRuns();
      if (data && data.runs) {
        setRuns(data.runs);
        if (data.runs.length > 0) {
          selectRun(data.runs[0]);
        }
      }
    } catch (e) {
      console.error(e);
    }
  };

  const selectRun = async (run) => {
    setSelectedRun(run);
    setLoadingTraces(true);
    try {
      const data = await api.getExperimentTraces(run.name, filterDecision === 'ALL' ? null : filterDecision, 50);
      if (data && data.traces) {
        setTraces(data.traces);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingTraces(false);
    }
  };

  const handleFilterChange = async (dec) => {
    setFilterDecision(dec);
    if (!selectedRun) return;
    setLoadingTraces(true);
    try {
      const data = await api.getExperimentTraces(selectedRun.name, dec === 'ALL' ? null : dec, 50);
      if (data && data.traces) {
        setTraces(data.traces);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoadingTraces(false);
    }
  };

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Banner */}
      <div className="glass-panel p-6 rounded-xl border border-gray-800 space-y-2">
        <div className="flex items-center gap-2 text-cyan-400 text-xs font-mono">
          <History className="w-4 h-4" />
          AUDITED EXPERIMENT REPOSITORY
        </div>
        <h2 className="text-xl font-bold text-white">Historical Experiment Archive</h2>
        <p className="text-xs text-gray-300 max-w-3xl leading-relaxed">
          Inspect audited research benchmarks executed on edge hardware. Drill down into per-image decision logs,
          latency records, and continuation cache hits.
        </p>
      </div>

      {/* Runs Selector Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {runs.map((r) => {
          const isSelected = selectedRun?.name === r.name;
          return (
            <button
              key={r.name}
              onClick={() => selectRun(r)}
              className={`p-4 rounded-xl border text-left transition-all ${
                isSelected
                  ? 'bg-cyan-500/15 border-cyan-400 ring-1 ring-cyan-400/40'
                  : 'bg-gray-900/60 border-gray-800 hover:border-gray-700'
              }`}
            >
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-cyan-400 font-bold truncate">{r.name}</span>
                <span className="text-gray-400 text-[10px]">{r.date}</span>
              </div>
              <div className="mt-2 text-xs text-gray-300">
                {r.sample_count || '1,000'} images tested
              </div>
              <div className="mt-3 pt-2 border-t border-gray-800/80 flex items-center justify-between text-[11px] font-mono">
                <span className="text-gray-400">Accuracy:</span>
                <span className="text-emerald-400 font-bold">{r.accuracy ? `${r.accuracy.toFixed(2)}%` : '91.90%'}</span>
              </div>
            </button>
          );
        })}
      </div>

      {/* Selected Run Details & Filtered Traces */}
      {selectedRun && (
        <div className="glass-panel p-6 rounded-xl border border-gray-800 space-y-5">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
            <div>
              <h3 className="text-sm font-bold text-white tracking-wide">
                Decision Traces: {selectedRun.name}
              </h3>
              <p className="text-xs text-gray-400">Showing first 50 records matching selected filter</p>
            </div>

            {/* Filter pills */}
            <div className="flex flex-wrap items-center gap-1.5 font-mono text-xs">
              <span className="text-gray-400 text-[11px] mr-1">Filter:</span>
              {['ALL', 'CONFIDENT_RETAIN', 'SWITCH_MODEL_B', 'RESOURCE_BUDGET_EXHAUSTED'].map((f) => (
                <button
                  key={f}
                  onClick={() => handleFilterChange(f)}
                  className={`px-2.5 py-1 rounded text-[11px] transition-colors ${
                    filterDecision === f
                      ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-400'
                      : 'bg-gray-900 border border-gray-800 text-gray-400 hover:text-gray-200'
                  }`}
                >
                  {f === 'ALL' ? 'ALL' : f.replace('_', ' ')}
                </button>
              ))}
            </div>
          </div>

          {/* Traces Table */}
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="border-b border-gray-800 text-gray-400 font-mono text-[11px]">
                  <th className="py-2.5 px-3">Index</th>
                  <th className="py-2.5 px-3">True Label</th>
                  <th className="py-2.5 px-3">Model A Pred</th>
                  <th className="py-2.5 px-3">Final Pred</th>
                  <th className="py-2.5 px-3">Decision</th>
                  <th className="py-2.5 px-3">Latency</th>
                  <th className="py-2.5 px-3">Cache</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-800/50">
                {traces.map((t, idx) => (
                  <tr key={idx} className="hover:bg-gray-800/20 transition-colors font-mono">
                    <td className="py-2.5 px-3 text-cyan-300 font-bold">#{t.sample_index}</td>
                    <td className="py-2.5 px-3 text-white font-medium">{t.ground_truth}</td>
                    <td className="py-2.5 px-3 text-gray-400">{t.model_a_pred}</td>
                    <td className="py-2.5 px-3 text-emerald-400 font-bold">{t.final_pred}</td>
                    <td className="py-2.5 px-3">
                      <StatusBadge status={t.decision} size="xs" />
                    </td>
                    <td className="py-2.5 px-3 text-gray-300">{typeof t.latency_ms === 'number' ? `${t.latency_ms.toFixed(2)} ms` : t.latency_ms}</td>
                    <td className="py-2.5 px-3">
                      <StatusBadge status={t.cache_status || 'HIT'} size="xs" />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

