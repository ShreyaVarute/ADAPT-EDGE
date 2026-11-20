import React, { useState, useEffect } from 'react';
import {
  FileCheck2,
  Play,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  RotateCcw,
  ShieldCheck,
  Cpu,
  Layers,
  Sparkles
} from 'lucide-react';
import { api } from '../api';
import StatusBadge from '../components/StatusBadge';

export default function SystemValidation() {
  const [running, setRunning] = useState(false);
  const [report, setReport] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    runTests();
  }, []);

  const runTests = async () => {
    setRunning(true);
    setError(null);
    try {
      const data = await api.runDiagnostics();
      setReport(data);
    } catch (e) {
      setError(e.message || 'Diagnostic execution failed');
    } finally {
      setRunning(false);
    }
  };

  const getStatusIcon = (status) => {
    if (status === 'PASS') return <CheckCircle2 className="w-5 h-5 text-emerald-400" />;
    if (status === 'WARN') return <AlertTriangle className="w-5 h-5 text-amber-400" />;
    return <XCircle className="w-5 h-5 text-rose-400" />;
  };

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Top Banner */}
      <div className="glass-panel p-6 rounded-xl border border-gray-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-cyan-400 text-xs font-mono">
            <ShieldCheck className="w-4 h-4" />
            AUTOMATED INTEGRITY VERIFICATION
          </div>
          <h2 className="text-xl font-bold text-white">8-Point System Validation Suite</h2>
          <p className="text-xs text-gray-400">
            Comprehensive diagnostic auditing checkpoints, tensor dimensions, bitwise parity, and unit tests.
          </p>
        </div>

        <button
          onClick={runTests}
          disabled={running}
          className="px-5 py-2.5 rounded-xl bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 disabled:opacity-50 text-white font-bold text-xs flex items-center gap-2 shadow-lg shadow-cyan-500/20 transition-all cursor-pointer"
        >
          {running ? (
            <>
              <div className="w-3.5 h-3.5 border-2 border-white/20 border-t-white rounded-full animate-spin" />
              Running Diagnostics...
            </>
          ) : (
            <>
              <Play className="w-4 h-4 fill-white" />
              Run Validation Suite
            </>
          )}
        </button>
      </div>

      {/* Summary KPI Strip */}
      {report && (
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-4">
          <div className="glass-panel p-4 rounded-xl border border-gray-800 text-center">
            <span className="text-xs font-mono text-gray-400">TOTAL CHECKS</span>
            <div className="text-2xl font-bold text-white font-mono mt-1">{report.total_checks}</div>
          </div>
          <div className="glass-panel p-4 rounded-xl border border-emerald-500/20 text-center bg-emerald-500/5">
            <span className="text-xs font-mono text-emerald-400">PASSED</span>
            <div className="text-2xl font-bold text-emerald-400 font-mono mt-1">{report.passed}</div>
          </div>
          <div className="glass-panel p-4 rounded-xl border border-amber-500/20 text-center bg-amber-500/5">
            <span className="text-xs font-mono text-amber-400">WARNINGS</span>
            <div className="text-2xl font-bold text-amber-400 font-mono mt-1">{report.warnings}</div>
          </div>
          <div className="glass-panel p-4 rounded-xl border border-rose-500/20 text-center bg-rose-500/5">
            <span className="text-xs font-mono text-rose-400">FAILURES</span>
            <div className="text-2xl font-bold text-rose-400 font-mono mt-1">{report.failed}</div>
          </div>
        </div>
      )}

      {/* Test List */}
      <div className="space-y-3">
        {report && report.checks.map((c) => (
          <div
            key={c.id}
            className="glass-panel p-5 rounded-xl border border-gray-800 flex flex-col md:flex-row md:items-center justify-between gap-4 hover:border-gray-700 transition-colors"
          >
            <div className="flex items-start gap-3.5">
              <div className="p-2 rounded-lg bg-gray-900 border border-gray-800 mt-0.5">
                {getStatusIcon(c.status)}
              </div>
              <div className="space-y-1">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono text-cyan-400">Check #{c.id}</span>
                  <h3 className="text-sm font-bold text-white">{c.name}</h3>
                </div>
                <p className="text-xs text-gray-300">{c.description}</p>
                {c.details && (
                  <div className="text-[11px] font-mono text-gray-400 pt-1 flex flex-wrap gap-x-4 gap-y-1">
                    {Object.entries(c.details).map(([k, v]) => (
                      <span key={k}>
                        {k}: <strong className="text-gray-200">{typeof v === 'object' ? JSON.stringify(v) : String(v)}</strong>
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="flex items-center gap-3 shrink-0">
              <span className="text-xs font-mono text-gray-400">{c.execution_ms.toFixed(1)} ms</span>
              <StatusBadge status={c.status} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

