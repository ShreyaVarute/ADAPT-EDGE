import React, { useState, useEffect } from 'react';
import {
  DownloadCloud,
  FileText,
  Copy,
  CheckCircle2,
  Printer,
  FileSpreadsheet,
  Sparkles
} from 'lucide-react';
import { api } from '../api';

export default function ReportsExport() {
  const [markdown, setMarkdown] = useState('');
  const [copied, setCopied] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadReport();
  }, []);

  const loadReport = async () => {
    try {
      setLoading(true);
      const text = await api.getMarkdownReport();
      setMarkdown(text);
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const copyToClipboard = () => {
    navigator.clipboard.writeText(markdown);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Banner */}
      <div className="glass-panel p-6 rounded-xl border border-gray-800 space-y-2">
        <div className="flex items-center gap-2 text-cyan-400 text-xs font-mono">
          <DownloadCloud className="w-4 h-4" />
          PUBLICATIONS & DATA DISSEMINATION
        </div>
        <h2 className="text-xl font-bold text-white">Research Reporting & Experimental Exports</h2>
        <p className="text-xs text-gray-300 max-w-3xl leading-relaxed">
          Export verified experimental benchmarks, generate IEEE-formatted project summaries, and download raw CSV
          per-image traces for faculty review or thesis appendices.
        </p>
      </div>

      {/* Export Action Buttons */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {/* CSV Download */}
        <a
          href="/api/reports/download-benchmark-csv"
          download="adapt_edge_phase5_results.csv"
          className="glass-panel p-5 rounded-xl border border-gray-800 hover:border-cyan-500/50 transition-all flex items-center justify-between group"
        >
          <div className="space-y-1">
            <span className="text-xs font-mono text-cyan-400 uppercase">Benchmark Data</span>
            <div className="text-sm font-bold text-white group-hover:text-cyan-300">Download CSV</div>
            <p className="text-[11px] text-gray-400">1,000 per-sample evaluation records</p>
          </div>
          <FileSpreadsheet className="w-6 h-6 text-cyan-400 group-hover:scale-110 transition-transform" />
        </a>

        {/* Copy Report */}
        <button
          onClick={copyToClipboard}
          className="glass-panel p-5 rounded-xl border border-gray-800 hover:border-indigo-500/50 transition-all text-left flex items-center justify-between group cursor-pointer"
        >
          <div className="space-y-1">
            <span className="text-xs font-mono text-indigo-400 uppercase">Academic Text</span>
            <div className="text-sm font-bold text-white group-hover:text-indigo-300">
              {copied ? 'Copied to Clipboard!' : 'Copy Markdown'}
            </div>
            <p className="text-[11px] text-gray-400">IEEE project report formatting</p>
          </div>
          <Copy className="w-6 h-6 text-indigo-400 group-hover:scale-110 transition-transform" />
        </button>

        {/* Print / Save PDF */}
        <button
          onClick={() => window.print()}
          className="glass-panel p-5 rounded-xl border border-gray-800 hover:border-emerald-500/50 transition-all text-left flex items-center justify-between group cursor-pointer"
        >
          <div className="space-y-1">
            <span className="text-xs font-mono text-emerald-400 uppercase">Print & Archive</span>
            <div className="text-sm font-bold text-white group-hover:text-emerald-300">Print / PDF</div>
            <p className="text-[11px] text-gray-400">Print-ready executive summary</p>
          </div>
          <Printer className="w-6 h-6 text-emerald-400 group-hover:scale-110 transition-transform" />
        </button>
      </div>

      {/* Report Preview */}
      <div className="glass-panel p-6 rounded-xl border border-gray-800 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold text-white tracking-wide flex items-center gap-2">
            <FileText className="w-4 h-4 text-cyan-400" />
            Generated Academic Capstone Research Report
          </h3>
          <span className="text-xs font-mono text-gray-400">Markdown Format</span>
        </div>

        <div className="p-5 rounded-xl bg-gray-900/90 border border-gray-800 overflow-x-auto max-h-[500px] overflow-y-auto">
          <pre className="font-mono text-xs text-gray-300 whitespace-pre-wrap leading-relaxed">
            {markdown || 'Loading research report...'}
          </pre>
        </div>
      </div>
    </div>
  );
}

