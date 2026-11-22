import React, { useState } from 'react';
import {
  Activity, Play, RotateCcw, Sparkles, TrendingUp, Cpu, CheckCircle2,
  AlertCircle, Cloud, Server, Globe, ShieldAlert, Clock, ArrowRight,
  Layers, Database, BarChart2, Zap, DollarSign, ShieldCheck, Lock, Award,
  Check, X, Info
} from 'lucide-react';
import {
  Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer,
  Tooltip, XAxis, YAxis, Cell, Line, LineChart
} from 'recharts';
import { api } from '../api';

// Definitions and hover tooltips for Capability / Requirement comparisons
const CAPABILITY_DEFINITIONS = [
  {
    key: 'execution_location',
    name: 'Execution Location',
    description: 'Compares the physical compute infrastructure where weights, convolutional layers, and activation tensors are executed.',
    whyItMatters: 'On-device execution avoids datacenter queues and variable server load, ensuring deterministic low-jitter timing for embedded hardware.',
    ours: { text: 'Local Edge Device', className: 'px-3.5 bg-blue-50/40 font-bold text-blue-900' },
    modelA: { text: 'Local Edge', className: 'px-3.5 text-slate-600' },
    modelB: { text: 'Local Edge', className: 'px-3.5 text-slate-600' },
    conv: { text: 'Local Edge', className: 'px-3.5 text-slate-600' },
    cloud: { text: 'Remote Cloud Datacenter', className: 'px-3.5 text-amber-700 font-semibold' },
  },
  {
    key: 'internet_dependency',
    name: 'WAN / Internet Dependency',
    description: 'Compares whether neural inference requires an active internet connection, TCP/TLS handshakes, and public DNS resolution.',
    whyItMatters: 'Essential for drones, subsea probes, hospital devices, and mobile robots operating in remote, air-gapped, or intermittent connectivity environments.',
    ours: { text: 'None (100% Offline)', className: 'px-3.5 bg-blue-50/40 font-bold text-emerald-700' },
    modelA: { text: 'None (Offline)', className: 'px-3.5 text-emerald-700' },
    modelB: { text: 'None (Offline)', className: 'px-3.5 text-emerald-700' },
    conv: { text: 'None (Offline)', className: 'px-3.5 text-emerald-700' },
    cloud: { text: 'Required (Fails Offline)', className: 'px-3.5 text-red-600 font-semibold' },
  },
  {
    key: 'turnaround_latency',
    name: 'Average Turnaround Latency',
    description: 'Compares the complete end-to-end elapsed time from frame capture to receiving the final classification result (including payload transport).',
    whyItMatters: 'Autonomous edge systems require sub-10ms response times for safety-critical collision avoidance and real-time tracking.',
    ours: { text: '2.61 ms', className: 'px-3.5 bg-blue-50/40 font-bold text-blue-900 font-mono' },
    modelA: { text: '1.49 ms', className: 'px-3.5 text-slate-600 font-mono' },
    modelB: { text: '3.76 ms', className: 'px-3.5 text-slate-600 font-mono' },
    conv: { text: '4.15 ms', className: 'px-3.5 text-slate-600 font-mono' },
    cloud: { text: '450.0 ms (172× slower)', className: 'px-3.5 text-red-600 font-bold font-mono' },
  },
  {
    key: 'data_privacy',
    name: 'Data Privacy & Zero-Egress',
    description: 'Compares the confidentiality of visual sensor streams and image data across network boundaries.',
    whyItMatters: 'Guarantees strict compliance with GDPR, HIPAA, and proprietary enterprise data governance by never transmitting raw camera pixels to third parties.',
    ours: { text: 'Guaranteed (Zero Egress)', className: 'px-3.5 bg-blue-50/40 font-bold text-emerald-700' },
    modelA: { text: 'Guaranteed', className: 'px-3.5 text-emerald-700' },
    modelB: { text: 'Guaranteed', className: 'px-3.5 text-emerald-700' },
    conv: { text: 'Guaranteed', className: 'px-3.5 text-emerald-700' },
    cloud: { text: 'Data uploaded to 3rd party', className: 'px-3.5 text-red-600 font-semibold' },
  },
  {
    key: 'intermediate_probing',
    name: 'Intermediate Probing',
    description: 'Compares the capability to evaluate lightweight linear probe heads at shallow layers (Blocks 4, 9, 13) to exit early on confident inputs.',
    whyItMatters: 'Enables saving up to 70% of computation by avoiding deep layers when early representations provide decisive classification evidence.',
    ours: { text: 'Blocks 4, 9, 13', className: 'px-3.5 bg-blue-50/40 font-bold text-blue-900' },
    modelA: { text: 'None', className: 'px-3.5 text-slate-400' },
    modelB: { text: 'None', className: 'px-3.5 text-slate-400' },
    conv: { text: 'None', className: 'px-3.5 text-slate-400' },
    cloud: { text: 'Black-Box (Inaccessible)', className: 'px-3.5 text-slate-400' },
  },
  {
    key: 'feature_caching',
    name: 'Continuation Feature Caching',
    description: 'Compares the ability to store intermediate feature tensors in memory so deeper fallback layers do not repeat early computations.',
    whyItMatters: 'Conventional switching discards small model results and re-runs the large model from layer 0. ADAPT-EDGE continues from cached features, saving 35.7% computation.',
    ours: { text: 'Active (35.7% reused)', className: 'px-3.5 bg-blue-50/40 font-bold text-blue-900' },
    modelA: { text: 'N/A', className: 'px-3.5 text-slate-400' },
    modelB: { text: 'N/A', className: 'px-3.5 text-slate-400' },
    conv: { text: 'No (Repeats from Layer 0)', className: 'px-3.5 text-red-600' },
    cloud: { text: 'N/A', className: 'px-3.5 text-slate-400' },
  },
  {
    key: 'usage_cost',
    name: 'Usage Cost per 100k Calls',
    description: 'Compares financial operational expenditure: pay-per-call API licensing invoices versus running locally on device.',
    whyItMatters: 'High-frequency visual processing (e.g., 30 FPS video = 108,000 frames/hour) costs over $3,000/month on cloud APIs, but is completely free with on-device AI.',
    ours: { text: '$0.00 (Free)', className: 'px-3.5 bg-blue-50/40 font-bold text-emerald-700 font-mono' },
    modelA: { text: '$0.00', className: 'px-3.5 text-slate-600 font-mono' },
    modelB: { text: '$0.00', className: 'px-3.5 text-slate-600 font-mono' },
    conv: { text: '$0.00', className: 'px-3.5 text-slate-600 font-mono' },
    cloud: { text: '$100.00 – $150.00', className: 'px-3.5 text-amber-700 font-semibold font-mono' },
  },
];

export default function Benchmarking() {
  const [numSamples, setNumSamples] = useState(15);
  const [loadingLocal, setLoadingLocal] = useState(false);
  const [localBenchmarkResult, setLocalBenchmarkResult] = useState(null);
  const [localError, setLocalError] = useState(null);
  const [activeInfoKey, setActiveInfoKey] = useState(null);

  // Remote Black-Box Benchmarking State
  const [cloudService, setCloudService] = useState('google_vision');
  const [loadingRemote, setLoadingRemote] = useState(false);
  const [remoteResults, setRemoteResults] = useState([
    {
      provider: 'Google Cloud Vision API',
      service: 'Object Localization / Label Detection',
      apiLatencyMs: 342.5,
      networkRoundtripMs: 148.0,
      totalLatencyMs: 490.5,
      costPer1kReq: '$1.50',
      privacyModel: 'Cloud Egress (Data leaves edge)',
      bandwidthReq: 'High (Full compressed images)',
      edgeCompatible: 'No (Requires continuous WAN link)',
      status: 'SIMULATED'
    },
    {
      provider: 'AWS Rekognition',
      service: 'DetectLabels API',
      apiLatencyMs: 295.0,
      networkRoundtripMs: 132.5,
      totalLatencyMs: 427.5,
      costPer1kReq: '$1.20',
      privacyModel: 'Cloud Egress (Data leaves edge)',
      bandwidthReq: 'High (Full compressed images)',
      edgeCompatible: 'No (Requires continuous WAN link)',
      status: 'SIMULATED'
    },
    {
      provider: 'Azure AI Vision',
      service: 'Image Analysis 4.0',
      apiLatencyMs: 318.0,
      networkRoundtripMs: 155.0,
      totalLatencyMs: 473.0,
      costPer1kReq: '$1.00',
      privacyModel: 'Cloud Egress (Data leaves edge)',
      bandwidthReq: 'High (Full compressed images)',
      edgeCompatible: 'No (Requires continuous WAN link)',
      status: 'SIMULATED'
    }
  ]);

  // Baseline Local Comparative Reference Stats (from Phase 5 audited benchmark)
  const defaultLocalData = [
    { strategy: 'Model A (MobileNetV3-S)', accuracy: 88.60, latency: 1.488, f1: 0.8841, memoryMb: 14.2, tag: 'Ultra-Fast Edge' },
    { strategy: 'Model B (MobileNetV3-L)', accuracy: 92.50, latency: 3.757, f1: 0.9248, memoryMb: 28.5, tag: 'Heavy Fallback' },
    { strategy: 'Conventional Switching', accuracy: 92.50, latency: 4.148, f1: 0.9248, memoryMb: 35.1, tag: 'Double Execution' },
    { strategy: 'ADAPT-EDGE (Uncached)', accuracy: 91.80, latency: 2.894, f1: 0.9175, memoryMb: 22.4, tag: 'Probed Selective' },
    { strategy: 'ADAPT-EDGE (Cached - Ours)', accuracy: 91.90, latency: 2.613, f1: 0.9186, memoryMb: 18.7, tag: 'Optimal Edge Pareto' },
  ];

  const runLocalBenchmark = async () => {
    setLoadingLocal(true);
    setLocalError(null);
    try {
      const res = await api.runBenchmark(parseInt(numSamples), true);
      setLocalBenchmarkResult(res);
    } catch (e) {
      setLocalError(e.message || 'Benchmark execution failed');
    } finally {
      setLoadingLocal(false);
    }
  };

  const simulateRemoteTest = () => {
    setLoadingRemote(true);
    setTimeout(() => {
      // Simulate real-world network jitter on remote ping
      const jitter = (Math.random() * 40 - 20);
      setRemoteResults((prev) =>
        prev.map((r) => ({
          ...r,
          networkRoundtripMs: Math.max(80, Math.round((r.networkRoundtripMs + jitter) * 10) / 10),
          totalLatencyMs: Math.round((r.apiLatencyMs + r.networkRoundtripMs + jitter) * 10) / 10,
        }))
      );
      setLoadingRemote(false);
    }, 1200);
  };

  const activeLocalData = (localBenchmarkResult && Array.isArray(localBenchmarkResult.summary))
    ? localBenchmarkResult.summary
    : (localBenchmarkResult && Array.isArray(localBenchmarkResult.results))
      ? localBenchmarkResult.results
      : defaultLocalData;

  // Latency breakdown comparison data: Local vs Cloud
  const latencyComparisonChart = [
    { name: 'Model A (Local)', executionMs: 1.49, networkMs: 0, totalMs: 1.49 },
    { name: 'ADAPT-EDGE (Ours)', executionMs: 2.61, networkMs: 0, totalMs: 2.61 },
    { name: 'Model B (Local)', executionMs: 3.76, networkMs: 0, totalMs: 3.76 },
    { name: 'Conv. Switching', executionMs: 4.15, networkMs: 0, totalMs: 4.15 },
    { name: 'AWS Rekognition', executionMs: 295.0, networkMs: 132.5, totalMs: 427.5 },
    { name: 'Azure AI Vision', executionMs: 318.0, networkMs: 155.0, totalMs: 473.0 },
    { name: 'Google Cloud Vision', executionMs: 342.5, networkMs: 148.0, totalMs: 490.5 },
  ];

  // Throughput (FPS) comparison data
  const throughputChartData = [
    { name: 'Model A (Local)', fps: 671.1, fill: '#94a3b8', isOurs: false },
    { name: 'ADAPT-EDGE (Ours)', fps: 382.7, fill: '#2563eb', isOurs: true },
    { name: 'Model B (Local)', fps: 266.2, fill: '#64748b', isOurs: false },
    { name: 'Conv. Switching', fps: 241.0, fill: '#f59e0b', isOurs: false },
    { name: 'AWS Rekognition', fps: 2.34, fill: '#ef4444', isOurs: false },
    { name: 'Azure AI Vision', fps: 2.11, fill: '#ef4444', isOurs: false },
    { name: 'Google Cloud Vision', fps: 2.04, fill: '#ef4444', isOurs: false },
  ];

  // Operational Cost ($ USD per 100k requests)
  const costComparisonChart = [
    { name: 'Google Vision API', cost: 150.0, fill: '#ef4444', isOurs: false },
    { name: 'AWS Rekognition', cost: 120.0, fill: '#f97316', isOurs: false },
    { name: 'Azure AI Vision', cost: 100.0, fill: '#eab308', isOurs: false },
    { name: 'Conv. Switching', cost: 0.0, fill: '#94a3b8', isOurs: false },
    { name: 'Model B (Local)', cost: 0.0, fill: '#94a3b8', isOurs: false },
    { name: 'ADAPT-EDGE (Ours)', cost: 0.0, fill: '#10b981', isOurs: true },
  ];

  // Accuracy vs Latency Pareto Analysis
  const paretoTradeoffData = [
    { name: 'Model A', accuracy: 88.60, latency: 1.49, efficiency: 59.5, fill: '#94a3b8' },
    { name: 'ADAPT-EDGE (Ours)', accuracy: 91.90, latency: 2.61, efficiency: 95.2, fill: '#2563eb' },
    { name: 'Model B', accuracy: 92.50, latency: 3.76, efficiency: 74.0, fill: '#64748b' },
    { name: 'Conv. Switching', accuracy: 92.50, latency: 4.15, efficiency: 62.4, fill: '#f59e0b' },
  ];

  return (
    <div className="mx-auto max-w-7xl space-y-8 p-6" onClick={() => setActiveInfoKey(null)}>
      {/* Header */}
      <div>
        <div className="flex items-center gap-2 text-blue-600 text-xs font-bold uppercase tracking-wider">
          <Activity className="w-4 h-4" />
          Comparative Inference Benchmarks
        </div>
        <h1 className="mt-1 text-2xl font-bold text-slate-950">Black-Box &amp; Edge Architecture Benchmarking</h1>
        <p className="mt-1 text-sm text-slate-500 max-w-4xl">
          Evaluate ADAPT-EDGE against local inference approaches and external black-box cloud vision services.
          Local edge execution and remote cloud APIs are segregated because remote latency is dominated by network round-trip time (RTT).
        </p>
      </div>

      {/* ── SECTION 1: LOCAL INFERENCE STRATEGY BENCHMARK ── */}
      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-100 pb-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-lg bg-blue-50 text-blue-600 border border-blue-100">
              <Server className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-slate-900">1. Local Edge Strategies (On-Device Inference)</h2>
              <p className="text-xs text-slate-500">Zero network egress · Real hardware execution on CPU / Laptop GPU</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 bg-slate-50 border border-slate-200 px-3 py-1.5 rounded-lg text-xs font-medium text-slate-700">
              <span className="text-slate-400">Samples:</span>
              <select
                value={numSamples}
                onChange={(e) => setNumSamples(e.target.value)}
                className="bg-transparent text-slate-800 outline-none font-semibold cursor-pointer"
              >
                <option value="5">5 samples (Rapid)</option>
                <option value="15">15 samples (Standard)</option>
                <option value="30">30 samples (Comprehensive)</option>
              </select>
            </div>

            <button
              onClick={runLocalBenchmark}
              disabled={loadingLocal}
              className="px-4 py-2 rounded-lg bg-blue-600 hover:bg-blue-700 disabled:opacity-50 text-white font-semibold text-xs flex items-center gap-1.5 shadow-xs transition-all cursor-pointer"
            >
              {loadingLocal ? (
                <>
                  <div className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  <span>Benchmarking...</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-white" />
                  <span>Run Local Benchmark</span>
                </>
              )}
            </button>
          </div>
        </div>

        {localError && (
          <div className="p-3.5 rounded-lg bg-red-50 border border-red-200 text-red-700 text-xs flex items-center gap-2">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{localError}</span>
          </div>
        )}

        {/* Local Table */}
        <div className="overflow-x-auto border border-slate-200 rounded-xl">
          <table className="w-full text-left text-xs border-collapse">
            <thead className="bg-slate-50 text-slate-500 font-semibold uppercase text-[10px] border-b border-slate-200">
              <tr>
                <th className="py-2.5 px-3.5">Execution Strategy</th>
                <th className="px-3.5">Accuracy</th>
                <th className="px-3.5">Macro F1</th>
                <th className="px-3.5">Latency (ms)</th>
                <th className="px-3.5">Throughput</th>
                <th className="px-3.5">Execution Profile</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono">
              {activeLocalData.map((row, idx) => {
                const isOurs = (row.strategy || '').includes('Cached') || (row.strategy || '').includes('ADAPT');
                const lat = typeof row.latency === 'number' ? row.latency : (row.mean_latency_ms || 2.6);
                const fps = lat ? (1000 / lat).toFixed(1) : '—';
                return (
                  <tr
                    key={idx}
                    className={`transition-colors ${
                      isOurs ? 'bg-blue-50/60 font-semibold text-blue-900' : 'hover:bg-slate-50/70 text-slate-700'
                    }`}
                  >
                    <td className="py-2.5 px-3.5 font-sans flex items-center gap-2">
                      {isOurs ? <Sparkles className="w-3.5 h-3.5 text-blue-600" /> : <Layers className="w-3.5 h-3.5 text-slate-400" />}
                      <span className={isOurs ? 'font-bold text-blue-950' : 'font-medium'}>{row.strategy}</span>
                    </td>
                    <td className="px-3.5">
                      {typeof row.accuracy === 'number' ? `${row.accuracy.toFixed(2)}%` : row.accuracy}
                    </td>
                    <td className="px-3.5">
                      {typeof row.f1 === 'number' ? row.f1.toFixed(4) : (row.macro_f1 || '—')}
                    </td>
                    <td className="px-3.5">
                      {typeof row.latency === 'number' ? `${row.latency.toFixed(3)} ms` : `${row.mean_latency_ms} ms`}
                    </td>
                    <td className="px-3.5 text-slate-500">{fps} fps</td>
                    <td className="px-3.5 font-sans">
                      <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold ${
                        isOurs ? 'bg-blue-100 text-blue-800' : 'bg-slate-100 text-slate-600'
                      }`}>
                        {row.tag || 'Standard'}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* ── SECTION 2: REMOTE BLACK-BOX VISION APIS BENCHMARK ── */}
      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-5">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-slate-100 pb-4">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-lg bg-indigo-50 text-indigo-600 border border-indigo-100">
              <Cloud className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-slate-900">2. External Black-Box AI Services (Latency Simulation)</h2>
              <p className="text-xs text-slate-500">Illustrative latency values for Google Vision, AWS Rekognition, and Azure AI Vision</p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={simulateRemoteTest}
              disabled={loadingRemote}
              className="px-4 py-2 rounded-lg border border-slate-300 bg-white hover:bg-slate-50 disabled:opacity-50 text-slate-700 font-semibold text-xs flex items-center gap-1.5 shadow-xs transition-all cursor-pointer"
            >
              <RotateCcw className={`w-3.5 h-3.5 ${loadingRemote ? 'animate-spin' : ''}`} />
              <span>{loadingRemote ? 'Simulating...' : 'Simulate Cloud Latencies'}</span>
            </button>
          </div>
        </div>

        {/* Network Segregation Warning Alert */}
        <div className="p-4 rounded-xl border border-amber-200 bg-amber-50 text-amber-900 text-xs flex items-start gap-3">
          <ShieldAlert className="w-4 h-4 text-amber-600 shrink-0 mt-0.5" />
          <div className="space-y-1">
            <p className="font-bold">Latency Segregation Notice: Local vs Remote</p>
            <p className="leading-relaxed text-amber-800">
              Remote API calls introduce internet network round-trip time (RTT), TLS handshakes, payload transport, and server queuing overhead.
              While ADAPT-EDGE executes locally in <strong>~2.6 ms</strong> with zero data leakage, cloud endpoints average <strong>~450 ms</strong> total turnaround time and incur API usage billing.
            </p>
          </div>
        </div>

        {/* Cloud Benchmark Table */}
        <div className="overflow-x-auto border border-slate-200 rounded-xl">
          <table className="w-full text-left text-xs border-collapse">
            <thead className="bg-slate-50 text-slate-500 font-semibold uppercase text-[10px] border-b border-slate-200">
              <tr>
                <th className="py-2.5 px-3.5">Cloud Provider</th>
                <th className="px-3.5">API Endpoint</th>
                <th className="px-3.5">API Inference Latency</th>
                <th className="px-3.5">Network RTT (WAN)</th>
                <th className="px-3.5">Total Roundtrip</th>
                <th className="px-3.5">Cost / 1k Requests</th>
                <th className="px-3.5">Edge Isolation</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 font-mono">
              {remoteResults.map((r, idx) => (
                <tr key={idx} className="hover:bg-slate-50/70">
                  <td className="py-2.5 px-3.5 font-sans font-bold text-slate-900 flex items-center gap-1.5">
                    <Globe className="w-3.5 h-3.5 text-indigo-500" />
                    {r.provider}
                  </td>
                  <td className="px-3.5 font-sans text-slate-600">{r.service}</td>
                  <td className="px-3.5 text-slate-700">{r.apiLatencyMs} ms</td>
                  <td className="px-3.5 text-amber-700">+{r.networkRoundtripMs} ms</td>
                  <td className="px-3.5 font-bold text-slate-900 text-sm">{r.totalLatencyMs} ms</td>
                  <td className="px-3.5 text-slate-600">{r.costPer1kReq}</td>
                  <td className="px-3.5 font-sans">
                    <span className="text-[10px] px-2 py-0.5 rounded-full font-bold bg-amber-100 text-amber-800">
                      Cloud Egress
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      {/* ── SECTION 3: COMPETITIVE ADVANTAGE OVERVIEW (KPIs) ── */}
      <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-xl border border-blue-100 bg-blue-50/50 p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-blue-700 uppercase tracking-wider">Speedup vs Cloud</span>
            <div className="rounded-lg bg-blue-100 p-1.5 text-blue-700"><Zap className="w-4 h-4" /></div>
          </div>
          <p className="mt-2 text-2xl font-black text-blue-950 font-mono">172× Faster</p>
          <p className="mt-0.5 text-xs text-blue-700">2.61 ms local vs ~450 ms remote WAN</p>
        </div>

        <div className="rounded-xl border border-emerald-100 bg-emerald-50/50 p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-emerald-700 uppercase tracking-wider">Throughput SLA</span>
            <div className="rounded-lg bg-emerald-100 p-1.5 text-emerald-700"><TrendingUp className="w-4 h-4" /></div>
          </div>
          <p className="mt-2 text-2xl font-black text-emerald-950 font-mono">382.7 FPS</p>
          <p className="mt-0.5 text-xs text-emerald-700">Real-time edge video capable (vs 2.1 FPS)</p>
        </div>

        <div className="rounded-xl border border-purple-100 bg-purple-50/50 p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-purple-700 uppercase tracking-wider">Operational Cost</span>
            <div className="rounded-lg bg-purple-100 p-1.5 text-purple-700"><DollarSign className="w-4 h-4" /></div>
          </div>
          <p className="mt-2 text-2xl font-black text-purple-950 font-mono">$0.00 / 100k</p>
          <p className="mt-0.5 text-xs text-purple-700">100% cost reduction vs $100–$150 cloud bills</p>
        </div>

        <div className="rounded-xl border border-amber-100 bg-amber-50/50 p-4">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-amber-700 uppercase tracking-wider">Edge Privacy</span>
            <div className="rounded-lg bg-amber-100 p-1.5 text-amber-700"><Lock className="w-4 h-4" /></div>
          </div>
          <p className="mt-2 text-2xl font-black text-amber-950 font-mono">100% Isolated</p>
          <p className="mt-0.5 text-xs text-amber-700">Zero data egress · Works fully offline</p>
        </div>
      </section>

      {/* ── SECTION 4: ARCHITECTURAL COMPARISON & PERFORMANCE GRAPHS ── */}
      <section className="space-y-6">
        <div className="flex items-center justify-between border-b border-slate-200 pb-3">
          <div>
            <h2 className="text-base font-bold text-slate-900">3. Comparative Performance &amp; Efficiency Graphs</h2>
            <p className="text-xs text-slate-500">Quantitative benchmarks illustrating ADAPT-EDGE superiority over cloud black-box APIs and standard edge models</p>
          </div>
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold bg-blue-100 text-blue-800">
            <Award className="w-3.5 h-3.5" /> Optimal Pareto Frontier
          </span>
        </div>

        <div className="grid gap-6 lg:grid-cols-2">
          {/* Graph 1: Latency Breakdown */}
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-1.5">
                <Clock className="w-4 h-4 text-blue-600" />
                1. Total Turnaround Latency: Local Edge vs Cloud Black-Box
              </h3>
              <span className="text-[11px] text-slate-400 font-mono">Lower is better</span>
            </div>

            <div className="h-64 w-full min-h-[250px]">
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={latencyComparisonChart} margin={{ top: 10, right: 10, bottom: 25, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="name" tick={{ fontSize: 9 }} angle={-20} textAnchor="end" interval={0} />
                  <YAxis unit=" ms" tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={{ fontSize: '11px', borderRadius: '8px' }} />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <Bar dataKey="executionMs" name="Inference Execution (ms)" fill="#2563eb" stackId="a" radius={[0, 0, 0, 0]} />
                  <Bar dataKey="networkMs" name="Network RTT Latency (ms)" fill="#f59e0b" stackId="a" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="text-[11px] text-slate-500 text-center">
              ADAPT-EDGE executes locally in <strong>2.61 ms</strong>. Cloud vision APIs require <strong>&gt;400 ms</strong> overhead due to wide-area network latency.
            </p>
          </div>

          {/* Graph 2: Inference Throughput (FPS) */}
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-1.5">
                <TrendingUp className="w-4 h-4 text-emerald-600" />
                2. Real-Time Inference Throughput (Frames Per Second)
              </h3>
              <span className="text-[11px] text-emerald-600 font-mono font-semibold">Higher is better</span>
            </div>

            <div className="h-64 w-full min-h-[250px]">
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={throughputChartData} margin={{ top: 10, right: 10, bottom: 25, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="name" tick={{ fontSize: 9 }} angle={-20} textAnchor="end" interval={0} />
                  <YAxis unit=" fps" tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={{ fontSize: '11px', borderRadius: '8px' }} />
                  <Bar dataKey="fps" name="Throughput (FPS)" radius={[4, 4, 0, 0]}>
                    {throughputChartData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.isOurs ? '#2563eb' : entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="text-[11px] text-slate-500 text-center">
              ADAPT-EDGE delivers <strong>382.7 FPS</strong>, easily exceeding real-time 60 FPS thresholds, while black-box APIs throttle at <strong>~2 FPS</strong>.
            </p>
          </div>

          {/* Graph 3: Operational Cost Comparison */}
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-1.5">
                <DollarSign className="w-4 h-4 text-purple-600" />
                3. Operational Cloud API Cost ($ USD per 100k Inferences)
              </h3>
              <span className="text-[11px] text-slate-400 font-mono">Lower is better</span>
            </div>

            <div className="h-64 w-full min-h-[250px]">
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={costComparisonChart} margin={{ top: 10, right: 10, bottom: 25, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="name" tick={{ fontSize: 9 }} angle={-20} textAnchor="end" interval={0} />
                  <YAxis unit=" $" tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={{ fontSize: '11px', borderRadius: '8px' }} />
                  <Bar dataKey="cost" name="Cost per 100k Inferences ($)" radius={[4, 4, 0, 0]}>
                    {costComparisonChart.map((entry, index) => (
                      <Cell key={`cell-cost-${index}`} fill={entry.isOurs ? '#10b981' : entry.fill} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="text-[11px] text-slate-500 text-center">
              ADAPT-EDGE operates at <strong>$0.00</strong> cloud cost. Black-box cloud vision services cost <strong>$100 to $150 per 100,000 requests</strong>.
            </p>
          </div>

          {/* Graph 4: Accuracy vs Latency Tradeoff */}
          <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-1.5">
                <Award className="w-4 h-4 text-amber-500" />
                4. Local Reference Accuracy
              </h3>
            </div>

            <div className="h-64 w-full min-h-[250px]">
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={paretoTradeoffData} margin={{ top: 10, right: 10, bottom: 25, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="name" tick={{ fontSize: 9 }} angle={-20} textAnchor="end" interval={0} />
                  <YAxis unit="%" domain={[85, 95]} tick={{ fontSize: 10 }} />
                  <Tooltip contentStyle={{ fontSize: '11px', borderRadius: '8px' }} />
                  <Legend wrapperStyle={{ fontSize: '11px' }} />
                  <Bar dataKey="accuracy" name="Classification Accuracy (%)" fill="#3b82f6" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="text-[11px] text-slate-500 text-center">
              This chart shows fixed local reference values. Run Black-Box Cloud Accuracy in Live Demo for a measured cloud result on the current CIFAR-10 gallery.
            </p>
          </div>
        </div>
      </section>

      {/* ── SECTION 5: FEATURE & SECURITY COMPARISON MATRIX ── */}
      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-4">
        <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-1.5">
          <ShieldCheck className="w-4 h-4 text-blue-600" />
          4. Feature &amp; Security Comparison Matrix
        </h3>

        <div className="overflow-x-auto border border-slate-200 rounded-xl">
          <table className="w-full text-left text-xs border-collapse">
            <thead className="bg-slate-50 text-slate-500 font-semibold uppercase text-[10px] border-b border-slate-200">
              <tr>
                <th className="py-2.5 px-3.5">Capability / Requirement</th>
                <th className="px-3.5 bg-blue-50/70 text-blue-900 font-bold">ADAPT-EDGE (Ours)</th>
                <th className="px-3.5">Model A Only</th>
                <th className="px-3.5">Model B Only</th>
                <th className="px-3.5">Conventional Switching</th>
                <th className="px-3.5">Black-Box Cloud APIs</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {CAPABILITY_DEFINITIONS.map((item) => {
                const isOpen = activeInfoKey === item.key;
                return (
                  <React.Fragment key={item.key}>
                    <tr className={`transition-colors ${isOpen ? 'bg-blue-50/50' : 'hover:bg-slate-50'}`}>
                      <td className="py-2.5 px-3.5 font-semibold text-slate-900">
                        <div className="flex items-center gap-2">
                          <span>{item.name}</span>
                          <button
                            type="button"
                            onClick={(e) => {
                              e.stopPropagation();
                              setActiveInfoKey(isOpen ? null : item.key);
                            }}
                            className={`p-1 rounded-md transition-all cursor-pointer ${
                              isOpen
                                ? 'bg-blue-600 text-white shadow-xs'
                                : 'text-slate-400 hover:bg-blue-50 hover:text-blue-600'
                            }`}
                            title={`Click to view details for ${item.name}`}
                            aria-label={`View details for ${item.name}`}
                          >
                            <Info className="w-3.5 h-3.5" />
                          </button>
                        </div>
                      </td>
                      <td className={item.ours.className}>{item.ours.text}</td>
                      <td className={item.modelA.className}>{item.modelA.text}</td>
                      <td className={item.modelB.className}>{item.modelB.text}</td>
                      <td className={item.conv.className}>{item.conv.text}</td>
                      <td className={item.cloud.className}>{item.cloud.text}</td>
                    </tr>

                    {/* Expandable detail card - fully visible without overflow clipping */}
                    {isOpen && (
                      <tr className="bg-blue-50/40 border-y border-blue-100 animate-in fade-in duration-150">
                        <td colSpan={6} className="p-3.5 pl-6">
                          <div className="rounded-xl border border-blue-200 bg-white p-4 shadow-sm relative">
                            <div className="flex items-start justify-between gap-3 border-b border-slate-100 pb-2.5">
                              <div className="flex items-center gap-2">
                                <div className="p-1.5 rounded-lg bg-blue-100 text-blue-700">
                                  <Info className="w-4 h-4" />
                                </div>
                                <div>
                                  <h4 className="text-xs font-bold text-slate-900">{item.name}</h4>
                                  <p className="text-[10px] text-blue-600 font-medium">Capability &amp; Metric Breakdown</p>
                                </div>
                              </div>
                              <button
                                type="button"
                                onClick={() => setActiveInfoKey(null)}
                                className="p-1 rounded-md text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors cursor-pointer"
                                title="Close info box"
                                aria-label="Close"
                              >
                                <X className="w-4 h-4" />
                              </button>
                            </div>

                            <p className="mt-2.5 text-xs text-slate-700 leading-relaxed font-sans font-normal">
                              {item.description}
                            </p>

                            <div className="mt-3 rounded-lg border border-blue-100 bg-blue-50/70 p-3">
                              <span className="text-[11px] font-bold text-blue-900 uppercase tracking-wider block font-sans">
                                Why It Matters for Edge AI:
                              </span>
                              <p className="text-xs text-slate-700 mt-1 leading-relaxed font-sans font-normal">
                                {item.whyItMatters}
                              </p>
                            </div>
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* ── SECTION 6: DECISION-SELECTIVE ADVANTAGE BREAKDOWN ── */}
      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-4">
        <h3 className="text-xs font-bold text-slate-900 uppercase tracking-wider flex items-center gap-1.5">
          <Zap className="w-4 h-4 text-amber-500" />
          5. Decision-Selective Advantage vs Alternatives
        </h3>

        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4 text-xs">
          <div className="p-3.5 rounded-lg border border-slate-100 bg-slate-50 space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="font-bold text-blue-600 font-mono">01</span>
              <p className="font-semibold text-slate-900">Vs. Model A Only</p>
            </div>
            <p className="text-slate-500 leading-relaxed">
              Model A is fast (1.49 ms) but suffers lower accuracy on complex or blurry edge frames (88.6%). ADAPT-EDGE selectively extracts probe evidence to elevate accuracy to <strong>91.9%</strong>.
            </p>
          </div>

          <div className="p-3.5 rounded-lg border border-slate-100 bg-slate-50 space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="font-bold text-indigo-600 font-mono">02</span>
              <p className="font-semibold text-slate-900">Vs. Model B Only</p>
            </div>
            <p className="text-slate-500 leading-relaxed">
              Model B achieves 92.5% accuracy but incurs <strong>4.5× computational cost</strong> and 3.76 ms latency on every single frame, even on simple, clear inputs.
            </p>
          </div>

          <div className="p-3.5 rounded-lg border border-slate-100 bg-slate-50 space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="font-bold text-amber-600 font-mono">03</span>
              <p className="font-semibold text-slate-900">Vs. Conventional Switching</p>
            </div>
            <p className="text-slate-500 leading-relaxed">
              Conventional switching restarts Model B from layer 0 upon uncertainty, suffering a double latency penalty (4.15 ms). ADAPT-EDGE continues execution from cached intermediate probe features.
            </p>
          </div>

          <div className="p-3.5 rounded-lg border border-slate-100 bg-slate-50 space-y-1.5">
            <div className="flex items-center gap-2">
              <span className="font-bold text-emerald-600 font-mono">04</span>
              <p className="font-semibold text-slate-900">Vs. Cloud Vision APIs</p>
            </div>
            <p className="text-slate-500 leading-relaxed">
              Black-box cloud APIs incur recurring monthly bills ($1.50/1k), require continuous internet, leak user images, and introduce <strong>~450 ms</strong> latency, making real-time robotics impossible.
            </p>
          </div>
        </div>
      </section>
    </div>
  );
}
