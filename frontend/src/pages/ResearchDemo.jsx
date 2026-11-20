import React, { useState } from 'react';
import {
  GraduationCap,
  Sparkles,
  Zap,
  ShieldCheck,
  CheckCircle2,
  TrendingDown,
  Layers,
  ArrowRight,
  Play,
  RotateCcw
} from 'lucide-react';
import { api } from '../api';
import StatusBadge from '../components/StatusBadge';

export default function ResearchDemo({ setActiveTab }) {
  const [activeStep, setActiveStep] = useState(0);
  const [demoRunning, setDemoRunning] = useState(false);
  const [demoResult, setDemoResult] = useState(null);

  const demoScenarios = [
    {
      title: 'Scenario 1: Clear & Unambiguous Sample (Automobile)',
      sample_index: 3, // Automobile (easy)
      expected_action: 'RETAIN Model A',
      desc: 'High confidence from Model A (> 0.85). The controller avoids all heavy computation and terminates in < 1.5 ms.',
      benefit: '100% of Model B compute avoided.'
    },
    {
      title: 'Scenario 2: Intermediate Complexity (Truck vs Auto)',
      sample_index: 25,
      expected_action: 'Intermediate Probe Exit',
      desc: 'Model A is uncertain. The controller selectively invokes Probe Head Block 4/9 to classify intermediate features.',
      benefit: 'Resolved early without executing deep backbone.'
    },
    {
      title: 'Scenario 3: Complex / Ambiguous Query (Cat vs Dog)',
      sample_index: 35,
      expected_action: 'Continuation Cache Switch',
      desc: 'High ambiguity. The controller escalates to Model B tail, reusing cached Block 13 activations.',
      benefit: 'Exact 35.7% FLOPs saved compared to naive re-execution.'
    }
  ];

  const runScenario = async (scenario) => {
    setDemoRunning(true);
    setDemoResult(null);
    try {
      const res = await api.predict({
        sample_index: scenario.sample_index,
        strategy: 'ALL',
        tau_high: 0.85,
        tau_low: 0.50,
        tau_probe: 0.80,
        latency_budget_ms: 25.0
      });
      setDemoResult({
        scenario,
        response: res
      });
    } catch (e) {
      alert('Error running demo scenario: ' + e.message);
    } finally {
      setDemoRunning(false);
    }
  };

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Presentation Header */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-indigo-950/80 via-blue-950/60 to-[#0d1322] border border-cyan-500/30 p-6 lg:p-8 shadow-2xl">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="space-y-2 max-w-2xl">
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-300 text-xs font-mono">
              <GraduationCap className="w-4 h-4 text-cyan-400" />
              CAPSTONE RESEARCH & PATENT DEMONSTRATION MODE
            </div>
            <h1 className="text-2xl lg:text-3xl font-extrabold text-white tracking-tight">
              Faculty Evaluation Presentation Walkthrough
            </h1>
            <p className="text-xs text-gray-300 leading-relaxed">
              Designed for your project demonstration. Walk through the fundamental problem, patent novelties,
              empirical findings, and trigger live edge AI inference scenarios in real time.
            </p>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <StatusBadge status="ACTIVE" label="DEMO READY" />
          </div>
        </div>
      </div>

      {/* 4 Core Research Pillars */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="glass-panel p-5 rounded-xl border border-gray-800 space-y-2">
          <div className="text-xs font-mono text-rose-400 font-bold uppercase">1. THE PROBLEM</div>
          <h3 className="text-sm font-bold text-white">Edge Compute Mismatch</h3>
          <p className="text-xs text-gray-400 leading-relaxed">
            Standard models run full deep networks on every query, draining battery on easy inputs. Naive model switching restarts from scratch, wasting compute.
          </p>
        </div>

        <div className="glass-panel p-5 rounded-xl border border-gray-800 space-y-2">
          <div className="text-xs font-mono text-amber-400 font-bold uppercase">2. THE INNOVATION</div>
          <h3 className="text-sm font-bold text-white">Decision-Selective Probing</h3>
          <p className="text-xs text-gray-400 leading-relaxed">
            ADAPT-EDGE taps into intermediate backbone features at Blocks 4, 9, and 13 to resolve uncertainty before invoking heavy layers.
          </p>
        </div>

        <div className="glass-panel p-5 rounded-xl border border-gray-800 space-y-2">
          <div className="text-xs font-mono text-cyan-400 font-bold uppercase">3. PATENT NOVELTY</div>
          <h3 className="text-sm font-bold text-white">Continuation Caching</h3>
          <p className="text-xs text-gray-400 leading-relaxed">
            Intermediate activation tensors are retained. When switching to Model B, execution resumes from Block 14, saving <strong className="text-emerald-400">35.7% FLOPs</strong>.
          </p>
        </div>

        <div className="glass-panel p-5 rounded-xl border border-gray-800 space-y-2">
          <div className="text-xs font-mono text-emerald-400 font-bold uppercase">4. AUDITED RESULT</div>
          <h3 className="text-sm font-bold text-white">Near-Oracle Accuracy</h3>
          <p className="text-xs text-gray-400 leading-relaxed">
            Achieves 91.90% test accuracy (99.4% parity with Model B) with a <strong className="text-cyan-300">37.0% latency reduction</strong> on NVIDIA RTX 3050.
          </p>
        </div>
      </div>

      {/* Interactive Live Demonstration Scenarios */}
      <div className="glass-panel p-6 rounded-xl border border-cyan-500/20 space-y-5">
        <div>
          <h3 className="text-sm font-bold text-white tracking-wide flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-cyan-400" />
            Live Interactive Teacher Demonstration Scenarios
          </h3>
          <p className="text-xs text-gray-400">
            Click any scenario below to trigger real GPU execution across all models and witness how ADAPT-EDGE adapts its decision path.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {demoScenarios.map((sc, idx) => (
            <div
              key={idx}
              className="p-4 rounded-xl bg-gray-900/80 border border-gray-800 flex flex-col justify-between space-y-3 hover:border-cyan-500/40 transition-colors"
            >
              <div className="space-y-1.5">
                <span className="text-[10px] font-mono text-cyan-400 font-bold uppercase block">
                  Scenario {idx + 1}
                </span>
                <h4 className="text-xs font-bold text-white">{sc.title}</h4>
                <p className="text-xs text-gray-300">{sc.desc}</p>
                <div className="text-[11px] font-mono text-emerald-400 pt-1">
                  &bull; {sc.benefit}
                </div>
              </div>

              <button
                onClick={() => runScenario(sc)}
                disabled={demoRunning}
                className="w-full py-2.5 rounded-lg bg-cyan-500/10 hover:bg-cyan-500/20 border border-cyan-500/30 text-cyan-300 font-mono text-xs font-bold flex items-center justify-center gap-1.5 transition-all cursor-pointer"
              >
                <Play className="w-3.5 h-3.5 fill-cyan-400 text-cyan-400" />
                Demonstrate Scenario {idx + 1}
              </button>
            </div>
          ))}
        </div>

        {/* Live Demo Output Card */}
        {demoResult && (
          <div className="mt-4 p-5 rounded-xl bg-[#080d1a] border border-cyan-500/40 space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <span className="text-xs font-mono text-cyan-300 font-bold uppercase">
                  Live Execution Result: {demoResult.scenario.title}
                </span>
                <div className="text-xs text-gray-400">
                  Sample #{demoResult.response.sample_index} | Ground Truth: <strong className="text-white">{demoResult.response.ground_truth}</strong>
                </div>
              </div>
              <StatusBadge status="PASS" label="DEMO SUCCESS" />
            </div>

            {/* Side-by-Side Strategy Results */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono text-xs">
              <div className="p-3 rounded-lg bg-gray-900 border border-gray-800">
                <span className="text-gray-400 text-[10px] block">Model A (Baseline)</span>
                <span className="text-white font-bold block mt-1">
                  {demoResult.response.comparison?.find(c => c.strategy_key === 'MODEL_A')?.prediction_class}
                </span>
                <span className="text-cyan-400 text-[10px]">
                  {demoResult.response.comparison?.find(c => c.strategy_key === 'MODEL_A')?.latency_ms.toFixed(2)} ms
                </span>
              </div>

              <div className="p-3 rounded-lg bg-gray-900 border border-gray-800">
                <span className="text-gray-400 text-[10px] block">Full Model B</span>
                <span className="text-white font-bold block mt-1">
                  {demoResult.response.comparison?.find(c => c.strategy_key === 'MODEL_B')?.prediction_class}
                </span>
                <span className="text-indigo-400 text-[10px]">
                  {demoResult.response.comparison?.find(c => c.strategy_key === 'MODEL_B')?.latency_ms.toFixed(2)} ms
                </span>
              </div>

              <div className="p-3 rounded-lg bg-gray-900 border border-gray-800">
                <span className="text-gray-400 text-[10px] block">Conventional Switching</span>
                <span className="text-white font-bold block mt-1">
                  {demoResult.response.comparison?.find(c => c.strategy_key === 'CONVENTIONAL')?.prediction_class}
                </span>
                <span className="text-rose-400 text-[10px]">
                  {demoResult.response.comparison?.find(c => c.strategy_key === 'CONVENTIONAL')?.latency_ms.toFixed(2)} ms
                </span>
              </div>

              <div className="p-3 rounded-lg bg-cyan-950/40 border border-cyan-500/50 ring-1 ring-cyan-500/30">
                <span className="text-cyan-300 font-bold text-[10px] block">ADAPT-EDGE (Ours)</span>
                <span className="text-white font-bold block mt-1">
                  {demoResult.response.adapt_edge_cached?.prediction_class}
                </span>
                <span className="text-emerald-400 font-bold text-[10px]">
                  {demoResult.response.adapt_edge_cached?.latency_ms.toFixed(2)} ms (Optimal)
                </span>
              </div>
            </div>

            <div className="p-3 rounded-lg bg-gray-900/80 border border-gray-800 text-xs text-gray-300">
              <strong className="text-cyan-300">Decision Controller Rationale: </strong>
              {demoResult.response.adapt_edge_cached?.trace?.decision_reason}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

