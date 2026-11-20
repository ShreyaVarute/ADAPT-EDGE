import React, { useState } from 'react';
import {
  Workflow,
  Sparkles,
  Info,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  ShieldCheck,
  Zap
} from 'lucide-react';
import StatusBadge from '../components/StatusBadge';

export default function DecisionTrace() {
  const [selectedCase, setSelectedCase] = useState('retain');

  const cases = {
    retain: {
      title: 'Confident Retain Pathway (Model A)',
      scenario: 'High-confidence input with low prediction entropy (e.g., unambiguous automobile image)',
      math: 'Confidence(x) >= tau_high (0.85)',
      action: 'RETAIN Model A prediction immediately. Skip all Model B probe blocks and tail network.',
      latency: '1.488 ms',
      savings: '100% of Model B compute avoided',
      outcome: 'Model A Prediction accepted as final output.'
    },
    probe: {
      title: 'Selective Probing Pathway (Block 4/9/13)',
      scenario: 'Intermediate uncertainty: Model A confidence falls between tau_low and tau_high',
      math: 'tau_low (0.50) <= Confidence(x) < tau_high (0.85)',
      action: 'Execute lightweight intermediate probe heads to collect additional feature representations.',
      latency: '1.85 - 2.40 ms',
      savings: 'Avoids heavy Model B tail if probe resolves uncertainty',
      outcome: 'Probe Head prediction accepted if probe confidence >= tau_probe (0.80).'
    },
    switch: {
      title: 'Continuation Caching Switch Pathway (Model B Tail)',
      scenario: 'High ambiguity: Model A and probes remain uncertain (e.g., subtle bird vs airplane)',
      math: 'Confidence(x) < tau_low OR Probes inconclusive',
      action: 'Escalate to Model B. Retrieve cached Block 13 activations and execute only tail blocks 14-16.',
      latency: '2.613 ms',
      savings: '35.7% of Model B compute eliminated via cached activations',
      outcome: 'Model B full classifier prediction accepted with highest accuracy.'
    },
    thermal: {
      title: 'Thermal & Latency Budget Exhaustion Guard',
      scenario: 'Device running hot (> 80°C) or inference latency budget expired (< 1.5 ms remaining)',
      math: 'GPU_Temp > 80°C OR Remaining_Budget < Exec_Cost',
      action: 'Safety throttle: Force immediate RETAIN of best available prediction to prevent device shutdown.',
      latency: '1.488 ms (Safety Capped)',
      savings: 'Protects hardware lifetime and avoids system stall',
      outcome: 'Safe fallback with explicit RESOURCE_BUDGET_EXHAUSTED audit tag.'
    }
  };

  const c = cases[selectedCase];

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Banner */}
      <div className="glass-panel p-6 rounded-xl border border-gray-800 space-y-2">
        <div className="flex items-center gap-2 text-cyan-400 text-xs font-mono">
          <Workflow className="w-4 h-4" />
          POLICY ENGINE & ROUTING LOGIC
        </div>
        <h2 className="text-xl font-bold text-white">Decision Tracing & Policy Verification</h2>
        <p className="text-xs text-gray-300 max-w-3xl leading-relaxed">
          Inspect the mathematical mechanics behind the ADAPT-EDGE decision controller. Understand how the policy
          guarantees high accuracy while adhering to rigid edge latency and thermal budgets.
        </p>
      </div>

      {/* Case Selector Tabs */}
      <div className="flex flex-wrap gap-2">
        {Object.entries(cases).map(([key, item]) => {
          const isSelected = selectedCase === key;
          return (
            <button
              key={key}
              onClick={() => setSelectedCase(key)}
              className={`px-4 py-2.5 rounded-xl text-xs font-mono transition-all flex items-center gap-2 ${
                isSelected
                  ? 'bg-cyan-500/20 border border-cyan-400 text-cyan-300 shadow-lg shadow-cyan-500/10'
                  : 'bg-gray-900 border border-gray-800 text-gray-400 hover:text-gray-200'
              }`}
            >
              {item.title.split(' ')[0]} {item.title.split(' ')[1]}
            </button>
          );
        })}
      </div>

      {/* Case Details Card */}
      <div className="glass-panel p-6 rounded-xl border border-gray-800 space-y-5">
        <div className="flex items-center justify-between">
          <h3 className="text-base font-bold text-white">{c.title}</h3>
          <StatusBadge status="PASS" label="POLICY VERIFIED" />
        </div>

        <p className="text-xs text-gray-300">{c.scenario}</p>

        <div className="p-4 rounded-xl bg-gray-900/90 border border-gray-800 space-y-3 font-mono text-xs">
          <div className="text-[11px] text-gray-400 uppercase tracking-wider font-semibold">Mathematical Trigger Condition</div>
          <div className="p-2.5 rounded bg-black/60 border border-gray-800 text-cyan-300 font-bold">
            {c.math}
          </div>

          <div className="pt-2 text-gray-300">
            <span className="text-amber-400 font-bold">Controller Policy Action: </span>
            {c.action}
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 font-mono text-xs">
          <div className="p-3 rounded-lg bg-gray-900 border border-gray-800">
            <span className="text-gray-400 text-[10px] uppercase block">Execution Latency</span>
            <span className="text-white font-bold">{c.latency}</span>
          </div>
          <div className="p-3 rounded-lg bg-gray-900 border border-gray-800">
            <span className="text-gray-400 text-[10px] uppercase block">Compute Savings</span>
            <span className="text-emerald-400 font-bold">{c.savings}</span>
          </div>
          <div className="p-3 rounded-lg bg-gray-900 border border-gray-800">
            <span className="text-gray-400 text-[10px] uppercase block">System Outcome</span>
            <span className="text-indigo-300 font-bold">{c.outcome}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

