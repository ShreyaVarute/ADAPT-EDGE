import React, { useState } from 'react';
import {
  Sliders,
  Sparkles,
  Zap,
  ShieldCheck,
  Save,
  RotateCcw,
  CheckCircle2
} from 'lucide-react';
import StatusBadge from '../components/StatusBadge';

export default function ExperimentConfig() {
  const [preset, setPreset] = useState('balanced');
  const [tauHigh, setTauHigh] = useState(0.85);
  const [tauLow, setTauLow] = useState(0.50);
  const [tauProbe, setTauProbe] = useState(0.80);
  const [budget, setBudget] = useState(25.0);
  const [thermalLimit, setThermalLimit] = useState(80.0);
  const [cacheEnabled, setCacheEnabled] = useState(true);
  const [savedNotice, setSavedNotice] = useState(false);

  const presets = {
    iot: {
      name: 'Ultra-Low Power IoT',
      desc: 'Maximizes battery life and minimizes latency. Accepts slightly lower accuracy in exchange for 85%+ Model A retention.',
      tauHigh: 0.70,
      tauLow: 0.35,
      tauProbe: 0.75,
      budget: 10.0,
      thermalLimit: 75.0,
      cache: true
    },
    balanced: {
      name: 'Balanced Edge AI (Default)',
      desc: 'The recommended operating point for NVIDIA RTX 3050. Yields 91.90% accuracy with 2.61ms mean latency.',
      tauHigh: 0.85,
      tauLow: 0.50,
      tauProbe: 0.80,
      budget: 25.0,
      thermalLimit: 80.0,
      cache: true
    },
    mission: {
      name: 'Mission-Critical Accuracy',
      desc: 'Favors maximum accuracy over energy conservation. Escalates rapidly to Model B if any ambiguity is detected.',
      tauHigh: 0.95,
      tauLow: 0.70,
      tauProbe: 0.88,
      budget: 50.0,
      thermalLimit: 85.0,
      cache: true
    },
    probing: {
      name: 'Aggressive Probe Exploration',
      desc: 'Maximizes reliance on intermediate feature probes (Blocks 4, 9, 13) before resorting to full Model B.',
      tauHigh: 0.90,
      tauLow: 0.30,
      tauProbe: 0.75,
      budget: 30.0,
      thermalLimit: 80.0,
      cache: true
    }
  };

  const applyPreset = (key) => {
    setPreset(key);
    const p = presets[key];
    setTauHigh(p.tauHigh);
    setTauLow(p.tauLow);
    setTauProbe(p.tauProbe);
    setBudget(p.budget);
    setThermalLimit(p.thermalLimit);
    setCacheEnabled(p.cache);
  };

  const handleSave = () => {
    setSavedNotice(true);
    setTimeout(() => setSavedNotice(false), 2500);
  };

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Banner */}
      <div className="glass-panel p-6 rounded-xl border border-gray-800 space-y-2">
        <div className="flex items-center gap-2 text-cyan-400 text-xs font-mono">
          <Sliders className="w-4 h-4" />
          POLICY TUNING & RUNTIME KNOBS
        </div>
        <h2 className="text-xl font-bold text-white">Decision Controller Configuration</h2>
        <p className="text-xs text-gray-300 max-w-3xl leading-relaxed">
          Fine-tune the operational thresholds of the ADAPT-EDGE policy engine. Select a pre-calibrated deployment
          profile or customize confidence boundaries, thermal limits, and continuation caching.
        </p>
      </div>

      {/* Preset Profiles Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {Object.entries(presets).map(([key, p]) => {
          const isSelected = preset === key;
          return (
            <button
              key={key}
              onClick={() => applyPreset(key)}
              className={`p-4 rounded-xl border text-left transition-all ${
                isSelected
                  ? 'bg-cyan-500/15 border-cyan-400 ring-1 ring-cyan-400/40'
                  : 'bg-gray-900/60 border-gray-800 hover:border-gray-700'
              }`}
            >
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-white font-bold">{p.name}</span>
                {isSelected && <StatusBadge status="ACTIVE" size="xs" />}
              </div>
              <p className="mt-2 text-xs text-gray-400 leading-relaxed">{p.desc}</p>
            </button>
          );
        })}
      </div>

      {/* Custom Knobs Panel */}
      <div className="glass-panel p-6 rounded-xl border border-gray-800 space-y-6">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold text-white tracking-wide">Threshold Knobs & Safety Boundaries</h3>
          <button
            onClick={handleSave}
            className="px-4 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-gray-950 font-bold text-xs flex items-center gap-1.5 transition-colors cursor-pointer"
          >
            <Save className="w-4 h-4" />
            Save Profile
          </button>
        </div>

        {savedNotice && (
          <div className="p-3 rounded-lg bg-emerald-500/10 border border-emerald-500/30 text-emerald-300 text-xs flex items-center gap-2">
            <CheckCircle2 className="w-4 h-4 shrink-0" />
            <span>Controller parameters updated in active pipeline instance!</span>
          </div>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 font-mono text-xs">
          {/* Tau High */}
          <div className="space-y-2 p-4 rounded-xl bg-gray-900/60 border border-gray-800">
            <div className="flex justify-between items-center text-gray-300">
              <span>Confidence Threshold: Tau High (Retain)</span>
              <span className="text-cyan-400 font-bold">{tauHigh}</span>
            </div>
            <input
              type="range"
              min="0.5"
              max="0.99"
              step="0.01"
              value={tauHigh}
              onChange={(e) => setTauHigh(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-400"
            />
            <p className="text-[11px] text-gray-400 font-sans">
              Model A predictions with confidence &ge; Tau High are accepted immediately without invoking probes.
            </p>
          </div>

          {/* Tau Low */}
          <div className="space-y-2 p-4 rounded-xl bg-gray-900/60 border border-gray-800">
            <div className="flex justify-between items-center text-gray-300">
              <span>Confidence Threshold: Tau Low (Switch)</span>
              <span className="text-cyan-400 font-bold">{tauLow}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.8"
              step="0.01"
              value={tauLow}
              onChange={(e) => setTauLow(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-400"
            />
            <p className="text-[11px] text-gray-400 font-sans">
              Confidence &lt; Tau Low triggers immediate escalation to Model B. Intermediate values invoke probes.
            </p>
          </div>

          {/* Latency Budget */}
          <div className="space-y-2 p-4 rounded-xl bg-gray-900/60 border border-gray-800">
            <div className="flex justify-between items-center text-gray-300">
              <span>Latency SLA Budget</span>
              <span className="text-amber-400 font-bold">{budget} ms</span>
            </div>
            <input
              type="range"
              min="5.0"
              max="60.0"
              step="1.0"
              value={budget}
              onChange={(e) => setBudget(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-amber-400"
            />
            <p className="text-[11px] text-gray-400 font-sans">
              Strict per-query execution limit. Probe blocks will not execute if remaining budget is insufficient.
            </p>
          </div>

          {/* Thermal Throttle Limit */}
          <div className="space-y-2 p-4 rounded-xl bg-gray-900/60 border border-gray-800">
            <div className="flex justify-between items-center text-gray-300">
              <span>GPU Thermal Throttle Ceiling</span>
              <span className="text-rose-400 font-bold">{thermalLimit} °C</span>
            </div>
            <input
              type="range"
              min="65.0"
              max="88.0"
              step="1.0"
              value={thermalLimit}
              onChange={(e) => setThermalLimit(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-rose-400"
            />
            <p className="text-[11px] text-gray-400 font-sans">
              Emergency thermal brake. Forces immediate Model A exit if device temperature exceeds this value.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

