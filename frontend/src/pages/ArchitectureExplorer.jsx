import React, { useState } from 'react';
import {
  Layers,
  Cpu,
  Database,
  ArrowRight,
  ShieldAlert,
  CheckCircle2,
  Workflow,
  Sparkles,
  Zap,
  Info
} from 'lucide-react';
import StatusBadge from '../components/StatusBadge';

export default function ArchitectureExplorer() {
  const [selectedComponent, setSelectedComponent] = useState('cache');

  const components = {
    model_a: {
      name: 'Model A (MobileNetV3-Small)',
      category: 'Baseline Edge Network',
      params: '0.93 Million',
      latency: '1.488 ms (NVIDIA RTX 3050)',
      layers: '12 Inverted Residual Blocks (0-11)',
      output_dim: '10 classes (CIFAR-10)',
      role: 'Serves as the always-active first-stage inference backbone. Handles the majority of straightforward edge inference queries with minimal energy consumption and sub-2ms latency.',
      details: [
        'Architecture: Hard-swish activations, Squeeze-and-Excitation attention modules',
        'Input Resolution: 32x32 RGB images (CIFAR-10)',
        'Checkpoint: outputs/baseline/model_a_best.pt',
        'Top-1 Test Accuracy: 88.60% (1,000 stratified samples)'
      ]
    },
    controller: {
      name: 'Adaptive Decision Controller',
      category: 'Inference Policy Engine',
      params: '< 100 Parameters (Dynamic Policy)',
      latency: '< 0.05 ms Overhead',
      layers: 'Analytical Policy Engine',
      output_dim: 'Decision Enum {RETAIN, PROBE, SWITCH}',
      role: 'Dynamically arbitrates execution based on prediction entropy, confidence margins, and live hardware resource constraints.',
      details: [
        'Confidence Evaluation: tau_high (0.85 default), tau_low (0.50 default)',
        'Thermal Guard: Automatically forces RETAIN if GPU temp > 80°C or budget exhausted',
        'Probe Utility Score: Evaluates expected information gain vs incremental FLOPs cost',
        'Safety Guarantee: Zero false-confidence switches when budget is constrained'
      ]
    },
    probe4: {
      name: 'Intermediate Probe Head (Block 4)',
      category: 'Early Feature Exit',
      params: '0.02 Million',
      latency: '0.35 ms Incremental',
      layers: 'AdaptiveAvgPool2d -> Linear(24 -> 10)',
      output_dim: '10 classes',
      role: 'Taps into low-level texture and edge representations after Block 4 of MobileNetV3-Large to provide an ultra-fast intermediate exit.',
      details: [
        'Input Channels: 24 channels (Feature Map: 8x8)',
        'Checkpoint: outputs/probes/probe_block4_best.pt',
        'Early Exit Accuracy: ~78.2%',
        'Purpose: Resolves coarse-grained classification queries early'
      ]
    },
    probe9: {
      name: 'Intermediate Probe Head (Block 9)',
      category: 'Mid-Stage Feature Exit',
      params: '0.04 Million',
      latency: '0.62 ms Incremental',
      layers: 'AdaptiveAvgPool2d -> Linear(40 -> 10)',
      output_dim: '10 classes',
      role: 'Captures mid-level semantic parts and object shapes at Block 9 of MobileNetV3-Large.',
      details: [
        'Input Channels: 40 channels (Feature Map: 4x4)',
        'Checkpoint: outputs/probes/probe_block9_best.pt',
        'Exit Accuracy: ~85.4%',
        'Purpose: Differentiates similar object classes before entering deep stages'
      ]
    },
    probe13: {
      name: 'Intermediate Probe Head (Block 13)',
      category: 'Deep Representation Exit & Cache Anchor',
      params: '0.16 Million',
      latency: '0.94 ms Incremental',
      layers: 'AdaptiveAvgPool2d -> Linear(160 -> 10)',
      output_dim: '10 classes',
      role: 'Extracts high-level semantic features right before the classifier tail. Serves as the anchor point for Continuation Caching.',
      details: [
        'Input Channels: 160 channels (Feature Map: 4x4)',
        'Checkpoint: outputs/probes/probe_block13_best.pt',
        'Exit Accuracy: ~89.8%',
        'Anchor: Activations are stored in Continuation Cache for seamless Model B tail execution'
      ]
    },
    cache: {
      name: 'Continuation Caching Engine',
      category: 'Patent-Core Computational Memory',
      params: '0 Memory Overhead (Transient Activation Buffer)',
      latency: '< 0.01 ms Retrieval',
      layers: 'Block 13 Activation Buffer -> Model B Tail (Blocks 14-16)',
      output_dim: 'Shape: [Batch, 160, 4, 4]',
      role: 'The flagship novelty of ADAPT-EDGE. When a probe at Block 13 is evaluated, its activation map is retained in fast GPU VRAM. If the system escalates to full Model B, execution resumes directly from Block 14, eliminating 35.7% of redundant computation.',
      details: [
        'Compute Savings: Exactly 35.7% of full Model B FLOPs eliminated on switch',
        'Numerical Parity: Bitwise identical to standard end-to-end forward pass (max diff = 0.000e+00)',
        'Cache Keying: Managed per inference session with automatic invalidation',
        'Safety: Thread-safe activation storage with automatic fallback if unpopulated'
      ]
    },
    model_b: {
      name: 'Model B (MobileNetV3-Large)',
      category: 'Full High-Capacity Network',
      params: '4.20 Million (4.5x larger than Model A)',
      latency: '3.757 ms (NVIDIA RTX 3050)',
      layers: '17 Inverted Residual Blocks (0-16) + Dual Linear Classifier',
      output_dim: '10 classes (CIFAR-10)',
      role: 'The heavy oracle model deployed on edge nodes. Delivers maximum classification accuracy (92.50%) when early stages are uncertain.',
      details: [
        'Architecture: 16 bottleneck residual stages with squeeze-and-excitation attention',
        'Checkpoint: outputs/baseline/model_b_best.pt',
        'Test Accuracy: 92.50% (Phase 5 verified)',
        'Execution: Can run from scratch OR resume from Block 13 via Continuation Cache'
      ]
    }
  };

  const comp = components[selectedComponent];

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Header Banner */}
      <div className="glass-panel p-6 rounded-xl border border-gray-800 space-y-2">
        <div className="flex items-center gap-2 text-cyan-400 text-xs font-mono">
          <Layers className="w-4 h-4" />
          SYSTEM TOPOLOGY & REUSE PATHWAYS
        </div>
        <h2 className="text-xl font-bold text-white">ADAPT-EDGE Multi-Stage Architectural Anatomy</h2>
        <p className="text-xs text-gray-300 max-w-3xl leading-relaxed">
          Explore the internal structure of the ADAPT-EDGE pipeline. Click on any sub-system to inspect its
          parameter footprints, layer topologies, intermediate channel dimensions, and continuation caching guarantees.
        </p>
      </div>

      {/* Component Selector Pills */}
      <div className="flex flex-wrap gap-2">
        {Object.entries(components).map(([key, item]) => {
          const isSelected = selectedComponent === key;
          return (
            <button
              key={key}
              onClick={() => setSelectedComponent(key)}
              className={`px-3.5 py-2 rounded-xl text-xs font-medium transition-all flex items-center gap-2 ${
                isSelected
                  ? 'bg-cyan-500/20 border border-cyan-400 text-cyan-300 shadow-lg shadow-cyan-500/10'
                  : 'bg-gray-900/80 border border-gray-800 text-gray-400 hover:text-gray-200'
              }`}
            >
              {key === 'cache' && <Sparkles className="w-3.5 h-3.5 text-amber-400" />}
              {item.name}
            </button>
          );
        })}
      </div>

      {/* Active Component Deep Dive */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left 2 Cols: Technical Specification */}
        <div className="glass-panel p-6 rounded-xl border border-gray-800 space-y-5 lg:col-span-2">
          <div className="flex items-center justify-between">
            <div>
              <span className="text-[11px] font-mono text-cyan-400 uppercase tracking-wider">{comp.category}</span>
              <h3 className="text-lg font-bold text-white">{comp.name}</h3>
            </div>
            <StatusBadge status="ACTIVE" label="VERIFIED" />
          </div>

          <p className="text-xs text-gray-300 leading-relaxed">{comp.role}</p>

          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            <div className="p-3 rounded-lg bg-gray-900/80 border border-gray-800">
              <span className="text-[10px] font-mono text-gray-400 uppercase block">Parameters</span>
              <span className="text-sm font-bold text-white font-mono">{comp.params}</span>
            </div>
            <div className="p-3 rounded-lg bg-gray-900/80 border border-gray-800">
              <span className="text-[10px] font-mono text-gray-400 uppercase block">Latency Overhead</span>
              <span className="text-sm font-bold text-cyan-300 font-mono">{comp.latency}</span>
            </div>
            <div className="p-3 rounded-lg bg-gray-900/80 border border-gray-800 col-span-2 sm:col-span-1">
              <span className="text-[10px] font-mono text-gray-400 uppercase block">Layer Boundary</span>
              <span className="text-xs font-bold text-indigo-300 font-mono truncate block">{comp.layers}</span>
            </div>
          </div>

          <div className="space-y-2 pt-2">
            <span className="text-xs font-bold text-white font-mono uppercase tracking-wider">Implementation Specifications</span>
            <ul className="space-y-1.5">
              {comp.details.map((d, idx) => (
                <li key={idx} className="flex items-start gap-2 text-xs text-gray-300">
                  <CheckCircle2 className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5" />
                  <span>{d}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Right Col: Continuation Caching Technical Insight */}
        <div className="glass-panel p-6 rounded-xl border border-gray-800 space-y-4">
          <div className="flex items-center gap-2 text-amber-400">
            <Sparkles className="w-4 h-4" />
            <h4 className="text-xs font-bold uppercase tracking-wider">Patent Innovation: Activation Reuse</h4>
          </div>

          <div className="p-4 rounded-xl bg-gray-900/90 border border-gray-800 space-y-3 text-xs">
            <div className="text-cyan-300 font-mono font-bold">Why Continuation Caching is Novel:</div>
            <p className="text-gray-300 leading-relaxed">
              Standard speculative inference pipelines throw away early probe features and re-execute the target model
              from scratch (<span className="text-rose-400 font-mono">5.5x FLOPs penalty</span>).
            </p>
            <p className="text-gray-300 leading-relaxed">
              ADAPT-EDGE preserves Block 13 activations. When the controller escalates, it executes <strong className="text-white">only Blocks 14-16</strong>,
              saving exactly <span className="text-emerald-400 font-bold font-mono">35.7%</span> of computation without losing a single bit of precision.
            </p>
            <div className="p-2.5 rounded bg-black/40 border border-gray-800 font-mono text-[11px] text-gray-400 space-y-1">
              <div>Output Tensor: <span className="text-white">[1, 160, 4, 4]</span></div>
              <div>Numerical Parity: <span className="text-emerald-400">0.000e+00 Error</span></div>
              <div>Memory Footprint: <span className="text-cyan-300">~10.2 KB VRAM</span></div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

