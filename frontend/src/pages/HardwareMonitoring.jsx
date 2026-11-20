import React, { useState, useEffect } from 'react';
import {
  Cpu,
  Database,
  Thermometer,
  Activity,
  HardDrive,
  CheckCircle2,
  AlertTriangle,
  RotateCcw
} from 'lucide-react';
import { api } from '../api';
import StatusBadge from '../components/StatusBadge';

export default function HardwareMonitoring() {
  const [hw, setHw] = useState(null);
  const [history, setHistory] = useState([]);
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    let mounted = true;
    const fetchHw = async () => {
      try {
        const data = await api.getHardware();
        if (mounted) {
          setHw(data);
          setHistory((prev) => [
            ...prev.slice(-19),
            {
              time: new Date().toLocaleTimeString(),
              cpu: data.cpu_utilization_percent || 0,
              gpu: data.gpu_utilization_percent || 0,
              temp: data.gpu_temperature_c || 0,
              vram: data.gpu_memory_allocated_mb || 0
            }
          ]);
        }
      } catch (e) {
        console.error(e);
      }
    };

    fetchHw();
    let interval = null;
    if (autoRefresh) {
      interval = setInterval(fetchHw, 2000);
    }
    return () => {
      mounted = false;
      if (interval) clearInterval(interval);
    };
  }, [autoRefresh]);

  const temp = hw?.gpu_temperature_c ?? 0;
  let tempStatus = 'PASS';
  let tempLabel = 'OPTIMAL';
  if (temp > 80) {
    tempStatus = 'FAIL';
    tempLabel = 'THROTTLING RISK';
  } else if (temp > 72) {
    tempStatus = 'WARN';
    tempLabel = 'ELEVATED';
  }

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Top Banner */}
      <div className="glass-panel p-6 rounded-xl border border-gray-800 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 text-cyan-400 text-xs font-mono">
            <Cpu className="w-4 h-4" />
            EDGE HARDWARE TELEMETRY
          </div>
          <h2 className="text-xl font-bold text-white">Live System & GPU Telemetry</h2>
          <p className="text-xs text-gray-400">
            Real-time hardware sensors monitoring device thermal budget and VRAM allocation.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setAutoRefresh(!autoRefresh)}
            className={`px-3 py-1.5 rounded-lg border text-xs font-mono transition-colors flex items-center gap-2 ${
              autoRefresh
                ? 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300'
                : 'bg-gray-800 border-gray-700 text-gray-400'
            }`}
          >
            <span className={`w-2 h-2 rounded-full ${autoRefresh ? 'bg-emerald-400 animate-pulse' : 'bg-gray-500'}`} />
            {autoRefresh ? 'Live Polling (2s)' : 'Paused'}
          </button>
        </div>
      </div>

      {/* Sensor Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* CPU */}
        <div className="glass-panel p-5 rounded-xl border border-gray-800 space-y-2">
          <div className="flex items-center justify-between text-xs font-mono text-gray-400">
            <span>CPU UTILIZATION</span>
            <Cpu className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-3xl font-bold text-white font-mono">
            {hw?.cpu_utilization_percent ?? '--'}%
          </div>
          <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-cyan-400 rounded-full transition-all duration-300"
              style={{ width: `${Math.min(100, hw?.cpu_utilization_percent || 0)}%` }}
            />
          </div>
          <div className="text-[11px] text-gray-400 font-mono">Windows Host System</div>
        </div>

        {/* RAM */}
        <div className="glass-panel p-5 rounded-xl border border-gray-800 space-y-2">
          <div className="flex items-center justify-between text-xs font-mono text-gray-400">
            <span>SYSTEM RAM</span>
            <HardDrive className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="text-3xl font-bold text-white font-mono">
            {hw?.ram_utilization_percent ?? '--'}%
          </div>
          <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-indigo-500 rounded-full transition-all duration-300"
              style={{ width: `${Math.min(100, hw?.ram_utilization_percent || 0)}%` }}
            />
          </div>
          <div className="text-[11px] text-gray-400 font-mono">
            {hw?.ram_used_gb ? `${hw.ram_used_gb} GB / ${hw.ram_total_gb} GB` : '16 GB Total'}
          </div>
        </div>

        {/* GPU Temperature */}
        <div className="glass-panel p-5 rounded-xl border border-gray-800 space-y-2">
          <div className="flex items-center justify-between text-xs font-mono text-gray-400">
            <span>GPU TEMPERATURE</span>
            <Thermometer className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-3xl font-bold text-white font-mono">
            {hw?.gpu_temperature_c !== null ? `${hw?.gpu_temperature_c}°C` : 'N/A'}
          </div>
          <div className="flex items-center justify-between">
            <StatusBadge status={tempStatus} label={tempLabel} size="xs" />
            <span className="text-[11px] text-gray-400 font-mono">Limit: 85°C</span>
          </div>
        </div>

        {/* PyTorch VRAM */}
        <div className="glass-panel p-5 rounded-xl border border-gray-800 space-y-2">
          <div className="flex items-center justify-between text-xs font-mono text-gray-400">
            <span>PYTORCH VRAM ALLOCATED</span>
            <Database className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-3xl font-bold text-emerald-400 font-mono">
            {hw?.gpu_memory_allocated_mb ?? '--'} MB
          </div>
          <div className="w-full h-1.5 bg-gray-800 rounded-full overflow-hidden">
            <div
              className="h-full bg-emerald-500 rounded-full transition-all duration-300"
              style={{ width: `${Math.min(100, ((hw?.gpu_memory_allocated_mb || 0) / 4096) * 100)}%` }}
            />
          </div>
          <div className="text-[11px] text-gray-400 font-mono">
            Total VRAM: 4,096 MB (RTX 3050)
          </div>
        </div>
      </div>

      {/* GPU Hardware Specifications */}
      <div className="glass-panel p-6 rounded-xl border border-gray-800 space-y-4">
        <h3 className="text-sm font-bold text-white tracking-wide">Target Edge Device Architecture</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 text-xs font-mono">
          <div className="p-3 rounded-lg bg-gray-900 border border-gray-800">
            <span className="text-gray-400 block">GPU Name:</span>
            <span className="text-white font-bold">{hw?.gpu_name || 'NVIDIA GeForce RTX 3050 Laptop GPU'}</span>
          </div>
          <div className="p-3 rounded-lg bg-gray-900 border border-gray-800">
            <span className="text-gray-400 block">PyTorch CUDA Status:</span>
            <span className="text-emerald-400 font-bold">{hw?.cuda_available ? 'CUDA 12.6 Enabled' : 'CPU Fallback'}</span>
          </div>
          <div className="p-3 rounded-lg bg-gray-900 border border-gray-800">
            <span className="text-gray-400 block">Continuation Cache Status:</span>
            <span className="text-cyan-400 font-bold">Active in GPU Memory</span>
          </div>
        </div>
      </div>
    </div>
  );
}

