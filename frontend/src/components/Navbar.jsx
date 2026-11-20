import React, { useState, useEffect } from 'react';
import { Menu, RefreshCw, Trash2, Cpu, Thermometer, Database, ExternalLink } from 'lucide-react';
import { api } from '../api';

export default function Navbar({ activeTab, onRefresh, onToggleSidebar }) {
  const [hw, setHw] = useState(null);
  const [clearing, setClearing] = useState(false);

  // Poll hardware status (light‑theme colors)
  useEffect(() => {
    let mounted = true;
    const fetchHw = async () => {
      try {
        const data = await api.getHardware();
        if (mounted) setHw(data);
      } catch {}
    };
    fetchHw();
    const interval = setInterval(fetchHw, 3000);
    return () => {
      mounted = false;
      clearInterval(interval);
    };
  }, []);

  const handleClearCache = async () => {
    if (!window.confirm("Clear the actual pipeline continuation cache?")) return;
    try {
      setClearing(true);
      await api.clearContinuationCache();
      if (onRefresh) onRefresh();
    } catch (e) {
      alert('Failed to clear cache: ' + e.message);
    } finally {
      setClearing(false);
    }
  };

  const pageTitles = {
    overview: { title: 'Executive Overview', desc: 'System architecture, baseline comparison and live status' },
    datatrain: { title: 'Dataset & Training', desc: 'Upload a ZIP dataset & start model training' },
    livedemo: { title: 'Live Demo', desc: 'Interactively run inference on the edge model' },
    results: { title: 'Results & Comparison', desc: 'View experiment results and benchmark comparisons' },
    settings: { title: 'Settings', desc: 'Configure thresholds, cache options and UI preferences' },
  };

  const curr = pageTitles[activeTab] || { title: 'ADAPT-EDGE Platform', desc: 'Edge AI Research' };

  return (
    <header className="h-16 bg-[#f9fafb] border-b border-gray-300 px-6 flex items-center justify-between shrink-0 z-20">
      {/* Hamburger for mobile */}
      <button
        className="md:hidden text-gray-600 hover:text-gray-900 p-2"
        onClick={onToggleSidebar}
      >
        <Menu className="w-5 h-5" />
      </button>

      {/* Title / description */}
      <div className="flex-1 ml-2">
        <h1 className="text-base font-bold text-primary tracking-tight">{curr.title}</h1>
        <p className="text-xs text-secondary">{curr.desc}</p>
      </div>

      {/* Right‑side actions */}
      <div className="flex items-center space-x-3">
        {/* Hardware status pill (light theme) */}
        {hw && (
          <div className="hidden lg:flex items-center space-x-3 px-3 py-1.5 rounded-lg bg-gray-100 border border-gray-200 text-xs font-mono">
            <div className="flex items-center space-x-1.5 text-gray-600">
              <Cpu className="w-3.5 h-3.5 text-cyan-500" />
              <span>CPU: <strong className="text-gray-800">{hw.telemetry?.cpu_percent != null ? `${hw.telemetry.cpu_percent}%` : 'N/A'}</strong></span>
            </div>
            <div className="w-px h-3 bg-gray-300" />
            <div className="flex items-center space-x-1.5 text-gray-600">
              <Database className="w-3.5 h-3.5 text-indigo-500" />
              <span>VRAM: <strong className="text-gray-800">{hw.telemetry?.gpu_torch_allocated_mb != null ? `${hw.telemetry.gpu_torch_allocated_mb}MB` : 'N/A'}</strong></span>
            </div>
            {hw.telemetry?.gpu_temperature_c != null && (
              <>
                <div className="w-px h-3 bg-gray-300" />
                <div className="flex items-center space-x-1.5 text-gray-600">
                  <Thermometer className="w-3.5 h-3.5 text-amber-500" />
                  <span>{hw.telemetry.gpu_temperature_c}°C</span>
                </div>
              </>
            )}
          </div>
        )}

        {/* Clear cache button */}
        <button
          onClick={handleClearCache}
          disabled={clearing}
          title="Flush PyTorch CUDA and Continuation Cache"
          className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-gray-200 hover:bg-gray-300 border border-gray-300 text-xs text-gray-700 transition-colors"
        >
          <Trash2 className="w-3.5 h-3.5 text-rose-600" />
          <span className="hidden sm:inline">Flush Cache</span>
        </button>

        {/* Refresh button */}
        {onRefresh && (
          <button
            onClick={onRefresh}
            title="Refresh View"
            className="p-2 rounded-lg bg-gray-200 hover:bg-gray-300 border border-gray-300 text-gray-700 transition-colors"
          >
            <RefreshCw className="w-3.5 h-3.5" />
          </button>
        )}

        {/* API Docs link */}
        <a
          href="/docs"
          target="_blank"
          rel="noreferrer"
          className="hidden md:flex items-center space-x-1 px-2.5 py-1.5 rounded-lg bg-cyan-100 hover:bg-cyan-200 text-cyan-800 border border-cyan-200 text-xs font-mono transition-colors"
        >
          <span>Swagger API</span>
          <ExternalLink className="w-3 h-3" />
        </a>
      </div>
    </header>
  );
}
