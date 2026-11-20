import React, { useState, useEffect } from 'react';
import {
  Database,
  Upload,
  CheckCircle2,
  AlertCircle,
  FileArchive,
  Image as ImageIcon,
  FolderOpen,
  Filter
} from 'lucide-react';
import { api } from '../api';
import StatusBadge from '../components/StatusBadge';

export default function DatasetManagement() {
  const [stats, setStats] = useState(null);
  const [samples, setSamples] = useState([]);
  const [selectedClass, setSelectedClass] = useState(null);
  const [page, setPage] = useState(0);
  const [uploadStatus, setUploadStatus] = useState(null);

  const classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
  ];

  useEffect(() => {
    loadStats();
    loadSamples(0, selectedClass);
  }, []);

  const loadStats = async () => {
    try {
      const data = await api.getCifarStats();
      setStats(data);
    } catch (e) {
      console.error(e);
    }
  };

  const loadSamples = async (p = 0, cls = null) => {
    try {
      const data = await api.getCifarSamples(p, 18, cls);
      if (data && data.samples) {
        setSamples(data.samples);
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleClassFilter = (clsIdx) => {
    setSelectedClass(clsIdx);
    setPage(0);
    loadSamples(0, clsIdx);
  };

  return (
    <div className="p-6 space-y-6 max-w-7xl mx-auto">
      {/* Banner */}
      <div className="glass-panel p-6 rounded-xl border border-gray-800 space-y-2">
        <div className="flex items-center gap-2 text-cyan-400 text-xs font-mono">
          <Database className="w-4 h-4" />
          DATASET GOVERNANCE & INTEGRITY
        </div>
        <h2 className="text-xl font-bold text-white">Dataset Management & CIFAR-10 Inspection</h2>
        <p className="text-xs text-gray-300 max-w-3xl leading-relaxed">
          Explore the evaluation benchmark dataset. Review class distributions, inspect sample imagery,
          or validate custom datasets for edge AI deployment.
        </p>
      </div>

      {/* Dataset Statistics Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="glass-panel p-5 rounded-xl border border-gray-800">
          <span className="text-xs font-mono text-gray-400 uppercase">Dataset Name</span>
          <div className="mt-1 text-xl font-bold text-white font-mono">CIFAR-10</div>
          <span className="text-[11px] text-gray-400">10 Distinct Object Classes</span>
        </div>

        <div className="glass-panel p-5 rounded-xl border border-gray-800">
          <span className="text-xs font-mono text-gray-400 uppercase">Test Set Size</span>
          <div className="mt-1 text-xl font-bold text-cyan-400 font-mono">10,000 Images</div>
          <span className="text-[11px] text-gray-400">1,000 per class (Balanced)</span>
        </div>

        <div className="glass-panel p-5 rounded-xl border border-gray-800">
          <span className="text-xs font-mono text-gray-400 uppercase">Input Dimension</span>
          <div className="mt-1 text-xl font-bold text-indigo-400 font-mono">32 x 32 x 3</div>
          <span className="text-[11px] text-gray-400">Normalized RGB Tensors</span>
        </div>

        <div className="glass-panel p-5 rounded-xl border border-gray-800">
          <span className="text-xs font-mono text-gray-400 uppercase">Audited Benchmark</span>
          <div className="mt-1 text-xl font-bold text-emerald-400 font-mono">1,000 Stratified</div>
          <span className="text-[11px] text-gray-400">Seed 42 / Exact Class Parity</span>
        </div>
      </div>

      {/* Class Filter Bar */}
      <div className="glass-panel p-5 rounded-xl border border-gray-800 space-y-3">
        <div className="flex items-center justify-between">
          <div className="text-xs font-mono text-gray-300 font-semibold flex items-center gap-2">
            <Filter className="w-3.5 h-3.5 text-cyan-400" />
            FILTER TEST SAMPLES BY CLASS:
          </div>
          {selectedClass !== null && (
            <button
              onClick={() => handleClassFilter(null)}
              className="text-xs text-rose-400 hover:text-rose-300 font-mono"
            >
              Clear Filter
            </button>
          )}
        </div>

        <div className="flex flex-wrap gap-2">
          {classes.map((clsName, idx) => (
            <button
              key={idx}
              onClick={() => handleClassFilter(idx)}
              className={`px-3 py-1.5 rounded-lg text-xs font-mono transition-all ${
                selectedClass === idx
                  ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-400 font-bold'
                  : 'bg-gray-900 border border-gray-800 text-gray-400 hover:text-gray-200'
              }`}
            >
              {clsName} ({idx})
            </button>
          ))}
        </div>
      </div>

      {/* Samples Grid */}
      <div className="glass-panel p-6 rounded-xl border border-gray-800 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-bold text-white tracking-wide">
            CIFAR-10 Visual Sample Explorer ({samples.length} items)
          </h3>
          <div className="flex items-center gap-2 font-mono text-xs">
            <button
              onClick={() => {
                const nextP = Math.max(0, page - 1);
                setPage(nextP);
                loadSamples(nextP, selectedClass);
              }}
              disabled={page === 0}
              className="px-2.5 py-1 bg-gray-900 border border-gray-800 rounded text-gray-300 hover:bg-gray-800 disabled:opacity-40"
            >
              Prev
            </button>
            <span className="text-gray-400">Page {page + 1}</span>
            <button
              onClick={() => {
                const nextP = page + 1;
                setPage(nextP);
                loadSamples(nextP, selectedClass);
              }}
              className="px-2.5 py-1 bg-gray-900 border border-gray-800 rounded text-gray-300 hover:bg-gray-800"
            >
              Next
            </button>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-6 gap-3">
          {samples.map((s) => (
            <div
              key={s.index}
              className="p-2.5 rounded-lg bg-gray-900/60 border border-gray-800 flex flex-col items-center gap-2"
            >
              <img
                src={s.image_base64}
                alt={s.class_name}
                className="w-14 h-14 object-cover rounded bg-black border border-gray-700"
              />
              <div className="text-center font-mono w-full">
                <span className="text-[10px] text-gray-400 block">#{s.index}</span>
                <span className="text-xs font-semibold text-white truncate block">{s.class_name}</span>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Custom Dataset Upload Validator */}
      <div className="glass-panel p-6 rounded-xl border border-gray-800 space-y-4">
        <h3 className="text-sm font-bold text-white tracking-wide flex items-center gap-2">
          <Upload className="w-4 h-4 text-cyan-400" />
          Custom Dataset ZIP Validator
        </h3>
        <p className="text-xs text-gray-400 leading-relaxed">
          Upload a ZIP archive structured as subdirectories representing class labels (e.g. <code>airplane/</code>, <code>automobile/</code>).
          The server validates image dimensions, formats, and generates normalization statistics for edge deployment.
        </p>

        <div className="p-8 border-2 border-dashed border-gray-700 rounded-xl bg-gray-900/40 text-center flex flex-col items-center justify-center gap-3">
          <FileArchive className="w-8 h-8 text-cyan-400" />
          <div className="text-xs text-gray-300">
            <span className="font-semibold text-cyan-300">Click to browse</span> or drag ZIP archive here
          </div>
          <span className="text-[11px] text-gray-400 font-mono">Max size: 50MB (.zip)</span>
        </div>
      </div>
    </div>
  );
}

