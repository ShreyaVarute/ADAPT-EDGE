import React, { useEffect, useRef, useState, useCallback } from 'react';
import {
  UploadCloud, Database, FolderOpen, Image as ImageIcon,
  AlertTriangle, CheckCircle2, XCircle, Info, ChevronLeft,
  ChevronRight, RefreshCw, Search, X, ShieldCheck, FlaskConical,
  BarChart2, Layers, Cpu, Zap, Box, TrendingUp, Sliders,
} from 'lucide-react';
import {
  Bar, BarChart, CartesianGrid, Cell, Line, LineChart,
  ResponsiveContainer, Tooltip, XAxis, YAxis, Legend,
} from 'recharts';
import { api } from '../api';

/* ─── palette for class bars ──────────────────────────────────────────────── */
const BAR_COLORS = [
  '#2563eb','#0891b2','#7c3aed','#059669','#d97706',
  '#dc2626','#9333ea','#0d9488','#b45309','#1d4ed8',
];

/* ─── small atoms ─────────────────────────────────────────────────────────── */
const fieldClass =
  'mt-1 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 outline-none focus:border-blue-500';

function Stat({ label, value, sub, color = 'text-slate-900' }) {
  return (
    <div className="rounded-xl border border-slate-100 bg-slate-50 p-4">
      <p className="text-xs text-slate-500">{label}</p>
      <p className={`mt-1 text-xl font-bold ${color}`}>{value ?? '—'}</p>
      {sub && <p className="mt-0.5 text-xs text-slate-400">{sub}</p>}
    </div>
  );
}

function StatusPill({ status }) {
  const map = {
    ready:     'bg-green-50 text-green-700',
    not_found: 'bg-red-50 text-red-700',
  };
  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-semibold ${map[status] ?? 'bg-slate-100 text-slate-600'}`}>
      {status === 'ready' ? 'Ready' : status === 'not_found' ? 'Not found' : status}
    </span>
  );
}

function SplitBar({ trainPct, valPct }) {
  return (
    <div className="mt-3 space-y-1.5">
      <p className="text-xs font-semibold text-slate-600">Recommended splits</p>
      <div className="flex h-5 w-full overflow-hidden rounded-full">
        <div
          className="flex items-center justify-center bg-blue-500 text-[10px] font-bold text-white"
          style={{ width: `${trainPct}%` }}
        >Train {trainPct}%</div>
        <div
          className="flex items-center justify-center bg-amber-400 text-[10px] font-bold text-white"
          style={{ width: `${valPct}%` }}
        >Val {valPct}%</div>
        {100 - trainPct - valPct > 0 && (
          <div
            className="flex items-center justify-center bg-slate-300 text-[10px] font-bold text-slate-700"
            style={{ width: `${100 - trainPct - valPct}%` }}
          >Test {100 - trainPct - valPct}%</div>
        )}
      </div>
    </div>
  );
}

/* ─── upload zone ─────────────────────────────────────────────────────────── */
function UploadZone({ onUpload, uploading, uploadProgress, uploadResult }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const handleDrop = (e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) onUpload(file);
  };

  return (
    <div className="space-y-4">
      <div
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        onClick={() => inputRef.current?.click()}
        className={`flex min-h-40 cursor-pointer flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed px-4 text-center transition-colors ${
          dragging ? 'border-blue-500 bg-blue-50' : 'border-slate-300 bg-slate-50 hover:border-blue-400 hover:bg-blue-50/40'
        }`}
      >
        <input ref={inputRef} type="file" accept=".zip" className="hidden" onChange={(e) => onUpload(e.target.files[0])} />
        <UploadCloud className={`h-10 w-10 ${dragging ? 'text-blue-500' : 'text-slate-400'}`} />
        <div>
          <p className="text-sm font-semibold text-slate-700">Drop a ZIP archive or click to browse</p>
          <p className="mt-1 text-xs text-slate-500">Class-folder structure required · Max 500 MB · JPG, PNG, BMP, WebP</p>
        </div>
      </div>

      {/* upload progress */}
      {uploading && (
        <div className="space-y-1.5">
          <div className="flex items-center justify-between text-xs text-slate-600">
            <span>Uploading and validating…</span>
            <span>{uploadProgress}%</span>
          </div>
          <div className="h-2 w-full overflow-hidden rounded-full bg-slate-200">
            <div className="h-2 rounded-full bg-blue-500 transition-all" style={{ width: `${uploadProgress}%` }} />
          </div>
        </div>
      )}

      {/* upload result */}
      {uploadResult && (
        <div className={`rounded-xl border p-4 ${uploadResult.success ? 'border-green-200 bg-green-50' : 'border-red-200 bg-red-50'}`}>
          {uploadResult.success ? (
            <div className="flex items-start gap-3">
              <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-green-600" />
              <div>
                <p className="text-sm font-semibold text-green-800">Upload successful</p>
                <p className="mt-0.5 text-xs text-green-700">
                  <strong>{uploadResult.name}</strong> · {uploadResult.total_images} images · {uploadResult.class_count} classes
                  {uploadResult.corrupted_images > 0 && ` · ${uploadResult.corrupted_images} corrupted (skipped)`}
                </p>
                <div className="mt-2 flex flex-wrap gap-1">
                  {(uploadResult.classes_detected || []).map((c) => (
                    <span key={c} className="rounded bg-green-100 px-1.5 py-0.5 text-[10px] font-medium text-green-800">{c}</span>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <div className="flex items-start gap-3">
              <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-600" />
              <div>
                <p className="text-sm font-semibold text-red-800">Upload failed</p>
                <p className="mt-0.5 text-xs text-red-700">{uploadResult.error}</p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* expected structure hint */}
      <div className="rounded-xl border border-slate-100 bg-slate-50 p-4">
        <div className="flex items-center gap-2 mb-2">
          <Info className="h-4 w-4 text-blue-500 shrink-0" />
          <p className="text-xs font-semibold text-slate-700">Expected ZIP structure</p>
        </div>
        <pre className="text-[11px] leading-5 text-slate-600 font-mono bg-white rounded-lg p-3 border border-slate-100 overflow-x-auto">{`dataset.zip
├── class_name_1/
│   ├── image001.jpg
│   └── image002.png
├── class_name_2/
│   └── image003.jpg
└── class_name_3/
    └── image004.webp`}</pre>
        <p className="mt-2 text-[11px] text-slate-500">
          Each top-level folder becomes a class label. Images inside are validated for format and integrity.
          Corrupted or unsupported files are skipped and reported.
        </p>
      </div>
    </div>
  );
}

/* ─── dataset inspect panel ───────────────────────────────────────────────── */
function DatasetInspect({ info, isCustom }) {
  const [classFilter, setClassFilter] = useState(null);
  const [search, setSearch] = useState('');

  if (!info) return null;

  const dist = info.class_distribution || {};
  const classes = info.classes || [];
  const chartData = Object.entries(dist).map(([name, count], i) => ({ name, count, fill: BAR_COLORS[i % BAR_COLORS.length] }));
  const splits = info.recommended_splits || {};
  const trainPct = splits.train ?? 80;
  const valPct   = splits.validation ?? (splits.val ?? 10);

  /* image preview list (URL strings or b64 objects) */
  const previews = (info.samples_preview || []).filter((s) => {
    const src = typeof s === 'string' ? s : s?.image_b64;
    return !!src;
  });

  /* class chips with filter */
  const filteredClasses = classes.filter((c) =>
    !search || c.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-6">

      {/* ── stats row ──────────────────────────────────────────────── */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Stat
          label="Total images"
          value={(info.total_images ?? info.total_test_images ?? 0).toLocaleString()}
          color="text-blue-700"
        />
        <Stat
          label="Classes"
          value={classes.length}
          sub={isCustom ? 'detected from folders' : 'benchmark classes'}
        />
        <Stat
          label="Corrupted / skipped"
          value={info.corrupted_images ?? 0}
          color={(info.corrupted_images ?? 0) > 0 ? 'text-red-600' : 'text-green-700'}
          sub={(info.corrupted_images ?? 0) > 0 ? 'removed before training' : 'all images valid'}
        />
        <Stat
          label="Dimensions"
          value={
            Array.isArray(info.image_dimensions)
              ? `${info.image_dimensions[0]}×${info.image_dimensions[1]}`
              : typeof info.image_dimensions === 'object' && info.image_dimensions !== null
              ? 'Varies'
              : info.image_dimensions ?? '—'
          }
          sub="pixels (W×H)"
        />
      </div>

      {/* ── integrity banner ───────────────────────────────────────── */}
      {(info.corrupted_images ?? 0) > 0 ? (
        <div className="flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 p-4">
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
          <div>
            <p className="text-sm font-semibold text-amber-800">
              {info.corrupted_images} corrupted or unsupported image{info.corrupted_images !== 1 ? 's' : ''} detected
            </p>
            <p className="mt-0.5 text-xs text-amber-700">
              These files were skipped during extraction and will not appear in training or preview.
              Only valid JPG, PNG, BMP, and WebP images are retained.
            </p>
          </div>
        </div>
      ) : (
        <div className="flex items-center gap-3 rounded-xl border border-green-200 bg-green-50 p-4">
          <ShieldCheck className="h-4 w-4 shrink-0 text-green-600" />
          <p className="text-sm font-semibold text-green-800">All images validated — no corrupted files detected.</p>
        </div>
      )}

      {/* ── split info ─────────────────────────────────────────────── */}
      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <p className="text-sm font-semibold text-slate-800">Training / Validation split</p>
        <SplitBar trainPct={trainPct} valPct={valPct} />
        <div className="mt-3 grid grid-cols-3 gap-3 text-center text-xs text-slate-600">
          <div className="rounded-lg bg-blue-50 p-2">
            <p className="font-bold text-blue-700">{Math.round((info.total_images ?? 0) * trainPct / 100)}</p>
            <p>Train images</p>
          </div>
          <div className="rounded-lg bg-amber-50 p-2">
            <p className="font-bold text-amber-700">{Math.round((info.total_images ?? 0) * valPct / 100)}</p>
            <p>Val images</p>
          </div>
          <div className="rounded-lg bg-slate-100 p-2">
            <p className="font-bold text-slate-700">
              {Math.round((info.total_images ?? 0) * Math.max(0, 100 - trainPct - valPct) / 100)}
            </p>
            <p>Test images</p>
          </div>
        </div>
        <p className="mt-2 text-[11px] text-slate-400">
          These are recommended splits. Actual training split is set via the Training configuration below.
        </p>
      </div>

      {/* ── class distribution chart + chips ───────────────────────── */}
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <p className="text-sm font-semibold text-slate-800 mb-3">Class distribution</p>
          {chartData.length > 0 ? (
            <div className="h-56">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData} margin={{ top: 0, right: 8, bottom: 20, left: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                  <XAxis dataKey="name" tick={{ fontSize: 9 }} angle={-30} textAnchor="end" interval={0} />
                  <YAxis tick={{ fontSize: 10 }} />
                  <Tooltip />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {chartData.map((entry, i) => (
                      <Cell key={i} fill={BAR_COLORS[i % BAR_COLORS.length]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          ) : (
            <p className="text-sm text-slate-500">No distribution data available.</p>
          )}
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between mb-3">
            <p className="text-sm font-semibold text-slate-800">Classes ({classes.length})</p>
            <div className="relative">
              <Search className="absolute left-2 top-1/2 -translate-y-1/2 h-3 w-3 text-slate-400" />
              <input
                type="text"
                placeholder="Filter…"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                className="rounded-lg border border-slate-200 bg-slate-50 py-1 pl-6 pr-3 text-xs outline-none focus:border-blue-400 w-32"
              />
            </div>
          </div>
          <div className="flex flex-wrap gap-1.5 max-h-52 overflow-y-auto">
            {filteredClasses.map((c, i) => (
              <button
                key={c}
                onClick={() => setClassFilter(classFilter === c ? null : c)}
                className={`inline-flex items-center gap-1 rounded-full px-2.5 py-1 text-xs font-medium transition-colors ${
                  classFilter === c
                    ? 'text-white shadow-sm'
                    : 'bg-slate-100 text-slate-700 hover:bg-slate-200'
                }`}
                style={classFilter === c ? { backgroundColor: BAR_COLORS[i % BAR_COLORS.length] } : {}}
              >
                {c}
                {dist[c] != null && <span className="font-mono opacity-75">· {dist[c]}</span>}
              </button>
            ))}
          </div>
          {classFilter && (
            <button
              onClick={() => setClassFilter(null)}
              className="mt-3 flex items-center gap-1 text-xs text-slate-500 hover:text-slate-700"
            >
              <X className="h-3 w-3" /> Clear filter
            </button>
          )}
        </div>
      </div>

      {/* ── image preview grid ─────────────────────────────────────── */}
      {previews.length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <p className="text-sm font-semibold text-slate-800">
              Sample preview{classFilter ? ` — ${classFilter}` : ''}
            </p>
            <span className="text-xs text-slate-400">{previews.length} sample{previews.length !== 1 ? 's' : ''}</span>
          </div>
          <div className="grid grid-cols-3 gap-3 sm:grid-cols-4 md:grid-cols-6">
            {previews
              .filter((s) => !classFilter || (typeof s === 'object' && s.class_name === classFilter))
              .slice(0, 24)
              .map((s, i) => {
                const src  = typeof s === 'string' ? s : s.image_b64 || s.image_base64;
                const lbl  = typeof s === 'object' ? s.class_name : null;
                const idx  = typeof s === 'object' ? s.index : i;
                return (
                  <div key={i} className="group relative overflow-hidden rounded-lg border border-slate-100 bg-slate-50">
                    <img
                      src={src}
                      alt={lbl ?? `sample-${i}`}
                      className="aspect-square w-full object-cover transition-transform group-hover:scale-105"
                      loading="lazy"
                    />
                    {lbl && (
                      <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/70 to-transparent px-1.5 pb-1 pt-3">
                        <p className="truncate text-center text-[9px] font-semibold text-white">{lbl}</p>
                      </div>
                    )}
                  </div>
                );
              })}
          </div>
          {classFilter && previews.filter((s) => typeof s === 'object' && s.class_name === classFilter).length === 0 && (
            <p className="mt-3 text-xs text-slate-500 text-center">
              No preview images available for class "{classFilter}" in the sample cache.
            </p>
          )}
        </div>
      )}
    </div>
  );
}

/* ─── training panel (unchanged logic, improved UI) ──────────────────────── */
function TrainingPanel({
  config, setConfig, jobs, customModels, onStart, onStop, starting,
  selectedTypes, setSelectedTypes, recommendations, architectures = [], datasets = [],
}) {
  const selectedNames = selectedTypes.map((type) =>
    architectures.find((arch) => arch.model_type === type)?.display_name || type
  );

  return (
    <div className="space-y-6">
      {/* Model & Dataset Selection Banner */}
      <div className="flex flex-wrap items-center justify-between gap-4 p-4 rounded-xl border border-slate-200 bg-white shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-lg bg-blue-50 border border-blue-200 flex items-center justify-center text-blue-600">
            <FlaskConical className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-sm font-bold text-slate-900">Training Workbench</h3>
            <p className="text-xs text-slate-500">
              Training target: <span className="font-semibold text-slate-700">{selectedNames.join(', ') || 'Select an architecture'}</span> on <span className="font-semibold text-slate-700">{config.dataset_id}</span>
            </p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          <label className="text-xs font-semibold text-slate-600 flex items-center gap-1.5">
            <Database className="w-3.5 h-3.5 text-slate-400" />
            Target Dataset:
            <select
              className="ml-1 rounded-lg border border-slate-300 bg-slate-50 px-2.5 py-1 text-xs font-medium text-slate-800 outline-none focus:border-blue-500"
              value={config.dataset_id}
              onChange={(e) => setConfig({ ...config, dataset_id: e.target.value })}
            >
              {datasets.map((d) => (
                <option key={d.id} value={d.id}>
                  {d.name} ({d.classes_count ?? 10} classes)
                </option>
              ))}
            </select>
          </label>
        </div>
      </div>

      {/* Dynamic Model Architecture Selection Cards */}
      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex items-center justify-between mb-3">
          <div>
            <p className="text-sm font-semibold text-slate-800">1. Select Architectures</p>
            <p className="text-xs text-slate-500">Select one or more architectures. Jobs run in parallel using the settings below.</p>
          </div>
          <span className="text-xs font-medium text-blue-600 bg-blue-50 px-2.5 py-1 rounded-full">
            {selectedTypes.length} selected
          </span>
        </div>

        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {architectures.map((arch) => {
            const isSelected = selectedTypes.includes(arch.model_type);
            return (
              <button
                key={arch.model_type}
                type="button"
                aria-pressed={isSelected}
                onClick={() => setSelectedTypes((types) =>
                  types.includes(arch.model_type)
                    ? types.filter((type) => type !== arch.model_type)
                    : [...types, arch.model_type]
                )}
                className={`p-3.5 rounded-xl border text-left transition-all ${
                  isSelected
                    ? 'border-blue-500 bg-blue-50/70 ring-2 ring-blue-200 shadow-sm'
                    : 'border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50'
                }`}
              >
                <div className="flex items-start justify-between gap-1">
                  <span className="font-semibold text-xs text-slate-900 truncate">
                    {arch.display_name}
                  </span>
                  <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
                    arch.category.includes('Lightweight') || arch.category.includes('Compact')
                      ? 'bg-emerald-50 text-emerald-700'
                      : arch.category.includes('Efficient')
                      ? 'bg-cyan-50 text-cyan-700'
                      : 'bg-indigo-50 text-indigo-700'
                  }`}>
                    {arch.category}
                  </span>
                </div>
                <p className="mt-1.5 text-[11px] text-slate-500 line-clamp-2 leading-relaxed">
                  {arch.description}
                </p>
                <div className="mt-2.5 flex items-center justify-between text-[10px] text-slate-400 border-t border-slate-100 pt-2">
                  <span>~{arch.params_m}M params</span>
                  <span>Default: {arch.input_size}×{arch.input_size}</span>
                </div>
              </button>
            );
          })}
        </div>
      </div>

      {/* hardware recommendation */}
      {recommendations && (
        <div className="rounded-xl border border-blue-100 bg-blue-50 p-4">
          <div className="flex items-start gap-3">
            <Info className="mt-0.5 h-4 w-4 shrink-0 text-blue-600" />
            <div className="flex-1">
              <div className="flex items-center justify-between">
                <p className="text-sm font-semibold text-blue-800">{recommendations.hardware_tier}</p>
                <span className="text-[11px] font-mono font-medium text-blue-700 bg-blue-100/60 px-2 py-0.5 rounded">
                  {recommendations.device_name || 'System CPU'}
                </span>
              </div>
              <p className="mt-0.5 text-xs text-blue-700">{recommendations.hardware_advisory}</p>
              <dl className="mt-2 grid grid-cols-3 gap-2 text-xs">
                <div><dt className="text-blue-600">Recommended Device</dt><dd className="font-semibold text-blue-900 uppercase">{recommendations.recommended_device}</dd></div>
                <div><dt className="text-blue-600">Safe Batch Size</dt><dd className="font-semibold text-blue-900">{recommendations.recommended_batch_size}</dd></div>
                <div><dt className="text-blue-600">Available VRAM</dt><dd className="font-semibold text-blue-900">{recommendations.total_vram_mb ? `${Math.round(recommendations.total_vram_mb)} MB` : 'N/A'}</dd></div>
              </dl>
            </div>
          </div>
        </div>
      )}

      {/* config form */}
      <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-4">
        <div className="flex items-center justify-between">
          <p className="text-sm font-semibold text-slate-800 flex items-center gap-2">
            <Sliders className="w-4 h-4 text-slate-600" />
            2. Configure Hyperparameters & Environment
          </p>
          <span className="text-xs text-slate-400">Default batch size: 4 (edge memory-safe)</span>
        </div>

        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          <label className="text-sm font-medium text-slate-700">
            Batch Size
            <input
              type="number"
              min="1"
              max="32"
              className={fieldClass}
              value={config.batch_size}
              onChange={(e) => setConfig({ ...config, batch_size: Math.max(1, Math.min(32, Number(e.target.value))) })}
            />
            <span className="text-[10px] text-slate-400">Default 4 (range 1 - 32)</span>
          </label>

          <label className="text-sm font-medium text-slate-700">
            Number of Epochs
            <input
              type="number"
              min="1"
              max="100"
              className={fieldClass}
              value={config.epochs}
              onChange={(e) => setConfig({ ...config, epochs: Math.max(1, Math.min(100, Number(e.target.value))) })}
            />
            <span className="text-[10px] text-slate-400">Full training cycle count</span>
          </label>

          <label className="text-sm font-medium text-slate-700">
            Learning Rate
            <input
              type="number"
              step="0.0001"
              min="0.00001"
              max="0.1"
              className={fieldClass}
              value={config.learning_rate}
              onChange={(e) => setConfig({ ...config, learning_rate: Number(e.target.value) })}
            />
            <span className="text-[10px] text-slate-400">Step size (e.g. 0.001)</span>
          </label>

          <label className="text-sm font-medium text-slate-700">
            Optimizer
            <select className={fieldClass} value={config.optimizer_type} onChange={(e) => setConfig({ ...config, optimizer_type: e.target.value })}>
              <option value="adam">Adam (Adaptive Moment)</option>
              <option value="adamw">AdamW (Decoupled Weight Decay)</option>
              <option value="sgd">SGD with Momentum</option>
            </select>
            <span className="text-[10px] text-slate-400">Parameter update algorithm</span>
          </label>

          <label className="text-sm font-medium text-slate-700">
            Image Size (px)
            <select
              className={fieldClass}
              value={config.image_size ?? ''}
              onChange={(e) => setConfig({ ...config, image_size: e.target.value ? Number(e.target.value) : null })}
            >
              <option value="">Architecture default</option>
              <option value="128">128 × 128 (Ultra Fast / Low RAM)</option>
              <option value="224">224 × 224 (Standard ImageNet)</option>
              <option value="256">256 × 256 (High Resolution)</option>
              <option value="300">300 × 300 (EfficientNet-B3 Native)</option>
            </select>
            <span className="text-[10px] text-slate-400">Resized before tensor normalization</span>
          </label>

          <label className="text-sm font-medium text-slate-700">
            Device Execution
            <select className={fieldClass} value={config.device_selection} onChange={(e) => setConfig({ ...config, device_selection: e.target.value })}>
              <option value="auto">Auto (CUDA if ready, else CPU)</option>
              <option value="cuda">GPU / CUDA Dedicated</option>
              <option value="cpu">System CPU Only</option>
            </select>
            <span className="text-[10px] text-slate-400">Compute hardware target</span>
          </label>

          <label className="text-sm font-medium text-slate-700">
            Validation Split
            <input
              type="number"
              min="0.05"
              max="0.80"
              step="0.05"
              className={fieldClass}
              value={config.val_split}
              onChange={(e) => setConfig({ ...config, val_split: Number(e.target.value) })}
            />
            <span className="text-[10px] text-slate-400">Ratio held out for validation (e.g. 0.15 = 15%)</span>
          </label>
        </div>

        <div className="border-t border-slate-100 pt-3 flex flex-wrap items-center justify-between gap-3">
          <label className="text-xs text-slate-600 flex items-center gap-2">
            Training Sample Limit (subset for fast prototyping):
            <input
              type="number"
              min="20"
              placeholder="e.g. 500 (blank for all)"
              className="w-32 rounded-lg border border-slate-300 bg-white px-2 py-1 text-xs outline-none focus:border-blue-500"
              value={config.train_limit || ''}
              onChange={(e) => setConfig({ ...config, train_limit: e.target.value ? Number(e.target.value) : null })}
            />
          </label>

          <div className="flex gap-3">
            <button
              onClick={onStart}
              disabled={starting || selectedTypes.length === 0}
              className="rounded-lg bg-blue-600 px-5 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:bg-slate-300 flex items-center gap-1.5 shadow-sm"
            >
              <FlaskConical className="w-4 h-4" />
              {starting ? 'Starting…' : selectedTypes.length > 1 ? `Train ${selectedTypes.length} Models` : 'Start Training'}
            </button>
          </div>
        </div>
      </div>

      {jobs.length > 0 && (
        <section className="space-y-4" aria-label="Training jobs">
          <h3 className="text-sm font-bold text-slate-900">Training Jobs</h3>
          {jobs.map((job) => (
            <TrainingJob
              key={job.job_id}
              telemetry={job}
              name={architectures.find((arch) => arch.model_type === job.config.model_type)?.display_name || job.config.model_type}
              onStop={onStop}
            />
          ))}
        </section>
      )}

      {/* session candidate models */}
      {customModels.length > 0 && (
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center justify-between mb-3">
            <p className="text-sm font-semibold text-slate-800">
              Session Candidate Models ({customModels.length})
            </p>
            <span className="text-xs text-slate-400">Preserved in RAM until backend restart</span>
          </div>
          <div className="grid gap-3 md:grid-cols-2">
            {customModels.map((m) => (
              <div key={m.run_id} className="rounded-lg border border-slate-100 bg-slate-50 p-4 hover:border-slate-200 transition-all">
                <div className="flex items-start justify-between">
                  <p className="text-xs font-semibold text-slate-900 font-mono truncate max-w-[200px]">{m.run_id}</p>
                  <span className="text-[10px] font-semibold bg-blue-50 text-blue-700 px-2 py-0.5 rounded-full">
                    {m.architecture_display || m.architecture}
                  </span>
                </div>
                <p className="mt-1 text-xs text-slate-500">Dataset: {m.dataset_id} · {m.class_names?.length ?? '?'} classes</p>
                <div className="mt-2.5 flex items-center justify-between text-xs border-t border-slate-200/70 pt-2">
                  <span className="text-slate-600">Best Validation:</span>
                  <span className="font-bold text-blue-700 font-mono">
                    {m.metrics?.best_val_accuracy != null ? `${m.metrics.best_val_accuracy.toFixed(1)}%` : '—'}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function TrainingJob({ telemetry, name, onStop }) {
  const trainingHist = telemetry.history || [];
  const [requestingStop, setRequestingStop] = useState(false);
  const stopping = telemetry.status === 'RUNNING' && (requestingStop || telemetry.stop_requested);

  const stop = async () => {
    setRequestingStop(true);
    if (!(await onStop(telemetry.job_id))) setRequestingStop(false);
  };

  return (
    <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-blue-600" />
          <div>
            <h4 className="text-sm font-semibold text-slate-800">{name}</h4>
            <p className="text-xs text-slate-500">{telemetry.config.dataset_id} · {telemetry.job_id}</p>
          </div>
        </div>
        <span className={`rounded-full px-3 py-1 text-xs font-semibold ${
          telemetry.status === 'RUNNING'   ? 'bg-blue-50 text-blue-700 animate-pulse'   :
          telemetry.status === 'COMPLETED' ? 'bg-green-50 text-green-700' :
          telemetry.status === 'FAILED'    ? 'bg-red-50 text-red-700'     :
          telemetry.status === 'STOPPED'   ? 'bg-amber-50 text-amber-700' :
          'bg-slate-100 text-slate-600'
        }`}>{stopping ? 'Stopping…' : telemetry.status}</span>
        {telemetry.status === 'RUNNING' && (
          <button
            type="button"
            onClick={stop}
            disabled={stopping}
            className="rounded-lg border border-red-200 px-3 py-1 text-xs font-semibold text-red-700 hover:bg-red-50 disabled:opacity-50"
          >
            Stop
          </button>
        )}
      </div>

      <div className="space-y-1">
        <div className="flex justify-between text-xs text-slate-500">
          <span>Overall Progress</span>
          <span>{telemetry.progress_pct || 0}% (Epoch {telemetry.current_epoch}/{telemetry.total_epochs})</span>
        </div>
        <div className="h-2.5 w-full overflow-hidden rounded-full bg-slate-100">
          <div className="h-full bg-blue-600 transition-all duration-300" style={{ width: `${telemetry.progress_pct || 0}%` }} />
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3 text-xs md:grid-cols-6">
        {[
          { label: 'Current Epoch',  value: `${telemetry.current_epoch} / ${telemetry.total_epochs}` },
          { label: 'Train Loss',     value: telemetry.train_loss != null ? telemetry.train_loss.toFixed(4) : '—' },
          { label: 'Val Loss',       value: telemetry.val_loss != null ? telemetry.val_loss.toFixed(4) : '—' },
          { label: 'Train Accuracy', value: telemetry.train_acc != null ? `${telemetry.train_acc.toFixed(1)}%` : '—' },
          { label: 'Val Accuracy',   value: telemetry.val_acc != null ? `${telemetry.val_acc.toFixed(1)}%` : '—' },
          { label: 'Elapsed Time',   value: `${telemetry.elapsed_seconds ?? 0}s` },
        ].map((m) => (
          <div key={m.label} className="rounded-lg bg-slate-50 border border-slate-100 p-2.5 text-center">
            <p className="text-slate-500 text-[11px]">{m.label}</p>
            <p className="mt-1 font-bold text-slate-900 text-sm font-mono">{m.value}</p>
          </div>
        ))}
      </div>

      {/* Epoch-wise Loss Graph */}
      {trainingHist.length > 0 && (
        <div className="border border-slate-100 rounded-xl p-4 bg-slate-50/50 space-y-2">
          <div className="flex items-center justify-between text-xs">
            <span className="font-semibold text-slate-700">Epoch-wise Loss & Accuracy Curve</span>
            <span className="text-slate-400 font-mono">{trainingHist.length} recorded epochs</span>
          </div>
          <div className="h-52 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trainingHist} margin={{ top: 10, right: 20, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="epoch" tick={{ fontSize: 11 }} label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fontSize: 10 }} />
                <YAxis yAxisId="loss" tick={{ fontSize: 11 }} label={{ value: 'Loss', angle: -90, position: 'insideLeft', fontSize: 10 }} />
                <YAxis yAxisId="acc" orientation="right" unit="%" tick={{ fontSize: 11 }} domain={[0, 100]} />
                <Tooltip contentStyle={{ fontSize: '11px', borderRadius: '8px' }} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Line yAxisId="loss" type="monotone" dataKey="train_loss" name="Train Loss" stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} />
                <Line yAxisId="loss" type="monotone" dataKey="val_loss" name="Val Loss" stroke="#f97316" strokeWidth={2} dot={{ r: 3 }} strokeDasharray="4 4" />
                <Line yAxisId="acc" type="monotone" dataKey="val_acc" name="Val Acc (%)" stroke="#2563eb" strokeWidth={2} dot={{ r: 3 }} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      {telemetry.error_message && (
        <div className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 p-3">
          <XCircle className="h-4 w-4 shrink-0 text-red-600 mt-0.5" />
          <p className="text-xs text-red-700">{telemetry.error_message}</p>
        </div>
      )}

      {telemetry.candidate_model_id && (
        <div className="rounded-lg border border-green-200 bg-green-50 px-3.5 py-2.5 flex items-center justify-between">
          <div>
            <p className="text-xs font-semibold text-green-800">Candidate Model Ready in Session RAM</p>
            <p className="text-xs font-mono text-green-700 mt-0.5 truncate">{telemetry.candidate_model_id}</p>
          </div>
          <span className="text-[10px] bg-green-200/70 text-green-800 font-semibold px-2 py-0.5 rounded-full">
            Inference Ready
          </span>
        </div>
      )}

      {telemetry.logs?.length > 0 && (
        <div className="max-h-40 overflow-auto rounded-xl bg-slate-950 p-3 font-mono text-[11px] text-slate-100 leading-relaxed shadow-inner">
          {telemetry.logs.map((line, i) => <div key={i}>{line}</div>)}
        </div>
      )}
    </div>
  );
}


/* ─── main page ───────────────────────────────────────────────────────────── */
const TABS = [
  { id: 'datasets',  label: 'Datasets',          icon: Database    },
  { id: 'upload',    label: 'Upload Dataset',     icon: UploadCloud },
  { id: 'training',  label: 'Train Models',      icon: FlaskConical },
];

export default function DataTrain({ datasetId = 'cifar10', onDatasetChange }) {
  const [tab,          setTab]          = useState('datasets');
  const [datasets,     setDatasets]     = useState([]);
  const [architectures, setArchitectures] = useState([]);
  const [selected,     setSelected]     = useState(datasetId);
  const [info,         setInfo]         = useState(null);
  const [infoLoading,  setInfoLoading]  = useState(false);
  const [recommendations, setRec]       = useState(null);
  const [jobs, setJobs] = useState([]);
  const [starting, setStarting] = useState(false);
  const [selectedTypes, setSelectedTypes] = useState(['mobilenet_v3_small']);
  const [customModels, setCustomModels] = useState([]);
  const [error,        setError]        = useState('');
  const [uploading,    setUploading]    = useState(false);
  const [uploadPct,    setUploadPct]    = useState(0);
  const [uploadResult, setUploadResult] = useState(null);

  const [config, setConfig] = useState({
    dataset_id: datasetId, epochs: 3, batch_size: 4,
    learning_rate: 0.001, optimizer_type: 'adam', device_selection: 'auto',
    val_split: 0.15, train_limit: 500,
  });

  /* ── loaders ─────────────────────────────────────────────────────── */
  useEffect(() => {
    onDatasetChange?.(config.dataset_id);
  }, [config.dataset_id, onDatasetChange]);

  const loadDatasets = useCallback(async () => {
    try {
      const data = await api.getDatasets();
      setDatasets(data.datasets || []);
    } catch (e) { setError(e.message); }
  }, []);

  const loadArchitectures = useCallback(async () => {
    try {
      const data = await api.getArchitectures();
      if (data && data.architectures) {
        setArchitectures(data.architectures);
      }
    } catch (e) {
      console.warn("Failed to load architectures from backend, using defaults", e);
      setArchitectures([
        { model_type: 'mobilenet_v3_small', display_name: 'MobileNetV3-Small', category: 'Lightweight', params_m: 2.5, description: 'Fast and compact. Optimised for edge inference with low VRAM footprint.', input_size: 224 },
        { model_type: 'mobilenet_v3_large', display_name: 'MobileNetV3-Large', category: 'Efficient', params_m: 5.4, description: 'Larger MobileNet variant. Better accuracy with moderate compute cost.', input_size: 224 },
        { model_type: 'shufflenet_v2', display_name: 'ShuffleNetV2', category: 'Lightweight', params_m: 2.3, description: 'Channel-shuffle architecture. Very fast on CPU with minimal parameters.', input_size: 224 },
        { model_type: 'resnet18', display_name: 'ResNet-18', category: 'Residual CNN', params_m: 11.7, description: 'Classic residual network. Reliable accuracy, well understood behaviour.', input_size: 224 },
        { model_type: 'resnet50', display_name: 'ResNet-50', category: 'Deeper CNN', params_m: 25.6, description: 'Deeper residual network. Higher accuracy, requires more compute and VRAM.', input_size: 224 },
        { model_type: 'efficientnet_b0', display_name: 'EfficientNet-B0', category: 'Efficient CNN', params_m: 5.3, description: 'Compound-scaled network. Strong accuracy-to-parameter ratio.', input_size: 224 },
        { model_type: 'efficientnet_b3', display_name: 'EfficientNet-B3', category: 'Larger CNN', params_m: 12.2, description: 'Larger EfficientNet variant. Best accuracy in this family, more VRAM needed.', input_size: 300 },
        { model_type: 'squeezenet', display_name: 'SqueezeNet', category: 'Compact CNN', params_m: 1.2, description: 'Fire-module architecture. Fewest parameters; suitable for very constrained devices.', input_size: 224 },
      ]);
    }
  }, []);

  const inspect = useCallback(async (id) => {
    setInfoLoading(true);
    setError('');
    try {
      const data = await api.inspectDataset(id);
      setInfo(data);
      setSelected(id);
      setConfig((c) => ({ ...c, dataset_id: id }));
    } catch (e) { setError(e.message); setInfo(null); }
    finally { setInfoLoading(false); }
  }, []);

  useEffect(() => {
    loadDatasets();
    loadArchitectures();
    api.getTrainingRecommendations().then(setRec).catch(() => {});
  }, [loadDatasets, loadArchitectures]);

  useEffect(() => {
    if (tab !== 'training') return;
    let cancelled = false;
    let timer;
    let finishedRuns;

    const poll = async () => {
      try {
        const data = await api.listTrainingJobs();
        if (cancelled) return;
        setJobs([...data.jobs].reverse());

        const runIds = data.jobs.map((job) => job.candidate_model_id).filter(Boolean).join(',');
        if (runIds !== finishedRuns) {
          const candidates = await api.getCustomModels();
          if (cancelled) return;
          setCustomModels(candidates.models || []);
          finishedRuns = runIds;
        }
      } catch (e) {
        if (!cancelled) setError(e.message);
      } finally {
        if (!cancelled) timer = setTimeout(poll, 1500);
      }
    };

    poll();
    return () => {
      cancelled = true;
      clearTimeout(timer);
    };
  }, [tab]);

  useEffect(() => { inspect(selected); }, [selected]);

  /* ── upload handler ──────────────────────────────────────────────── */
  const handleUpload = (file) => {
    if (!file) return;
    setUploading(true);
    setUploadPct(1);
    setUploadResult(null);
    setError('');
    const form = new FormData();
    form.append('archive', file);
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/api/datasets/upload');
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) setUploadPct(Math.round(e.loaded / e.total * 90));
    };
    xhr.onload = async () => {
      setUploading(false);
      setUploadPct(0);
      let body;
      try { body = JSON.parse(xhr.responseText); } catch { body = { detail: 'Invalid response' }; }
      if (xhr.status >= 200 && xhr.status < 300 && body.success) {
        setUploadResult({ success: true, ...body });
        await loadDatasets();
        await inspect(body.dataset_id);
        setTab('datasets');
      } else {
        setUploadResult({ success: false, error: body.detail || `Upload failed (${xhr.status})` });
      }
    };
    xhr.onerror = () => { setUploading(false); setUploadPct(0); setUploadResult({ success: false, error: 'Network error during upload.' }); };
    xhr.send(form);
  };

  /* ── training actions ────────────────────────────────────────────── */
  const startTraining = async () => {
    setError('');
    setStarting(true);
    try {
      const results = await Promise.allSettled(selectedTypes.map((model_type) =>
        api.startTrainingJob({ ...config, model_type })
      ));
      const errors = results.flatMap((result, index) => {
        if (result.status === 'fulfilled') return [];
        const type = selectedTypes[index];
        const name = architectures.find((arch) => arch.model_type === type)?.display_name || type;
        return [`${name} could not start. ${result.reason.message}`];
      });
      if (errors.length) setError(errors.join(' '));
    } finally {
      setStarting(false);
    }
  };

  const stopTraining = async (jobId) => {
    try {
      const result = await api.stopJob(jobId);
      return result.success;
    } catch (e) {
      setError(e.message);
      return false;
    }
  };

  const cifar10 = datasets.find((d) => d.id === 'cifar10');
  const uploadedDS = datasets.filter((d) => d.id !== 'cifar10');

  return (
    <div className="mx-auto max-w-7xl space-y-6 p-6">

      {/* ── page header ─────────────────────────────────────────────── */}
      <div>
        <h1 className="text-2xl font-bold text-slate-950">Dataset &amp; Training</h1>
        <p className="mt-1 text-sm text-slate-500">
          Manage datasets, upload custom image archives, and train candidate models for ADAPT-EDGE.
        </p>
      </div>

      {/* session memory notice */}
      <div className="flex items-start gap-3 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3">
        <Info className="mt-0.5 h-4 w-4 shrink-0 text-amber-600" />
        <p className="text-xs text-amber-900">
          Candidate models are stored in backend RAM and cleared on restart. Research checkpoints and historical results are not affected.
        </p>
      </div>

      {error && (
        <div className="flex items-start gap-2 rounded-lg border border-red-200 bg-red-50 px-4 py-3">
          <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-600" />
          <p className="text-sm text-red-700">{error}</p>
        </div>
      )}

      {/* ── tab bar ─────────────────────────────────────────────────── */}
      <div className="flex gap-1 rounded-xl border border-slate-200 bg-white p-1 shadow-sm w-fit">
        {TABS.map((t) => {
          const Icon = t.icon;
          return (
            <button
              key={t.id}
              onClick={() => setTab(t.id)}
              className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-all ${
                tab === t.id
                  ? 'bg-blue-600 text-white shadow-sm'
                  : 'text-slate-600 hover:bg-slate-50 hover:text-slate-900'
              }`}
            >
              <Icon className="h-4 w-4" />
              {t.label}
            </button>
          );
        })}
      </div>

      {/* ── DATASETS tab ────────────────────────────────────────────── */}
      {tab === 'datasets' && (
        <div className="space-y-6">

          {/* dataset selector row */}
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {/* CIFAR-10 card */}
            <button
              onClick={() => setSelected('cifar10')}
              className={`rounded-xl border p-4 text-left transition-all shadow-sm ${
                selected === 'cifar10'
                  ? 'border-blue-400 bg-blue-50 ring-2 ring-blue-200'
                  : 'border-slate-200 bg-white hover:border-blue-300'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Database className="h-4 w-4 text-blue-500" />
                  <p className="text-sm font-semibold text-slate-800">CIFAR-10</p>
                </div>
                <StatusPill status={cifar10?.status ?? 'not_found'} />
              </div>
              <p className="mt-2 text-xs text-slate-500">10 classes · 60,000 images · 32×32</p>
              <p className="text-xs text-slate-400">Benchmark research dataset</p>
            </button>

            {/* uploaded datasets */}
            {uploadedDS.map((ds) => (
              <button
                key={ds.id}
                onClick={() => setSelected(ds.id)}
                className={`rounded-xl border p-4 text-left transition-all shadow-sm ${
                  selected === ds.id
                    ? 'border-blue-400 bg-blue-50 ring-2 ring-blue-200'
                    : 'border-slate-200 bg-white hover:border-blue-300'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <FolderOpen className="h-4 w-4 text-violet-500" />
                    <p className="text-sm font-semibold text-slate-800 truncate max-w-[130px]">{ds.name}</p>
                  </div>
                  <StatusPill status={ds.status} />
                </div>
                <p className="mt-2 text-xs text-slate-500">
                  {ds.classes_count ?? '?'} classes · {ds.total_samples ?? '?'} images
                </p>
                <p className="text-xs text-slate-400">User-uploaded dataset</p>
              </button>
            ))}

            {/* add dataset CTA */}
            <button
              onClick={() => setTab('upload')}
              className="flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed border-slate-200 bg-white p-4 text-slate-400 hover:border-blue-300 hover:text-blue-500 transition-colors"
            >
              <UploadCloud className="h-6 w-6" />
              <p className="text-xs font-medium">Upload new dataset</p>
            </button>
          </div>

          {/* inspect panel */}
          {infoLoading ? (
            <div className="flex items-center gap-2 py-8 text-sm text-slate-500">
              <RefreshCw className="h-4 w-4 animate-spin" /> Loading dataset info…
            </div>
          ) : (
            <DatasetInspect info={info} isCustom={selected !== 'cifar10'} />
          )}
        </div>
      )}

      {/* ── UPLOAD tab ──────────────────────────────────────────────── */}
      {tab === 'upload' && (
        <div className="max-w-2xl space-y-4">
          <div>
            <h2 className="text-base font-semibold text-slate-900">Upload a custom dataset</h2>
            <p className="mt-1 text-xs text-slate-500">
              Datasets are stored separately from protected research checkpoints and do not affect CIFAR-10 baselines.
            </p>
          </div>
          <UploadZone
            onUpload={handleUpload}
            uploading={uploading}
            uploadProgress={uploadPct}
            uploadResult={uploadResult}
          />
        </div>
      )}

      {/* ── TRAINING tab ────────────────────────────────────────────── */}
      {tab === 'training' && (
        <TrainingPanel
          config={config}
          setConfig={setConfig}
          jobs={jobs}
          starting={starting}
          selectedTypes={selectedTypes}
          setSelectedTypes={setSelectedTypes}
          customModels={customModels}
          onStart={startTraining}
          onStop={stopTraining}
          recommendations={recommendations}
          architectures={architectures}
          datasets={datasets}
        />
      )}
    </div>
  );
}
