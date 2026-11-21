import React, { useEffect, useState } from 'react';
import { api } from '../api';
import {
  Cpu, HardDrive, ShieldCheck, RefreshCw, Trash2, Sliders, CheckCircle2,
  AlertTriangle, XCircle, Zap, Activity, Database, Server, Lock
} from 'lucide-react';

export default function Settings() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [successMsg, setSuccessMsg] = useState('');
  
  // Settings state
  const [settings, setSettings] = useState(null);
  const [deviceConfig, setDeviceConfig] = useState({ device: 'cpu', cudnn_benchmark: false });
  const [trainingDefaults, setTrainingDefaults] = useState({
    batch_size: 4,
    epochs: 5,
    learning_rate: 0.001,
    optimizer: 'Adam',
    img_size: 224,
    val_split: 0.2,
    architecture: 'mobilenet_v3_small'
  });
  const [resourceBudget, setResourceBudget] = useState({
    latency_budget_ms: 25.0,
    ram_budget_mb: 2048.0,
    vram_budget_mb: 1024.0,
    cpu_throttle_threshold: 85.0,
    enforce_resource_budget: true,
    threshold_a: 0.85,
    threshold_probe: 0.80
  });
  const [backendConfig, setBackendConfig] = useState({
    telemetry_poll_interval_ms: 1000,
    cache_retention_limit: 100,
    simulation_mode: false
  });

  // Integrity state
  const [integrityData, setIntegrityData] = useState(null);
  const [verifying, setVerifying] = useState(false);
  const [cleaningSession, setCleaningSession] = useState(false);

  const loadAll = async () => {
    setLoading(true);
    setError('');
    try {
      const data = await api.getSettings();
      setSettings(data);
      if (data.device) {
        setDeviceConfig({
          device: data.device.active_device || 'cpu',
          cudnn_benchmark: !!data.device.cudnn_benchmark
        });
      }
      if (data.training_defaults) {
        setTrainingDefaults(prev => ({ ...prev, ...data.training_defaults }));
      }
      if (data.resource_budget) {
        setResourceBudget(prev => ({ ...prev, ...data.resource_budget }));
      }
      if (data.backend) {
        setBackendConfig(prev => ({ ...prev, ...data.backend }));
      }
    } catch (err) {
      setError(err.message || 'Failed to load system settings');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadAll();
  }, []);

  const handleDeviceSave = async (e) => {
    e.preventDefault();
    setError('');
    setSuccessMsg('');
    try {
      const res = await api.configureDevice(deviceConfig);
      setSuccessMsg(res.message || 'Device configuration updated successfully.');
      loadAll();
    } catch (err) {
      setError(err.message || 'Error updating device configuration.');
    }
  };

  const handleTrainingDefaultsSave = async (e) => {
    e.preventDefault();
    setError('');
    setSuccessMsg('');
    try {
      const res = await api.updateTrainingDefaults(trainingDefaults);
      setSuccessMsg(res.message || 'Training defaults updated.');
      loadAll();
    } catch (err) {
      setError(err.message || 'Error saving training defaults.');
    }
  };

  const handleResourceBudgetSave = async (e) => {
    e.preventDefault();
    setError('');
    setSuccessMsg('');
    try {
      const res = await api.updateResourceBudget(resourceBudget);
      setSuccessMsg(res.message || 'Resource budget configuration saved.');
      loadAll();
    } catch (err) {
      setError(err.message || 'Error saving resource budget.');
    }
  };

  const handleBackendSave = async (e) => {
    e.preventDefault();
    setError('');
    setSuccessMsg('');
    try {
      const res = await api.updateBackendSettings(backendConfig);
      setSuccessMsg(res.message || 'Backend telemetry settings updated.');
      loadAll();
    } catch (err) {
      setError(err.message || 'Error saving backend settings.');
    }
  };

  const handleSessionCleanup = async () => {
    if (!window.confirm('Are you sure you want to flush all in-memory candidate models, UI caches, and continuation caches?')) return;
    setCleaningSession(true);
    setError('');
    setSuccessMsg('');
    try {
      const res = await api.sessionCleanup();
      setSuccessMsg(res.status || 'Session memory cleanly reset.');
      loadAll();
    } catch (err) {
      setError(err.message || 'Session cleanup failed.');
    } finally {
      setCleaningSession(false);
    }
  };

  const handleRunIntegrityAudit = async () => {
    setVerifying(true);
    setError('');
    try {
      const data = await api.verifyProtectedCheckpoints();
      setIntegrityData(data);
    } catch (err) {
      setError(err.message || 'Failed to verify checkpoint integrity.');
    } finally {
      setVerifying(false);
    }
  };

  const device = settings?.device || {};
  const sessionStorage = settings?.session_storage || {};

  return (
    <div className="mx-auto max-w-7xl space-y-8 p-6 pb-16">
      {/* Header */}
      <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between border-b border-slate-200 pb-5">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">Platform Settings & Verification</h1>
          <p className="mt-1 text-sm text-slate-500">
            Phase 11: Hardware devices, default training parameters, resource bounds, RAM session lifecycle, and cryptographic audits.
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={loadAll}
            disabled={loading}
            className="inline-flex items-center gap-2 rounded-lg border border-slate-300 bg-white px-3 py-2 text-xs font-medium text-slate-700 shadow-sm hover:bg-slate-50"
          >
            <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
            Refresh Settings
          </button>
          <button
            onClick={handleSessionCleanup}
            disabled={cleaningSession}
            className="inline-flex items-center gap-2 rounded-lg bg-rose-600 px-3 py-2 text-xs font-semibold text-white shadow-sm hover:bg-rose-700"
          >
            <Trash2 className="h-3.5 w-3.5" />
            {cleaningSession ? 'Flushing...' : 'Purge Session Memory'}
          </button>
        </div>
      </div>

      {error && (
        <div className="flex items-center gap-2 rounded-lg border border-rose-200 bg-rose-50 px-4 py-3 text-sm text-rose-800">
          <AlertTriangle className="h-4 w-4 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {successMsg && (
        <div className="flex items-center gap-2 rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-800">
          <CheckCircle2 className="h-4 w-4 shrink-0" />
          <span>{successMsg}</span>
        </div>
      )}

      {/* Top Telemetry Strip */}
      <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">Active Device</span>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="text-xl font-bold uppercase text-slate-900">{device.active_device || 'Unknown'}</span>
            <span className="text-xs font-medium text-slate-500">{device.cuda_available ? 'CUDA Ready' : 'CPU Only'}</span>
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">RAM Candidates</span>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="text-xl font-bold text-slate-900">{sessionStorage.candidate_models_in_ram ?? 0}</span>
            <span className="text-xs font-medium text-emerald-600">RAM-only (0 files)</span>
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">Continuation Cache</span>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="text-xl font-bold text-slate-900">{sessionStorage.continuation_cache_entries ?? 0}</span>
            <span className="text-xs font-medium text-slate-500">Active features</span>
          </div>
        </div>

        <div className="rounded-xl border border-slate-200 bg-white p-4 shadow-sm">
          <span className="text-xs font-semibold uppercase tracking-wider text-slate-500">Research Isolation</span>
          <div className="mt-1 flex items-baseline gap-2">
            <span className="text-xl font-bold text-emerald-600">Strict</span>
            <span className="text-xs font-medium text-slate-500">Outputs Locked</span>
          </div>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Device & Compute Configuration */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
            <Cpu className="h-5 w-5 text-indigo-600" />
            <h2 className="font-semibold text-slate-900">Device & GPU Configuration</h2>
          </div>
          <form onSubmit={handleDeviceSave} className="mt-4 space-y-4">
            <div>
              <label className="text-xs font-medium text-slate-600">Execution Hardware Device</label>
              <select
                value={deviceConfig.device}
                onChange={(e) => setDeviceConfig({ ...deviceConfig, device: e.target.value })}
                className="mt-1 block w-full rounded-md border border-slate-300 bg-white px-3 py-2 text-sm shadow-sm focus:border-indigo-500 focus:outline-none"
              >
                {(device.available_devices || ['cpu']).map((d) => (
                  <option key={d} value={d}>
                    {d.toUpperCase()} {d.includes('cuda') ? `(${device.gpu_name || 'GPU'})` : '(Host Process)'}
                  </option>
                ))}
              </select>
            </div>

            <div className="flex items-center justify-between rounded-lg border border-slate-100 bg-slate-50 p-3">
              <div>
                <p className="text-xs font-semibold text-slate-800">cuDNN Benchmark Acceleration</p>
                <p className="text-xs text-slate-500">Enable torch.backends.cudnn.benchmark for static-shape autotuning.</p>
              </div>
              <input
                type="checkbox"
                checked={deviceConfig.cudnn_benchmark}
                onChange={(e) => setDeviceConfig({ ...deviceConfig, cudnn_benchmark: e.target.checked })}
                className="h-4 w-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
              />
            </div>

            <div className="rounded-lg bg-slate-50 p-3 text-xs text-slate-600 space-y-1">
              <div className="flex justify-between"><span>Compute Hardware:</span> <span className="font-medium text-slate-800">{device.gpu_name || 'CPU Host'}</span></div>
              <div className="flex justify-between"><span>PyTorch Version:</span> <span className="font-mono text-slate-800">{device.pytorch_version}</span></div>
              {device.cuda_available && (
                <>
                  <div className="flex justify-between"><span>CUDA Version:</span> <span className="font-mono text-slate-800">{device.cuda_version}</span></div>
                  <div className="flex justify-between"><span>Allocated VRAM:</span> <span className="font-mono text-slate-800">{device.gpu_memory_allocated_mb} MB</span></div>
                  <div className="flex justify-between"><span>Total VRAM:</span> <span className="font-mono text-slate-800">{device.gpu_total_memory_mb} MB</span></div>
                </>
              )}
            </div>

            <button
              type="submit"
              className="w-full rounded-lg bg-indigo-600 px-4 py-2 text-xs font-semibold text-white shadow-sm hover:bg-indigo-700"
            >
              Apply Device Configuration
            </button>
          </form>
        </div>

        {/* Resource Budget Configuration */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
            <Activity className="h-5 w-5 text-amber-600" />
            <h2 className="font-semibold text-slate-900">Resource Budget & Policy Constraints</h2>
          </div>
          <form onSubmit={handleResourceBudgetSave} className="mt-4 space-y-4">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-medium text-slate-600">Latency Budget (ms)</label>
                <input
                  type="number"
                  step="0.5"
                  value={resourceBudget.latency_budget_ms}
                  onChange={(e) => setResourceBudget({ ...resourceBudget, latency_budget_ms: parseFloat(e.target.value) })}
                  className="mt-1 block w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm"
                />
              </div>
              <div>
                <label className="text-xs font-medium text-slate-600">CPU Throttle Limit (%)</label>
                <input
                  type="number"
                  step="1"
                  value={resourceBudget.cpu_throttle_threshold}
                  onChange={(e) => setResourceBudget({ ...resourceBudget, cpu_throttle_threshold: parseFloat(e.target.value) })}
                  className="mt-1 block w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-medium text-slate-600">Model A Confidence (τ_A)</label>
                <input
                  type="number"
                  step="0.01"
                  min="0.1"
                  max="1.0"
                  value={resourceBudget.threshold_a}
                  onChange={(e) => setResourceBudget({ ...resourceBudget, threshold_a: parseFloat(e.target.value) })}
                  className="mt-1 block w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm"
                />
              </div>
              <div>
                <label className="text-xs font-medium text-slate-600">Probe Confidence (τ_probe)</label>
                <input
                  type="number"
                  step="0.01"
                  min="0.1"
                  max="1.0"
                  value={resourceBudget.threshold_probe}
                  onChange={(e) => setResourceBudget({ ...resourceBudget, threshold_probe: parseFloat(e.target.value) })}
                  className="mt-1 block w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm"
                />
              </div>
            </div>

            <div className="flex items-center justify-between rounded-lg border border-slate-100 bg-slate-50 p-3">
              <div>
                <p className="text-xs font-semibold text-slate-800">Strict Resource Budget Enforcement</p>
                <p className="text-xs text-slate-500">Reject full Model B fallback when latency budget is exhausted.</p>
              </div>
              <input
                type="checkbox"
                checked={resourceBudget.enforce_resource_budget}
                onChange={(e) => setResourceBudget({ ...resourceBudget, enforce_resource_budget: e.target.checked })}
                className="h-4 w-4 rounded border-slate-300 text-amber-600 focus:ring-amber-500"
              />
            </div>

            <button
              type="submit"
              className="w-full rounded-lg bg-amber-600 px-4 py-2 text-xs font-semibold text-white shadow-sm hover:bg-amber-700"
            >
              Save Resource Budgets
            </button>
          </form>
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        {/* Default Training Parameters */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
            <Sliders className="h-5 w-5 text-blue-600" />
            <h2 className="font-semibold text-slate-900">Default Training Parameters</h2>
          </div>
          <form onSubmit={handleTrainingDefaultsSave} className="mt-4 space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-medium text-slate-600">Default Batch Size</label>
                <input
                  type="number"
                  value={trainingDefaults.batch_size}
                  onChange={(e) => setTrainingDefaults({ ...trainingDefaults, batch_size: parseInt(e.target.value) })}
                  className="mt-1 block w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm"
                />
              </div>
              <div>
                <label className="text-xs font-medium text-slate-600">Default Epochs</label>
                <input
                  type="number"
                  value={trainingDefaults.epochs}
                  onChange={(e) => setTrainingDefaults({ ...trainingDefaults, epochs: parseInt(e.target.value) })}
                  className="mt-1 block w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm"
                />
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-medium text-slate-600">Learning Rate</label>
                <input
                  type="number"
                  step="0.0001"
                  value={trainingDefaults.learning_rate}
                  onChange={(e) => setTrainingDefaults({ ...trainingDefaults, learning_rate: parseFloat(e.target.value) })}
                  className="mt-1 block w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm"
                />
              </div>
              <div>
                <label className="text-xs font-medium text-slate-600">Default Optimizer</label>
                <select
                  value={trainingDefaults.optimizer}
                  onChange={(e) => setTrainingDefaults({ ...trainingDefaults, optimizer: e.target.value })}
                  className="mt-1 block w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm"
                >
                  <option value="Adam">Adam</option>
                  <option value="AdamW">AdamW</option>
                  <option value="SGD">SGD</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-medium text-slate-600">Image Size (px)</label>
                <input
                  type="number"
                  value={trainingDefaults.img_size}
                  onChange={(e) => setTrainingDefaults({ ...trainingDefaults, img_size: parseInt(e.target.value) })}
                  className="mt-1 block w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm"
                />
              </div>
              <div>
                <label className="text-xs font-medium text-slate-600">Validation Split</label>
                <input
                  type="number"
                  step="0.05"
                  min="0.05"
                  max="0.5"
                  value={trainingDefaults.val_split}
                  onChange={(e) => setTrainingDefaults({ ...trainingDefaults, val_split: parseFloat(e.target.value) })}
                  className="mt-1 block w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm"
                />
              </div>
            </div>

            <button
              type="submit"
              className="mt-2 w-full rounded-lg bg-blue-600 px-4 py-2 text-xs font-semibold text-white shadow-sm hover:bg-blue-700"
            >
              Save Training Defaults
            </button>
          </form>
        </div>

        {/* Backend & Session Memory Controls */}
        <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
          <div className="flex items-center gap-2 border-b border-slate-100 pb-3">
            <Server className="h-5 w-5 text-emerald-600" />
            <h2 className="font-semibold text-slate-900">Backend & In-RAM Session Lifecycle</h2>
          </div>
          <form onSubmit={handleBackendSave} className="mt-4 space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs font-medium text-slate-600">Telemetry Poll Interval (ms)</label>
                <input
                  type="number"
                  step="100"
                  value={backendConfig.telemetry_poll_interval_ms}
                  onChange={(e) => setBackendConfig({ ...backendConfig, telemetry_poll_interval_ms: parseInt(e.target.value) })}
                  className="mt-1 block w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm"
                />
              </div>
              <div>
                <label className="text-xs font-medium text-slate-600">Cache Retention Limit</label>
                <input
                  type="number"
                  value={backendConfig.cache_retention_limit}
                  onChange={(e) => setBackendConfig({ ...backendConfig, cache_retention_limit: parseInt(e.target.value) })}
                  className="mt-1 block w-full rounded-md border border-slate-300 px-3 py-1.5 text-sm"
                />
              </div>
            </div>

            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <span className="text-xs font-semibold text-slate-700">In-RAM Candidate Model Policy:</span>
              <p className="mt-1 text-xs text-slate-500">
                Candidate weights, epoch logs, and metrics reside purely in backend RAM. Restarting or calling session cleanup immediately flushes candidate models. Zero disk files are created.
              </p>
              <div className="mt-2 flex gap-2">
                <span className="rounded bg-indigo-100 px-2 py-0.5 font-mono text-[11px] font-semibold text-indigo-700">
                  {sessionStorage.candidate_models_in_ram ?? 0} Models Active
                </span>
                <span className="rounded bg-emerald-100 px-2 py-0.5 text-[11px] font-semibold text-emerald-700">
                  RAM Mode Active
                </span>
              </div>
            </div>

            <button
              type="submit"
              className="mt-2 w-full rounded-lg bg-emerald-600 px-4 py-2 text-xs font-semibold text-white shadow-sm hover:bg-emerald-700"
            >
              Save Backend Settings
            </button>
          </form>
        </div>
      </div>

      {/* Protected Checkpoint Integrity Verification (Cryptographic SHA-256 Audit) */}
      <section className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2">
            <Lock className="h-5 w-5 text-indigo-600" />
            <div>
              <h2 className="font-semibold text-slate-900">Protected Research Checkpoint Integrity Verification</h2>
              <p className="text-xs text-slate-500">
                Audits outputs/ directory against canonical cryptographic SHA-256 hashes to guarantee zero file tampering or candidate contamination.
              </p>
            </div>
          </div>
          <button
            onClick={handleRunIntegrityAudit}
            disabled={verifying}
            className="inline-flex items-center gap-1.5 rounded-lg bg-indigo-600 px-3 py-1.5 text-xs font-semibold text-white shadow-sm hover:bg-indigo-700"
          >
            <ShieldCheck className={`h-4 w-4 ${verifying ? 'animate-spin' : ''}`} />
            {verifying ? 'Auditing...' : 'Run Cryptographic SHA-256 Audit'}
          </button>
        </div>

        {integrityData && (
          <div className="mt-4 space-y-4">
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
              {Object.entries(integrityData.checkpoints || {}).map(([k, item]) => {
                const isVerified = item.status === 'VERIFIED';
                const isMismatch = item.status === 'MISMATCH';
                return (
                  <div key={k} className="rounded-lg border border-slate-200 p-3.5">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold text-slate-800">{item.name}</span>
                      <span className={`rounded-full px-2 py-0.5 text-[10px] font-semibold ${
                        isVerified ? 'bg-emerald-100 text-emerald-800' :
                        isMismatch ? 'bg-amber-100 text-amber-800' : 'bg-slate-100 text-slate-600'
                      }`}>
                        {item.status}
                      </span>
                    </div>
                    <p className="mt-1 text-[11px] text-slate-500">{item.role}</p>
                    <div className="mt-2 space-y-1 text-[10px] font-mono text-slate-600">
                      <div>File: <span className="text-slate-900">{item.rel_path}</span></div>
                      <div>Expected: <span className="text-slate-700">{item.expected_sha256.slice(0, 16)}...</span></div>
                      <div>Actual: <span className="text-slate-700">{item.actual_sha256 ? item.actual_sha256.slice(0, 16) + '...' : 'Not on disk (Initialized in-memory)'}</span></div>
                    </div>
                  </div>
                );
              })}
            </div>

            <div className="rounded-lg border border-emerald-100 bg-emerald-50/50 p-3 text-xs text-emerald-900">
              <span className="font-semibold">Research Data Isolation Audit:</span>{' '}
              {integrityData.research_data_isolation?.clean_isolation ? (
                <span>100% Clean. No rogue candidate files exist in outputs/. Checkpoint isolation verified.</span>
              ) : (
                <span className="text-amber-800">Notice: {integrityData.research_data_isolation?.rogue_candidate_checkpoints?.length} non-standard files detected.</span>
              )}
            </div>
          </div>
        )}
      </section>
    </div>
  );
}
