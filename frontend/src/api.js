/**
 * ADAPT-EDGE Frontend API Client
 * Centralized fetch helpers with error handling
 */

const BASE_URL = '/api';

export async function fetchJson(endpoint, options = {}) {
  const url = `${BASE_URL}${endpoint}`;
  try {
    const headers = options.body instanceof FormData ? {} : {
      'Content-Type': 'application/json',
      ...options.headers,
    };
    const res = await fetch(url, {
      headers,
      ...options,
    });
    if (!res.ok) {
      const errData = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(errData.detail || `Request failed with status ${res.status}`);
    }
    return await res.json();
  } catch (err) {
    console.error(`API Error [${endpoint}]:`, err);
    throw err;
  }
}

// API methods
export const api = {
  // System & Hardware
  getOverview: () => fetchJson('/system/overview'),
  getHardware: () => fetchJson('/system/hardware'),
  clearCache: () => fetchJson('/system/clear-cache', { method: 'POST' }),
  getHealth: () => fetchJson('/health'),
  getResources: () => fetchJson('/resources/status'),
  getModelsStatus: () => fetchJson('/models/status'),
  clearUiCache: () => fetchJson('/cache/clear?type=ui', { method: 'POST' }),
  clearContinuationCache: () => fetchJson('/cache/clear?type=continuation', { method: 'POST' }),

  // Inference
  getSamples: (count = 12) => fetchJson(`/inference/samples?count=${count}`),
  refreshSamples: (count = 12) => fetchJson(`/inference/samples/refresh?count=${count}`, { method: 'POST' }),
  getCustomModels: () => fetchJson('/training/custom-models'),
  predict: (payload) => fetchJson('/inference/predict', {
    method: 'POST',
    body: JSON.stringify(payload)
  }),
  uploadAndPredict: (formData, options = {}) => {
    const params = new URLSearchParams();
    params.set('model_mode', options.modelMode || 'cifar10');
    if (options.customRunId) params.set('custom_run_id', options.customRunId);
    if (options.strategy) params.set('strategy', options.strategy);
    return fetchJson(`/inference/upload?${params.toString()}`, { method: 'POST', body: formData });
  },
  compareInference: (payload) => fetchJson('/inference/compare', { method: 'POST', body: JSON.stringify(payload) }),
  getInferenceHistory: () => fetchJson('/inference/history'),

  // Datasets
  getDatasets: () => fetchJson('/datasets'),
  getCifarStats: () => fetchJson('/datasets/cifar10/stats'),
  getCifarSamples: (page = 0, limit = 20, classIdx = null) => {
    let q = `/datasets/cifar10/samples?page=${page}&limit=${limit}`;
    if (classIdx !== null && classIdx !== undefined) q += `&class_idx=${classIdx}`;
    return fetchJson(q);
  },
  uploadDataset: (formData) => fetchJson('/datasets/upload', { method: 'POST', body: formData }),
  inspectDataset: (id) => fetchJson(`/datasets/inspect/${encodeURIComponent(id)}`),

  getTrainingStatus: () => fetchJson('/training/status'),
  startTraining: (payload) => fetchJson('/training/start', { method: 'POST', body: JSON.stringify(payload) }),
  stopTraining: () => fetchJson('/training/stop', { method: 'POST' }),
  getTrainingRecommendations: () => fetchJson('/training/recommendations'),

  // Experiments
  getExperimentRuns: () => fetchJson('/experiments/runs'),
  getExperiments: () => fetchJson('/experiments/history'),
  getExperimentDetails: (runId) => fetchJson(`/experiments/${encodeURIComponent(runId)}?limit_traces=100`),
  exportExperiment: (runId, format = 'csv') => `/api/reports/download-benchmark-csv?run_name=${encodeURIComponent(runId)}`,
  getExperimentTraces: (runName, filterType = null, limit = 100) => {
    let q = `/experiments/traces?run_name=${encodeURIComponent(runName)}&limit=${limit}`;
    if (filterType) q += `&filter_decision=${encodeURIComponent(filterType)}`;
    return fetchJson(q);
  },
  runBenchmark: (numSamples = 10, useCuda = true) => fetchJson('/experiments/run', {
    method: 'POST',
    body: JSON.stringify({ num_samples: numSamples, use_cuda: useCuda })
  }),

  // Diagnostics
  runDiagnostics: () => fetchJson('/diagnostics/run', { method: 'POST' }),

  // Training Recommendations
  getTrainingRecommendation: () => fetchJson('/training/recommendation'),

  // Reports
  getResearchSummary: () => fetchJson('/reports/summary'),
  getMarkdownReport: async () => {
    const res = await fetch('/api/reports/markdown');
    return await res.text();
  }
};

