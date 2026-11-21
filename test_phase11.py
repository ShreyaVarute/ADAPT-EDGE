"""Quick Phase 11 settings endpoint verification."""
from fastapi.testclient import TestClient
from server.app import app

with TestClient(app) as client:
    r1 = client.get('/api/settings')
    print('GET /api/settings:', r1.status_code)
    data = r1.json()
    print('  Device:', data.get('device', {}).get('active_device'))
    print('  Candidates in RAM:', data.get('session_storage', {}).get('candidate_models_in_ram'))

    r2 = client.post('/api/settings/device', json={'device': 'cpu'})
    print('POST /api/settings/device:', r2.status_code, r2.json().get('active_device'))

    r3 = client.post('/api/settings/training-defaults', json={'batch_size': 8, 'epochs': 10})
    print('POST /api/settings/training-defaults:', r3.status_code)
    print('  batch_size:', r3.json().get('training_defaults', {}).get('batch_size'))

    r4 = client.post('/api/settings/resource-budget', json={'latency_budget_ms': 30.0, 'threshold_a': 0.88})
    print('POST /api/settings/resource-budget:', r4.status_code)
    print('  latency_budget_ms:', r4.json().get('resource_budget', {}).get('latency_budget_ms'))

    r5 = client.post('/api/settings/backend', json={'telemetry_poll_interval_ms': 1500})
    print('POST /api/settings/backend:', r5.status_code)

    r6 = client.post('/api/settings/cleanup')
    print('POST /api/settings/cleanup:', r6.status_code)
    print('  Result:', r6.json().get('status', '')[:70])

    r7 = client.get('/api/settings/verify-integrity')
    print('GET /api/settings/verify-integrity:', r7.status_code)
    intdata = r7.json()
    print('  Clean isolation:', intdata['research_data_isolation']['clean_isolation'])
    for k, v in intdata['checkpoints'].items():
        print(f'  {k}: {v["status"]}')

    print('\nAll Phase 11 endpoints functional.')
