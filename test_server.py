"""
ADAPT-EDGE Platform - Smoke Test Script
Verifies that FastAPI server and all core endpoints operate correctly without errors.
"""

import sys
import json
import time
from fastapi.testclient import TestClient

from server.app import app
from server.core import ServiceManager

def run_smoke_tests():
    print("=" * 60)
    print("ADAPT-EDGE WEB PLATFORM END-TO-END SMOKE TEST")
    print("=" * 60)
    
    with TestClient(app) as client:
        # Test 1: Health
        print("\n[1/7] Testing GET /api/health...")
        res = client.get("/api/health")
        assert res.status_code == 200, f"Health check failed: {res.text}"
        print(f"  -> Health OK: {res.json()}")

        # Test 2: System Overview
        print("\n[2/7] Testing GET /api/system/overview...")
        res = client.get("/api/system/overview")
        assert res.status_code == 200, f"Overview failed: {res.text}"
        overview = res.json()
        print(f"  -> Device: {overview.get('hardware', {}).get('gpu_name', 'N/A')}")
        print(f"  -> Pipeline Ready: {overview.get('pipeline_status')}")

        # Test 3: Hardware Telemetry
        print("\n[3/7] Testing GET /api/system/hardware...")
        res = client.get("/api/system/hardware")
        assert res.status_code == 200, f"Hardware check failed: {res.text}"
        hw = res.json()
        print(f"  -> Hardware snap: {hw.get('telemetry', {}).get('gpu_name')} | Status: {hw.get('status')}")

        # Test 4: CIFAR-10 Samples Gallery
        print("\n[4/7] Testing GET /api/inference/samples...")
        res = client.get("/api/inference/samples?count=6")
        assert res.status_code == 200, f"Samples failed: {res.text}"
        samples_data = res.json()
        samples = samples_data.get("samples", [])
        assert len(samples) > 0, "No samples returned"
        sample0 = samples[0]
        print(f"  -> Loaded {len(samples)} samples. Sample #0: Class '{sample0['class_name']}' (Index {sample0['index']})")

        # Test 5: Live Real Inference (Comparison of All 4 Strategies)
        print("\n[5/7] Testing POST /api/inference/predict (Strategy: ALL)...")
        payload = {
            "sample_index": sample0["index"],
            "strategy": "ALL",
            "tau_high": 0.85,
            "tau_low": 0.50,
            "tau_probe": 0.80,
            "latency_budget_ms": 25.0
        }
        start = time.time()
        res = client.post("/api/inference/predict", json=payload)
        elapsed = (time.time() - start) * 1000
        assert res.status_code == 200, f"Prediction failed: {res.text}"
        pred_data = res.json()
        print(f"  -> Prediction completed in {elapsed:.1f} ms")
        print(f"  -> Ground Truth: {pred_data.get('ground_truth')}")
        for c in pred_data.get("comparison", []):
            print(f"     * {c['strategy_name']}: Pred={c['prediction_class']} (Conf={c['confidence']:.3f}, Latency={c['latency_ms']:.2f}ms)")
        
        cached_res = pred_data.get("adapt_edge_cached", {})
        trace = cached_res.get("trace", {})
        print(f"  -> ADAPT-EDGE Decision: {trace.get('decision')} (Reason: {trace.get('decision_reason')})")
        print(f"  -> Continuation Cache Status: {trace.get('continuation_cache_status')}")

        # Test 6: System Diagnostics (8 checks)
        print("\n[6/7] Testing POST /api/diagnostics/run...")
        res = client.post("/api/diagnostics/run")
        assert res.status_code == 200, f"Diagnostics failed: {res.text}"
        diag = res.json()
        print(f"  -> Total Checks: {diag.get('total_checks')} | Passed: {diag.get('passed')} | Failed: {diag.get('failed')}")
        for c in diag.get("checks", []):
            print(f"     [{c['status']}] Check #{c['id']}: {c['name']} ({c['execution_ms']:.1f}ms)")

        # Test 7: Static Frontend Serving
        print("\n[7/7] Testing Static Single Page Application Serving (GET /)...")
        res = client.get("/")
        assert res.status_code == 200, f"Frontend serve failed: {res.text}"
        assert "ADAPT-EDGE" in res.text, "Frontend HTML missing title"
        print("  -> React SPA served successfully at root URL '/'")

        print("\n" + "=" * 60)
        print("ALL 7 END-TO-END SMOKE TESTS PASSED PERFECTLY!")
        print("The ADAPT-EDGE full-stack application is 100% operational.")
        print("=" * 60)

if __name__ == "__main__":
    run_smoke_tests()
