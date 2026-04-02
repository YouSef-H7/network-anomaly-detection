import requests
import json

BASE_URL = "https://network-anomaly-api-894733684211.us-central1.run.app"

def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] == True
    print("PASS: Health check")

def test_analyze_normal():
    payload = {
        "server_id": "TEST-001",
        "cpu_usage": 30,
        "memory_usage": 45,
        "disk_io": 20,
        "network_traffic": 150,
        "error_count": 1,
        "response_time": 100,
        "active_connections": 40,
        "packet_loss": 0.1
    }
    r = requests.post(f"{BASE_URL}/analyze", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["risk_level"] == "HEALTHY"
    assert data["is_anomaly"] == False
    print("PASS: Normal server detected as healthy")

def test_analyze_anomaly():
    payload = {
        "server_id": "TEST-002",
        "cpu_usage": 98,
        "memory_usage": 95,
        "disk_io": 90,
        "network_traffic": 1500,
        "error_count": 50,
        "response_time": 3000,
        "active_connections": 400,
        "packet_loss": 25
    }
    r = requests.post(f"{BASE_URL}/analyze", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["is_anomaly"] == True
    assert data["risk_level"] in ["HIGH", "CRITICAL"]
    print("PASS: Anomaly server detected correctly")

def test_model_info():
    r = requests.get(f"{BASE_URL}/model/info")
    assert r.status_code == 200
    data = r.json()
    assert data["model"] == "Isolation Forest"
    assert data["f1_score"] > 0.9
    print("PASS: Model info correct")

def test_dashboard():
    r = requests.get(f"{BASE_URL}/dashboard")
    assert r.status_code == 200
    assert "NetGuard" in r.text
    print("PASS: Dashboard loads correctly")

def test_report():
    r = requests.get(f"{BASE_URL}/report/generate")
    assert r.status_code == 200
    data = r.json()
    assert "summary" in data
    assert "top_problematic_servers" in data
    print("PASS: Report generates correctly")

if __name__ == "__main__":
    print("=" * 50)
    print("  Running Automated Tests")
    print("=" * 50)

    tests = [test_health, test_analyze_normal, test_analyze_anomaly, test_model_info, test_dashboard, test_report]
    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__} — {e}")
            failed += 1

    print(f"\nResults: {passed} passed, {failed} failed out of {len(tests)}")
    print("=" * 50)
