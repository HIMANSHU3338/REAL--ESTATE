#!/usr/bin/env python3
"""
OpenEnv Endpoint Test Script
Tests the FastAPI server endpoints to ensure OpenEnv compatibility.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi.testclient import TestClient
from server.app import app

def test_endpoints():
    """Test OpenEnv required endpoints."""
    client = TestClient(app)
    
    print("=" * 60)
    print("Testing OpenEnv Endpoints")
    print("=" * 60)
    
    # Test 1: GET /info
    print("\n[1/5] Testing GET /info...")
    resp = client.get("/info")
    print(f"Status: {resp.status_code}")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert "name" in data
    assert "tasks" in data
    print(f"✓ /info endpoint works: {data['name']}")
    
    # Test 2: POST /reset (empty body)
    print("\n[2/5] Testing POST /reset (empty body)...")
    resp = client.post("/reset", json={})
    print(f"Status: {resp.status_code}")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert "observation" in data
    assert "info" in data
    print(f"✓ /reset endpoint works: observation shape {len(data['observation'])}")
    
    # Test 3: POST /reset (with seed)
    print("\n[3/5] Testing POST /reset (with seed)...")
    resp = client.post("/reset", json={"seed": 42})
    print(f"Status: {resp.status_code}")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert "observation" in data
    print(f"✓ /reset with seed works")
    
    # Test 4: POST /step
    print("\n[4/5] Testing POST /step...")
    resp = client.post("/step", json={"action": [0, 0, 0, 0, 0]})
    print(f"Status: {resp.status_code}")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert "observation" in data
    assert "reward" in data
    assert "done" in data
    assert "terminated" in data
    assert "truncated" in data
    assert "info" in data
    print(f"✓ /step endpoint works: reward={data['reward']:.4f}, done={data['done']}")
    
    # Test 5: GET /state
    print("\n[5/5] Testing GET /state...")
    resp = client.get("/state")
    print(f"Status: {resp.status_code}")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    data = resp.json()
    assert "month" in data
    assert "cash" in data
    assert "net_worth" in data
    print(f"✓ /state endpoint works: month={data['month']}, cash=₹{data['cash']:,.0f}")
    
    print("\n" + "=" * 60)
    print("✓ All OpenEnv endpoints passed!")
    print("=" * 60)

if __name__ == "__main__":
    try:
        test_endpoints()
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
