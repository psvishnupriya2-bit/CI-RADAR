import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app
from src.models.baseline import time_split
from src.monitor.drift import psi


def test_time_split_sizes_and_order():
    df = pd.DataFrame({
        "started_at": pd.date_range("2026-01-01", periods=100, freq="h"),
        "failed": [0, 1] * 50,
    })
    tr, va, te = time_split(df)
    assert (len(tr), len(va), len(te)) == (70, 15, 15)
    assert tr["started_at"].max() <= va["started_at"].min()
    assert va["started_at"].max() <= te["started_at"].min()


def test_psi_same_data_is_small():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 2000))
    b = pd.Series(rng.normal(0, 1, 2000))
    assert psi(a, b) < 0.1


def test_psi_shifted_data_is_large():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 2000))
    b = pd.Series(rng.normal(3, 1, 2000))
    assert psi(a, b) > 0.2


def test_health_endpoint():
    client = TestClient(app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict_returns_valid_risk():
    client = TestClient(app)
    r = client.post("/predict", json={})
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["risk"] <= 1.0
    assert body["label"] in ("low", "medium", "high")
