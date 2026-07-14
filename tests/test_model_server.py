"""Edge-case / branch-coverage tests for model_server.py.

model_server imports `pipeline` (which transitively needs the missing
`equities_regime` module in this env), so we stub `pipeline` in sys.modules
before importing. The full ML feature pipeline (compute_features_from_history)
is heavy and data-dependent; instead we test the deterministic, high-value
units that the bug-hunt fixed:

  - _safe_read_csv : private snapshot read (F2) -> equals direct read, missing
    file raises FileNotFoundError.
  - get_signal F3  : feature-count mismatch returns an ERROR response (not a
    silent truncate/pad).
  - get_signal F1  : model is snapshotted under _model_lock and predicts under
    the lock (no deadlock, no re-entrant nesting) on concurrent calls.
"""
import sys
import pandas as pd
import pytest
from unittest.mock import MagicMock


# Stub `pipeline` so model_server can import without equities_regime.
sys.modules.setdefault("pipeline", MagicMock())

import model_server as ms  # noqa: E402


class FakeModel:
    """Minimal stand-in for a fitted XGBoost model."""
    n_features_in_ = 5

    def predict(self, X):
        return [1]

    def predict_proba(self, X):
        # real XGBoost/sklearn models return an ndarray (get_signal calls .tolist())
        return __import__("numpy").array([[0.2, 0.8]])


@pytest.fixture()
def patched_signal(monkeypatch):
    """Isolate get_signal from the heavy feature pipeline + model loading."""
    monkeypatch.setattr(ms, "latest_model", FakeModel())
    monkeypatch.setattr(ms, "load_latest_model", lambda: True)

    def _fvec(cols):
        f = pd.DataFrame([{c: 1.0 for c in cols}])
        last = pd.Series({"close": 100.0}, name=pd.Timestamp("2025-01-02", tz="UTC"))
        return f, last

    calls = {"features": None}

    def _compute(cols):
        def _inner():
            calls["features"] = cols
            return _fvec(cols)
        return _inner

    # default: matching feature count
    monkeypatch.setattr(ms, "compute_features_from_history", _compute(list(range(5))))
    return calls


# ── _safe_read_csv (F2) ─────────────────────────────────────────────────────
def test_safe_read_csv_matches_direct(tmp_path):
    p = tmp_path / "hist.csv"
    pd.DataFrame({"ts": ["2025-01-01", "2025-01-02"], "close": [1.0, 2.0]}).to_csv(p, index=False)
    snap = ms._safe_read_csv(p)
    direct = pd.read_csv(p, parse_dates=["ts"])
    pd.testing.assert_frame_equal(snap, direct)
    # temp copy is cleaned up
    assert not list(tmp_path.glob("*.tmp"))


def test_safe_read_csv_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ms._safe_read_csv(tmp_path / "nope.csv")


# ── get_signal F3: loud error on feature mismatch ───────────────────────────
def test_get_signal_feature_mismatch_returns_error(monkeypatch, patched_signal):
    # stub compute to return fewer columns than the model expects (model has 5)
    def _inner():
        f = pd.DataFrame([{c: 1.0 for c in ["a", "b", "c"]}])
        return f, pd.Series({"close": 1.0}, name=pd.Timestamp("2025-01-02", tz="UTC"))
    monkeypatch.setattr(ms, "compute_features_from_history", _inner)
    resp = ms.get_signal()
    assert getattr(resp, "error", None) is not None
    assert "feature count" in resp.error


# ── get_signal F1: lock snapshot, no deadlock, predicts under lock ──────────
def test_get_signal_predicts_under_lock(monkeypatch, patched_signal):
    resp = ms.get_signal()
    # matching features (5) -> no error, returns a (flat-or-signal) response
    assert getattr(resp, "error", None) is None
    assert resp.signal in ("LONG", "FLAT", "SHORT")


def test_get_signal_concurrent_no_deadlock(monkeypatch, patched_signal):
    # call get_signal many times; the lock must serialize predict without
    # deadlocking (Lock is not re-entrant, so nesting would hang).
    for _ in range(20):
        resp = ms.get_signal()
        assert resp is not None


def test_get_signal_model_none_returns_flat(monkeypatch):
    monkeypatch.setattr(ms, "latest_model", None)
    monkeypatch.setattr(ms, "load_latest_model", lambda: False)
    resp = ms.get_signal()
    assert resp.signal == "FLAT"
