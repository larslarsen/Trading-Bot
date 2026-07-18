"""Tests for data_quality.py -- reconciliation, spike detection, repair.

Uses SYNTHETIC DataFrames (no 1.3M-row files) so the suite is fast and
deterministic. Real-file integration is covered by the ad-hoc verification.
"""
import numpy as np
import pandas as pd
import pytest

import data_quality as dq


def _df(closes, start="2024-01-01", freq="5min"):
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz="UTC")
    return pd.DataFrame({"ts": idx, "close": closes})


# ── detect_spikes ────────────────────────────────────────────────────────
def test_detect_spikes_flags_over_threshold():
    df = _df([100, 105, 110, 50, 55])          # 55% drop at idx 3
    sp = dq.detect_spikes(df, max_pct=0.30)
    assert len(sp) == 1
    assert sp[0]["direction"] == "down"
    assert sp[0]["pct"] == pytest.approx(0.5454, rel=1e-3)


def test_detect_spikes_ignores_normal_moves():
    df = _df([100, 101, 99, 102, 100])
    assert dq.detect_spikes(df, max_pct=0.30) == []


def test_detect_spikes_handles_short_df():
    assert dq.detect_spikes(_df([100])) == []
    assert dq.detect_spikes(None) == []


# ── reconcile_venues (synthetic, aligned timestamps) ──────────────────────
def test_reconcile_flags_divergent_venue():
    idx = pd.date_range("2024-01-01", periods=4, freq="5min", tz="UTC")
    base = [100.0, 101, 102, 103]
    # venue A agrees with B/C except last bar spiked to 200
    a = pd.DataFrame({"ts": idx, "close": [100, 101, 102, 200.0]})
    b = pd.DataFrame({"ts": idx, "close": base})
    c = pd.DataFrame({"ts": idx, "close": base})
    import tempfile, os
    from pathlib import Path
    d = tempfile.mkdtemp()
    for name, fr in [("bybit", a), ("okx", b), ("blofin", c)]:
        fr.to_csv(Path(d) / f"TESTUSDT_5m_{name}_max.csv", index=False)
    try:
        # point the module's DATA dir at our temp dir
        old = dq.DATA
        dq.DATA = Path(d)
        rec = dq.reconcile_venues("TESTUSDT")
        assert rec["n_venues"] == 3
        bybit = [f for f in rec["flagged"] if f["venue"] == "bybit"]
        assert len(bybit) == 1
        assert bybit[0]["likely_corrupt_venue"] == "bybit"
    finally:
        dq.DATA = old
        for f in Path(d).glob("*"):
            f.unlink()
        os.rmdir(d)


def test_reconcile_skips_single_venue():
    import tempfile, os
    from pathlib import Path
    d = tempfile.mkdtemp()
    idx = pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC")
    pd.DataFrame({"ts": idx, "close": [1, 2, 3]}).to_csv(
        Path(d) / "ONLYUSDT_5m_bybit_max.csv", index=False)
    try:
        old = dq.DATA
        dq.DATA = Path(d)
        rec = dq.reconcile_venues("ONLYUSDT")
        assert rec["n_venues"] == 1
        assert rec["flagged"] == []   # cannot form a consensus
    finally:
        dq.DATA = old
        for f in Path(d).glob("*"):
            f.unlink()
        os.rmdir(d)


# ── cross_check_consensus repair logic ────────────────────────────────────
def test_cross_check_repairs_from_consensus():
    import tempfile, os
    from pathlib import Path
    d = tempfile.mkdtemp()
    idx = pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC")
    base = [100.0, 101, 102, 103, 104]
    # consolidated corrupted at idx 2 (60 instead of 102)
    cons = pd.DataFrame({"ts": idx, "close": [100, 101, 60, 103, 104]})
    cons.to_csv(Path(d) / "REPUSDT_5m_max.csv", index=False)
    for name, mult in [("bybit", 1.0), ("okx", 1.0), ("blofin", 1.0)]:
        pd.DataFrame({"ts": idx, "close": [x * mult for x in base]}).to_csv(
            Path(d) / f"REPUSDT_5m_{name}_max.csv", index=False)
    try:
        old = dq.DATA
        dq.DATA = Path(d)
        cc = dq.cross_check_consensus("REPUSDT", repair=True)
        assert cc["repaired"] == 1
        fixed = pd.read_csv(Path(d) / "REPUSDT_5m_max.csv", parse_dates=["ts"])
        assert abs(fixed["close"].iloc[2] - 102.0) < 1e-6
    finally:
        dq.DATA = old
        for f in Path(d).glob("*"):
            f.unlink()
        os.rmdir(d)


def test_cross_check_no_repair_when_clean():
    import tempfile, os
    from pathlib import Path
    d = tempfile.mkdtemp()
    idx = pd.date_range("2024-01-01", periods=5, freq="5min", tz="UTC")
    base = [100.0, 101, 102, 103, 104]
    pd.DataFrame({"ts": idx, "close": base}).to_csv(
        Path(d) / "CLEANUSDT_5m_max.csv", index=False)
    for name in ("bybit", "okx", "blofin"):
        pd.DataFrame({"ts": idx, "close": base}).to_csv(
            Path(d) / f"CLEANUSDT_5m_{name}_max.csv", index=False)
    try:
        old = dq.DATA
        dq.DATA = Path(d)
        cc = dq.cross_check_consensus("CLEANUSDT", repair=True)
        assert cc["repaired"] == 0
        assert cc["flagged"] == []
    finally:
        dq.DATA = old
        for f in Path(d).glob("*"):
            f.unlink()
        os.rmdir(d)


# ── gmgn best-effort: must never raise, returns None on failure ───────────
def test_gmgn_returns_none_on_bad_input(monkeypatch):
    # force requests.get to raise -> must return None, not throw
    import requests
    def boom(*a, **k):
        raise requests.RequestException("blocked")
    monkeypatch.setattr(requests, "get", boom)
    assert dq.gmgn_klines("deadbeef") is None
