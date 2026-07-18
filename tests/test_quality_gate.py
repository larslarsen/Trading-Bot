"""Tests for quality_gate.py — CEX/DEX universe gating + persistence.

Uses synthetic CSVs in a temp dir (no real data dependency).
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import quality_gate as qg


def _write_5m(path, n_rows, start_ts, close=100.0, vol=10_000.0):
    # 5m spacing; enough recent bars to clear the 30d recent-volume gate.
    idx = pd.date_range(start_ts, periods=n_rows, freq="5min", tz="UTC")
    df = pd.DataFrame({
        "ts": idx.astype(str),
        "close": close,
        "volume": vol,
    })
    df.to_csv(path, index=False)


def _make_cex_dir(tmp_path, good=True):
    d = tmp_path / "data"
    d.mkdir()
    # a good pair: 3y history, high volume, alive
    if good:
        # ~3.3y of 5m bars (>730d history gate; 3y = 315,288 bars)
        _write_5m(d / "GOODUSDT_5m_max.csv", n_rows=350_000,
                   start_ts="2021-01-01", close=100.0, vol=10_000.0)
    # too few bars
    _write_5m(d / "FEWUSDT_5m_max.csv", n_rows=100,
              start_ts="2023-01-01", close=100.0, vol=10_000.0)
    # thin volume
    _write_5m(d / "THINUSDT_5m_max.csv", n_rows=200_000,
              start_ts="2021-01-01", close=100.0, vol=10.0)
    # stablecoin (excluded by name)
    _write_5m(d / "USDCUSDT_5m_max.csv", n_rows=200_000,
              start_ts="2021-01-01", close=1.0, vol=10_000.0)
    # venue-suffixed duplicate (must be skipped to avoid double count)
    _write_5m(d / "GOOD-USDT_5m_max.csv", n_rows=200_000,
              start_ts="2021-01-01", close=100.0, vol=10_000.0)
    return d


def test_gated_universe_passes_good_and_rejects_bad(monkeypatch, tmp_path):
    d = _make_cex_dir(tmp_path)
    monkeypatch.setattr(qg, "DATA", d)
    passed = qg.gated_universe(data_dir=d)
    assert "GOODUSDT" in passed
    assert "FEWUSDT" not in passed          # too few bars
    assert "THINUSDT" not in passed         # thin volume
    assert "USDCUSDT" not in passed         # stable
    # venue-suffixed duplicate not double-counted
    assert sum("GOOD" in p for p in passed) == 1


def test_gated_universe_min_bars_threshold(monkeypatch, tmp_path):
    d = _make_cex_dir(tmp_path)
    passed = qg.gated_universe(data_dir=d, min_bars=400_000)
    assert "GOODUSDT" not in passed         # now exceeds the 350k good pair


def test_gated_universe_min_history_threshold(monkeypatch, tmp_path):
    d = _make_cex_dir(tmp_path)
    # good pair has ~3y; require 10y -> should fail the history gate
    passed = qg.gated_universe(data_dir=d, min_history_days=3650)
    assert "GOODUSDT" not in passed


def test_gated_universe_exclude_set(monkeypatch, tmp_path):
    d = _make_cex_dir(tmp_path)
    passed = qg.gated_universe(data_dir=d, exclude={"GOODUSDT"})
    assert "GOODUSDT" not in passed


def test_gated_universe_handles_unreadable_file(monkeypatch, tmp_path):
    d = _make_cex_dir(tmp_path)
    # a corrupt CSV that read_csv will choke on
    (d / "BADUSDT_5m_max.csv").write_text("not,a,valid\n1,2,3\n")
    passed = qg.gated_universe(data_dir=d)
    assert "BADUSDT" not in passed           # skipped, not crashed


def test_save_gate_writes_json(tmp_path):
    p = tmp_path / "universe_gated.json"
    out = qg.save_gate(["A", "B"], path=p)
    meta = json.loads(out.read_text())
    assert meta["n_pairs"] == 2 and meta["pairs"] == ["A", "B"]
    assert meta["venue"] == "cex"


def test_save_dex_gate_writes_json(tmp_path):
    p = tmp_path / "universe_dex_gated.json"
    out = qg.save_dex_gate(["X", "Y"], path=p)
    meta = json.loads(out.read_text())
    assert meta["venue"] == "dex" and meta["pairs"] == ["X", "Y"]


def test_gated_universe_verbose_does_not_crash(monkeypatch, tmp_path):
    d = _make_cex_dir(tmp_path)
    # verbose path just prints; must not raise
    passed = qg.gated_universe(data_dir=d, verbose=True)
    assert isinstance(passed, list)


# ── DEX gate ────────────────────────────────────────────────────────────────
def _make_dex_dir(tmp_path):
    d = tmp_path / "data" / "dex"
    d.mkdir(parents=True)
    # a token with enough 1m bars passes the min-bars gate
    idx = pd.date_range("2024-01-01", periods=200, freq="1min", tz="UTC")
    pd.DataFrame({"ts": idx.astype(str), "volume": 1.0}).to_csv(
        d / "TOKENA_1m_max.csv", index=False)
    pd.DataFrame({"ts": idx.astype(str), "volume": 1.0}).to_csv(
        d / "TOKENB_1m_max.csv", index=False)
    return d


def test_dex_gated_universe_fallback_when_too_few(monkeypatch, tmp_path):
    d = _make_dex_dir(tmp_path)
    # gated set (2 tokens) is smaller than min_candidates -> fallback fills in
    # from the live liquidity ranking (imported from dex_ohlcv_sampler),
    # appending up to fallback_top beyond the gated passes.
    passed = qg.dex_gated_universe(dex_dir=d, min_candidates=10, fallback_top=5)
    assert isinstance(passed, list)
    # the 2 bar-passing tokens are always kept, plus <=5 fallback entries
    assert "TOKENA" in passed and "TOKENB" in passed
    assert len(passed) <= 2 + 5


def test_dex_gated_universe_passes_bars(monkeypatch, tmp_path):
    d = _make_dex_dir(tmp_path)
    passed = qg.dex_gated_universe(dex_dir=d, min_candidates=1, fallback_top=20)
    # tokens with enough bars are selected (liquidity floor unenforced w/o rank)
    assert "TOKENA" in passed and "TOKENB" in passed


def test_dex_gated_universe_handles_bad_file(monkeypatch, tmp_path):
    d = _make_dex_dir(tmp_path)
    (d / "BAD_1m_max.csv").write_text("not,valid\n1,2\n")
    passed = qg.dex_gated_universe(dex_dir=d, min_candidates=1, fallback_top=20)
    assert "BAD" not in passed
