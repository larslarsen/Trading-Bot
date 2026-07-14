"""Edge-case / branch-coverage tests for collector_daemon.py.

Network/exchange calls are mocked (FakeEx) so tests are deterministic and
offline. Covers map_symbols (spot + derivative quoting), fetch_forward
(incremental append, empty response, dedup/sort, fresh file), RateGate
(sliding-window throttle) and build_jobs (screen-driven job list).
"""
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import collector_daemon as cd


class FakeEx:
    def __init__(self, markets):
        self.markets = markets

    def fetchOHLCV(self, sym, tf, since=None, limit=500):
        base = since if since is not None else 1_000_000_000_000
        return [
            [base, 100.0, 101.0, 99.0, 100.0, 10.0],
            [base + 60_000, 101.0, 102.0, 100.0, 101.0, 11.0],
        ]


# ── map_symbols ─────────────────────────────────────────────────────────────
def test_map_symbols_spot_style():
    ex = FakeEx({"BTC/USDT": {}, "ETH/USDT": {}})
    out = cd.map_symbols(ex, ["BTCUSDT", "ETHUSDT", "SOLUSDT"])
    assert out == {"BTCUSDT": "BTC/USDT", "ETHUSDT": "ETH/USDT"}
    assert "SOLUSDT" not in out  # not in markets


def test_map_symbols_derivative_style():
    # BloFin uses BASE/QUOTE:SETTLE; map_symbols must prefer the :SETTLE form
    ex = FakeEx({"BTC/USDT:USDT": {}, "ETH/USDT": {}})
    out = cd.map_symbols(ex, ["BTCUSDT"])
    assert out == {"BTCUSDT": "BTC/USDT:USDT"}


def test_map_symbols_usdc_quote():
    ex = FakeEx({"SOL/USDC": {}})
    out = cd.map_symbols(ex, ["SOLUSDC"])
    assert out == {"SOLUSDC": "SOL/USDC"}


# ── fetch_forward ───────────────────────────────────────────────────────────
def test_fetch_forward_creates_fresh_file(tmp_path):
    ex = FakeEx({"BTC/USDT": {}})
    f = tmp_path / "BTCUSDT_5m_mexc_max.csv"
    n = cd.fetch_forward(ex, "BTC/USDT", "5m", f)
    assert n == 2
    df = pd.read_csv(f)
    assert len(df) == 2
    assert list(df.columns)[:6] == ["ts", "open", "high", "low", "close", "volume"]


def test_fetch_forward_appends_incrementally(tmp_path):
    ex = FakeEx({"BTC/USDT": {}})
    f = tmp_path / "BTCUSDT_5m_mexc_max.csv"
    # seed an existing file with one bar at t=1000
    pd.DataFrame({
        "ts": [pd.to_datetime(1000, unit="ms", utc=True)],
        "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [10.0],
    }).to_csv(f, index=False)
    n = cd.fetch_forward(ex, "BTC/USDT", "5m", f)
    # fetch_forward asks since=last_ms+1 -> fake returns bars at base=1001 -> 2 new
    assert n == 2
    df = pd.read_csv(f)
    assert len(df) == 3  # 1 existing + 2 new, sorted, deduped


def test_fetch_forward_empty_response_no_change(tmp_path):
    class EmptyEx(FakeEx):
        def fetchOHLCV(self, sym, tf, since=None, limit=500):
            return []

    f = tmp_path / "BTCUSDT_5m_mexc_max.csv"
    pd.DataFrame({
        "ts": [pd.to_datetime(1000, unit="ms", utc=True)],
        "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [10.0],
    }).to_csv(f, index=False)
    n = cd.fetch_forward(EmptyEx({"BTC/USDT": {}}), "BTC/USDT", "5m", f)
    assert n == 0
    assert len(pd.read_csv(f)) == 1  # unchanged


def test_fetch_forward_dedupes_overlapping_bars(tmp_path):
    class OverlapEx(FakeEx):
        def fetchOHLCV(self, sym, tf, since=None, limit=500):
            # ignore `since` and return a bar whose ts equals the EXISTING one
            return [[1_000_000_000_000, 100.0, 101.0, 99.0, 100.0, 10.0]]

    f = tmp_path / "BTCUSDT_5m_mexc_max.csv"
    pd.DataFrame({
        "ts": [pd.to_datetime(1_000_000_000_000, unit="ms", utc=True)],
        "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [10.0],
    }).to_csv(f, index=False)
    n = cd.fetch_forward(OverlapEx({"BTC/USDT": {}}), "BTC/USDT", "5m", f)
    assert n == 0  # overlapping ts is deduped -> no new bars
    assert len(pd.read_csv(f)) == 1


# ── RateGate ────────────────────────────────────────────────────────────────
def test_rate_gate_throttles_when_window_full():
    gate = cd.RateGate(per_min=2)
    # tick() calls time.time() twice (start + append) -> 3 ticks need 6 values
    with patch.object(time, "time", side_effect=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), \
         patch.object(time, "sleep") as mock_sleep:
        gate.tick()  # t=0
        gate.tick()  # t=0 -> window now full (2)
        gate.tick()  # t=0 -> must sleep (>= per_min)
        mock_sleep.assert_called_once()
        assert len(gate.times) == 3


def test_rate_gate_no_sleep_under_limit():
    gate = cd.RateGate(per_min=5)
    with patch.object(time, "time", side_effect=[0.0, 0.0, 1.0, 1.0, 2.0, 2.0]), \
         patch.object(time, "sleep") as mock_sleep:
        gate.tick(); gate.tick(); gate.tick()
        mock_sleep.assert_not_called()


# ── build_jobs ──────────────────────────────────────────────────────────────
def test_build_jobs_from_screen(monkeypatch, tmp_path):
    # fake exchange with both spot and derivative forms
    fake_ex = FakeEx({"BTC/USDT": {}, "ETH/USDT:USDT": {}})
    monkeypatch.setattr(cd, "get_ex", lambda name: fake_ex)
    monkeypatch.setattr(cd, "ENABLE", {"mexc": True, "blofin": False, "kraken": False})

    screen_dir = tmp_path / "backtest_output"
    screen_dir.mkdir()
    screen = pd.DataFrame([{"stem": "BTCUSDT"}, {"stem": "ETHUSDT"}])
    screen.to_csv(screen_dir / "screen_liqu_idio_20250101_000000.csv", index=False)
    monkeypatch.setattr(cd, "SCREEN_DIR", screen_dir)

    jobs = cd.build_jobs()
    # 2 mapped symbols * len(TIMEFRAMES)=4 tf = 8 jobs
    assert len(jobs) == 8
    assert all(j[0] == "mexc" for j in jobs)
    assert all(len(j) == 4 for j in jobs)  # (ex, sym, ccxt_sym, tf)


def test_build_jobs_empty_screen_no_jobs(monkeypatch, tmp_path):
    fake_ex = FakeEx({"BTC/USDT": {}})
    monkeypatch.setattr(cd, "get_ex", lambda name: fake_ex)
    monkeypatch.setattr(cd, "ENABLE", {"mexc": False, "blofin": False, "kraken": False})
    screen_dir = tmp_path / "backtest_output"
    screen_dir.mkdir()
    screen = pd.DataFrame([{"stem": "BTCUSDT"}])
    screen.to_csv(screen_dir / "screen_liqu_idio_20250101_000000.csv", index=False)
    monkeypatch.setattr(cd, "SCREEN_DIR", screen_dir)
    assert cd.build_jobs() == []
