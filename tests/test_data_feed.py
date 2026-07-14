"""Edge-case / branch-coverage tests for data_feed.py (offline, ccxt mocked)."""
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch

import data_feed as df


class FakeEx:
    def __init__(self, bars):
        self.bars = bars
        self.rateLimit = 0

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=500):
        if since is None:
            return self.bars
        # backfill: return bars with ts >= since
        return [b for b in self.bars if b[0] >= since]


@pytest.fixture()
def fake_exchange(monkeypatch):
    bars = [
        [1_000_000_000_000, 100.0, 101.0, 99.0, 100.0, 10.0],
        [1_000_000_060_000, 101.0, 102.0, 100.0, 101.0, 11.0],
        [1_000_000_120_000, 102.0, 103.0, 101.0, 102.0, 12.0],
    ]
    ex = FakeEx(bars)
    monkeypatch.setattr(df, "get_exchange", lambda name=None: ex)
    return ex


# ── fetch_latest ────────────────────────────────────────────────────────────
def test_fetch_latest_returns_utc_sorted_df(fake_exchange):
    out = df.fetch_latest()
    assert list(out.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert out["ts"].dt.tz is not None
    assert out["ts"].is_monotonic_increasing
    assert not out["ts"].duplicated().any()


def test_fetch_latest_accepts_exchange_arg(fake_exchange):
    out = df.fetch_latest(exchange=fake_exchange)
    assert len(out) == 3


# ── append_to_history ──────────────────────────────────────────────────────
def test_append_to_history_creates_fresh(tmp_path, monkeypatch):
    monkeypatch.setattr(df, "HISTORY_CSV", tmp_path / "live_history.csv")
    new = pd.DataFrame({
        "ts": pd.to_datetime([1_000_000_000_000, 1_000_000_060_000], unit="ms", utc=True),
        "open": [100.0, 101.0], "high": [101.0, 102.0],
        "low": [99.0, 100.0], "close": [100.0, 101.0], "volume": [10.0, 11.0],
    })
    combined = df.append_to_history(new)
    assert len(combined) == 2
    assert (tmp_path / "live_history.csv").exists()
    assert not (tmp_path / "live_history.csv.tmp").exists()


def test_append_to_history_dedup_and_atomic(tmp_path, monkeypatch):
    monkeypatch.setattr(df, "HISTORY_CSV", tmp_path / "live_history.csv")
    existing = pd.DataFrame({
        "ts": pd.to_datetime([1_000_000_000_000], unit="ms", utc=True),
        "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [10.0],
    })
    existing.to_csv(tmp_path / "live_history.csv", index=False)
    new = pd.DataFrame({
        "ts": pd.to_datetime([1_000_000_000_000, 1_000_000_060_000], unit="ms", utc=True),
        "open": [100.0, 101.0], "high": [101.0, 102.0],
        "low": [99.0, 100.0], "close": [100.0, 101.0], "volume": [10.0, 11.0],
    })
    combined = df.append_to_history(new)
    assert len(combined) == 2  # dedup removes the overlapping ts
    on_disk = pd.read_csv(tmp_path / "live_history.csv", parse_dates=["ts"])
    assert len(on_disk) == 2
    assert not (tmp_path / "live_history.csv.tmp").exists()


def test_append_to_history_recovers_from_truncated_file(tmp_path, monkeypatch):
    monkeypatch.setattr(df, "HISTORY_CSV", tmp_path / "live_history.csv")
    (tmp_path / "live_history.csv").write_text(
        "ts,open,high,low,close,volume\n2025-01-01 00:00:00+00:00,1,1,1,1.0,5\n")
    new = pd.DataFrame({
        "ts": pd.to_datetime([1_000_000_000_000], unit="ms", utc=True),
        "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.0], "volume": [10.0],
    })
    combined = df.append_to_history(new)
    assert len(combined) == 2  # truncated + new both survived, valid frame


# ── backfill_since ─────────────────────────────────────────────────────────
def test_backfill_since_returns_bars_since(fake_exchange):
    since = pd.to_datetime(1_000_000_060_000, unit="ms", utc=True)
    out = df.backfill_since(since)
    assert len(out) == 2  # the last two bars
    assert (out["ts"] >= since.replace(microsecond=0)).all()


def test_backfill_since_empty_when_no_bars(fake_exchange):
    since = pd.to_datetime(9_999_999_999_999, unit="ms", utc=True)
    out = df.backfill_since(since)
    assert len(out) == 0
