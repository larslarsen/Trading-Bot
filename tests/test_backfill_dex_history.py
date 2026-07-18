"""Tests for backfill_dex_history.backfill_tokens (targeted new-token backfill)."""
import pandas as pd
import pytest

import backfill_dex_history as bdh


def test_backfill_tokens_writes_history(tmp_path, monkeypatch):
    # point DEX dir at a temp location
    monkeypatch.setattr(bdh, "DEX", tmp_path)
    # stub fetch_ohlcv to avoid network
    def fake_fetch(net, pool, limit):
        return pd.DataFrame([{"ts": "2026-01-01", "open": 1, "high": 2,
                              "low": 0.5, "close": 1.5, "volume": 10}])
    monkeypatch.setattr(bdh, "fetch_ohlcv", fake_fetch)

    rows = [{"symbol": "FOO", "network": "eth", "pool_address": "0xp1"},
            {"symbol": "BAR", "network": "bsc", "pool_address": "0xp2"}]
    done, skipped = bdh.backfill_tokens(rows, sleep=0, limit=10)
    assert done == 2
    assert (tmp_path / "FOO_1d_max.csv").exists()
    assert (tmp_path / "BAR_1d_max.csv").exists()
    assert len(pd.read_csv(tmp_path / "FOO_1d_max.csv")) == 1


def test_backfill_tokens_best_effort_on_error(tmp_path, monkeypatch):
    monkeypatch.setattr(bdh, "DEX", tmp_path)
    def boom(net, pool, limit):
        raise RuntimeError("404")
    monkeypatch.setattr(bdh, "fetch_ohlcv", boom)
    rows = [{"symbol": "BAD", "network": "eth", "pool_address": "0xp"}]
    # must not raise
    done, skipped = bdh.backfill_tokens(rows, sleep=0, limit=10)
    assert done == 0
    assert not (tmp_path / "BAD_1d_max.csv").exists()
