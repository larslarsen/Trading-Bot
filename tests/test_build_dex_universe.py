"""Tests for build_dex_universe.discover_and_merge (DEX universe expansion)."""
import pandas as pd
import pytest

import build_dex_universe as bdu
import data_quality as dq
import dex_micro_poller as mpol


def test_discover_and_merge_adds_new_tokens(tmp_path, monkeypatch):
    uni = tmp_path / "dex_universe.csv"
    pd.DataFrame([{"symbol": "EXIST", "network": "eth",
                   "pool_address": "0x1", "quote": "USDC", "vol24h": 1.0}]).to_csv(uni, index=False)
    monkeypatch.setattr(bdu, "OUT", uni)

    fake = [
        {"symbol": "NEW1", "address": "0xa", "chain": "eth", "source": "dexscreener"},
        {"symbol": "NEW2", "address": "0xb", "chain": "bsc", "source": "gmgn"},
    ]
    monkeypatch.setattr(dq, "discover_dex_tokens", lambda *a, **k: fake)
    monkeypatch.setattr(mpol, "resolve_address", lambda sym: ("eth", "0xPOOL_" + sym))

    added = bdu.discover_and_merge(top_n=10, existing={"EXIST"})
    df = pd.read_csv(uni)
    assert added == 2
    assert len(df) == 3
    assert set(df["symbol"]) == {"EXIST", "NEW1", "NEW2"}
    assert not df.duplicated(subset=["network", "symbol"]).any()


def test_discover_and_merge_skips_known_and_stables(tmp_path, monkeypatch):
    uni = tmp_path / "dex_universe.csv"
    pd.DataFrame([{"symbol": "EXIST", "network": "eth", "pool_address": "0x1",
                   "quote": "USDC", "vol24h": 1.0}]).to_csv(uni, index=False)
    monkeypatch.setattr(bdu, "OUT", uni)
    fake = [
        {"symbol": "EXIST", "address": "0xa", "chain": "eth", "source": "x"},
        {"symbol": "USDC", "address": "0xb", "chain": "eth", "source": "x"},
        {"symbol": "GOOD", "address": "0xc", "chain": "base", "source": "x"},
    ]
    monkeypatch.setattr(dq, "discover_dex_tokens", lambda *a, **k: fake)
    monkeypatch.setattr(mpol, "resolve_address", lambda sym: ("base", "0xp_" + sym))

    added = bdu.discover_and_merge(top_n=10, existing={"EXIST"})
    assert added == 1
    df = pd.read_csv(uni)
    assert set(df["symbol"]) == {"EXIST", "GOOD"}


def test_discover_and_merge_best_effort_on_failure(tmp_path, monkeypatch):
    uni = tmp_path / "dex_universe.csv"
    pd.DataFrame([{"symbol": "EXIST", "network": "eth", "pool_address": "0x1",
                   "quote": "USDC", "vol24h": 1.0}]).to_csv(uni, index=False)
    monkeypatch.setattr(bdu, "OUT", uni)
    monkeypatch.setattr(dq, "discover_dex_tokens", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("gated")))
    added = bdu.discover_and_merge(top_n=10, existing=set())
    assert added == 0
    assert len(pd.read_csv(uni)) == 1
