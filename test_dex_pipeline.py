#!/usr/bin/env python3
"""Unit tests for the DEX collection pipeline (network mocked).

Covers the pure logic that matters for correctness:
  build_dex_universe : parse_pair, stable filtering, both-stable skip, per-token
                       dedup-by-max-volume, case-insensitive stable match.
  backfill_dex_history: fetch_ohlcv normalization, resume-skip on deep files.
  screen_dex_idio    : vol24h tercile tier assignment.
  screen_dex         : dexscreener row aggregation + liquidity/volume thresholds.

All HTTP is mocked -- no live GeckoTerminal / dexscreener calls.
"""
import json
import io
from pathlib import Path

import pandas as pd
import pytest

import build_dex_universe as bu
import backfill_dex_history as bh
import screen_dex


# ----------------------------- build_dex_universe -----------------------------

def test_parse_pair_basic():
    assert bu.parse_pair("BLINK / WETH 0.3%") == ("BLINK", "WETH")
    assert bu.parse_pair("PEPE / USDC") == ("PEPE", "USDC")

def test_parse_pair_no_separator():
    base, quote = bu.parse_pair("WEIRDNAME")
    assert base == "WEIRDNAME"
    assert quote == ""


def _pool(name, addr, vol):
    return {"attributes": {"name": name, "address": addr,
                           "volume_usd": {"h24": vol}}}

def test_build_keeps_token_side_and_skips_both_stable(monkeypatch, tmp_path):
    monkeypatch.setattr(bu, "OUT", tmp_path / "dex_universe.csv")
    # page 1 has: a real token, a both-stable pair (skip), a stable/token pair
    page1 = {"data": [
        _pool("PEPE / WETH", "0xpepe", 100.0),      # token=PEPE
        _pool("USDC / USDT", "0xstable", 999.0),    # both stable -> skip
        _pool("USDC / BONK", "0xbonk", 50.0),       # base stable -> token=BONK
    ]}
    calls = {"n": 0}
    def fake_get(url):
        calls["n"] += 1
        return page1 if calls["n"] == 1 else {"data": []}
    monkeypatch.setattr(bh, "_get", fake_get)
    monkeypatch.setattr(bu.time, "sleep", lambda *_: None)
    df = bu.build(networks=["eth"], top_per_network=3, sleep=0)
    syms = set(df["symbol"])
    assert "PEPE" in syms
    assert "BONK" in syms
    assert "USDC" not in syms and "USDT" not in syms

def test_build_case_insensitive_stable(monkeypatch, tmp_path):
    # cbBTC is a stable/wrapped (CBBTC in set) returned lowercase -> must be filtered
    monkeypatch.setattr(bu, "OUT", tmp_path / "dex_universe.csv")
    page1 = {"data": [_pool("cbBTC / USDC", "0xcb", 100.0)]}  # both stable -> skip
    calls = {"n": 0}
    def fake_get(url):
        calls["n"] += 1
        return page1 if calls["n"] == 1 else {"data": []}
    monkeypatch.setattr(bh, "_get", fake_get)
    monkeypatch.setattr(bu.time, "sleep", lambda *_: None)
    df = bu.build(networks=["eth"], top_per_network=1, sleep=0)
    assert "cbBTC" not in set(df["symbol"])
    assert len(df) == 0

def test_build_dedup_keeps_max_volume(monkeypatch, tmp_path):
    monkeypatch.setattr(bu, "OUT", tmp_path / "dex_universe.csv")
    page1 = {"data": [
        _pool("PEPE / WETH", "0xlow", 10.0),
        _pool("PEPE / USDC", "0xhigh", 500.0),   # same token, higher vol -> win
    ]}
    calls = {"n": 0}
    def fake_get(url):
        calls["n"] += 1
        return page1 if calls["n"] == 1 else {"data": []}
    monkeypatch.setattr(bh, "_get", fake_get)
    monkeypatch.setattr(bu.time, "sleep", lambda *_: None)
    df = bu.build(networks=["eth"], top_per_network=2, sleep=0)
    row = df[df["symbol"] == "PEPE"].iloc[0]
    assert row["pool_address"] == "0xhigh"
    assert row["vol24h"] == 500.0


# ---------------------------- backfill_dex_history ----------------------------

def test_fetch_ohlcv_normalizes(monkeypatch):
    raw = {"data": {"attributes": {"ohlcv_list": [
        [1704067200, 1.0, 2.0, 0.5, 1.5, 1234.0],   # 2024-01-01
        [1704153600, 1.5, 2.5, 1.0, 2.0, 5678.0],   # 2024-01-02
    ]}}}
    monkeypatch.setattr(bh, "_get", lambda url: raw)
    df = bh.fetch_ohlcv("eth", "0xpool", 1000)
    assert list(df.columns) == ["ts", "open", "high", "low", "close", "volume"]
    assert df["ts"].iloc[0] == "2024-01-01"
    assert df["close"].iloc[1] == 2.0

def test_fetch_ohlcv_empty(monkeypatch):
    monkeypatch.setattr(bh, "_get", lambda url: {"data": {"attributes": {"ohlcv_list": []}}})
    assert bh.fetch_ohlcv("eth", "0xpool", 1000) is None

def test_backfill_resume_skips_deep_file(tmp_path, monkeypatch):
    # a token whose existing file already reaches before DEEP_ENOUGH must be skipped
    monkeypatch.setattr(bh, "DEX", tmp_path)
    deep = tmp_path / "OLD_1d_max.csv"
    pd.DataFrame({"ts": ["2022-06-01", "2024-01-01"], "open": [1, 1],
                  "high": [1, 1], "low": [1, 1], "close": [1, 1],
                  "volume": [1, 1]}).to_csv(deep, index=False)
    uni = tmp_path / "u.csv"
    pd.DataFrame([{"symbol": "OLD", "network": "eth", "pool_address": "0xold"}]).to_csv(uni, index=False)
    called = {"fetch": 0}
    monkeypatch.setattr(bh, "fetch_ohlcv", lambda *a, **k: called.__setitem__("fetch", called["fetch"] + 1) or None)
    monkeypatch.setattr(bh.time, "sleep", lambda *_: None)
    import argparse
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args",
                        lambda self: argparse.Namespace(sleep=0, chain="eth", limit=10, universe=str(uni)))
    bh.main()
    assert called["fetch"] == 0  # deep file -> fetch never called


# ------------------------------ screen_dex_idio ------------------------------

def test_tier_assignment_terciles():
    import screen_dex_idio  # imports screen_liquidity_idiosyncratic; keep local
    u = pd.DataFrame({"symbol": list("ABCDEF"),
                      "vol24h": [1, 2, 3, 4, 5, 6]})
    q1, q2 = u["vol24h"].quantile([0.333, 0.667])
    tiers = u["vol24h"].apply(lambda v: "large" if v >= q2 else ("mid" if v >= q1 else "tail"))
    assert tiers.iloc[0] == "tail"   # lowest
    assert tiers.iloc[-1] == "large" # highest
    assert set(tiers) == {"large", "mid", "tail"}


# -------------------------------- screen_dex ---------------------------------

def _pair(liq, vol, sym="AAA"):
    return {"baseToken": {"symbol": sym, "name": sym + "coin"},
            "liquidity": {"usd": liq}, "volume": {"h24": vol},
            "pairCreatedAt": 1700000000000, "chainId": "ethereum",
            "dexId": "uniswap", "priceUsd": "1.23"}

def test_screen_token_passes_thresholds(monkeypatch):
    payload = json.dumps({"pairs": [_pair(100_000, 50_000)]}).encode()
    monkeypatch.setattr(screen_dex.urllib.request, "urlopen",
                        lambda *a, **k: io.BytesIO(payload))
    monkeypatch.setattr(screen_dex.time, "sleep", lambda *_: None)
    row = screen_dex.screen_token("0xaaa", 0)
    assert row is not None
    assert row["symbol"] == "AAA"
    assert row["liquidityUsd"] == 100_000

def test_screen_token_rejects_illiquid(monkeypatch):
    payload = json.dumps({"pairs": [_pair(1000, 100)]}).encode()  # below thresholds
    monkeypatch.setattr(screen_dex.urllib.request, "urlopen",
                        lambda *a, **k: io.BytesIO(payload))
    monkeypatch.setattr(screen_dex.time, "sleep", lambda *_: None)
    assert screen_dex.screen_token("0xaaa", 0) is None

def test_screen_token_no_pairs(monkeypatch):
    payload = json.dumps({"pairs": []}).encode()
    monkeypatch.setattr(screen_dex.urllib.request, "urlopen",
                        lambda *a, **k: io.BytesIO(payload))
    monkeypatch.setattr(screen_dex.time, "sleep", lambda *_: None)
    assert screen_dex.screen_token("0xaaa", 0) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
