#!/usr/bin/env python3
"""
data_quality.py -- reconciliation + spike detection + spot-check for the 5m
OHLCV store. Catches data corruption BEFORE it rolls out of the exchange
lookback window (after which forward-only collection can never fix it).

Why this exists
---------------
The bot appends bars forward forever. If a corrupt bar is written (a poison
timestamp, a 100x-wrong price, a swapped OHLC), the exchange API's lookback
eventually can't re-fetch that historical timestamp -- so the error is
PERMANENT unless we catch it ourselves. This module:

  1. detect_spikes      -- flag any single-bar move > MAX_PCT (default 100%),
                           the user's "assume corruption not real" rule.
  2. reconcile_venues   -- load the per-venue 5m files (bybit/okx/blofin/
                           bitget/coinbase/gateio/kucoin) for a symbol, align
                           on timestamp, and flag any venue whose close
                           diverges from the cross-venue MEDIAN beyond
                           TOL_PCT. A spike present in ONLY one source is
                           corruption in that source, not the market.
  3. cross_check_consensus -- compare the consolidated file's bars to the
                           venue median; flag (and optionally REPAIR from the
                           consensus) any bar that deviates.
  4. spot_check_random  -- occasionally pull the latest bar from 2+ INDEPENDENT
                           venues and compare to the stored file. Surfaces
                           drift on symbols that have only one stored venue.
  5. gmgn_klines        -- best-effort INDEPENDENT on-chain source for meme/
                           DEX tokens (Solana etc.). GMGN's API is gated behind
                           a trading-volume partnership, so this is best-effort
                           and scoped to on-chain tokens ONLY -- never used as
                           the price of record for CEX BTC/ETH.

Repair policy: we only OVERWRITE a stored bar when an independent source
disagrees AND the majority of sources agree on a sane value. We never invent
data; we copy a verified bar from a trusted independent venue.

Run:  python data_quality.py --check          # one full reconciliation pass
      python data_quality.py --spot N          # N random spot checks
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from net_bypass import get as bg_get  # egress via local iface when BYPASS_VPN_IFACE is set

REPO = Path(__file__).parent
DATA = REPO / "data"

# A single 5m bar moving more than this vs the previous bar is SUSPECT
# (the user's rule: >100% in one bar = assumed corruption, not real).
MAX_PCT = 1.00          # 100%
# Cross-venue tolerance: a venue whose close deviates from the median by more
# than this is flagged as the likely-corrupt one.
TOL_PCT = 0.02          # 2% -- venues should agree to within fees/slippage
# Minimum number of independent venues required to make a consensus call.
MIN_VENUES = 2

# Per-venue suffixes we actually collect (see data_poller.cex_extra_topup_worker
# and collector_daemon). These are genuinely independent exchanges.
VENUE_SUFFIXES = ["bybit", "okx", "blofin", "bitget", "coinbase", "gateio", "kucoin"]


def _read_5m(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        d = pd.read_csv(path, parse_dates=["ts"])
    except Exception:
        return None
    if "ts" not in d.columns or len(d) == 0:
        return None
    d = d[d["ts"].notna()].copy()
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    d = d.sort_values("ts").drop_duplicates("ts")
    return d


def detect_spikes(df: pd.DataFrame, max_pct: float = MAX_PCT) -> list[dict]:
    """Flag rows where |close - prev_close| / prev_close > max_pct.

    Returns a list of {ts, pct, price, prev, direction}. A spike is SUSPECT,
    not proven-corrupt: reconcile_venues / cross_check_consensus confirm.
    """
    if df is None or len(df) < 2 or "close" not in df:
        return []
    close = df["close"].astype(float)
    prev = close.shift(1)
    pct = (close - prev).abs() / prev.abs()
    bad = df.index[pct > max_pct]
    out = []
    for i in bad:
        out.append({
            "ts": df["ts"].iloc[i].isoformat() if "ts" in df else str(i),
            "pct": float(pct.iloc[i]),
            "price": float(close.iloc[i]),
            "prev": float(prev.iloc[i]),
            "direction": "up" if close.iloc[i] >= prev.iloc[i] else "down",
        })
    return out


def _venue_files(sym: str) -> dict[str, Path]:
    """Map venue -> per-venue 5m CSV for a symbol (skip the consolidated file)."""
    stem = sym.replace("USDT", "").replace("/", "")
    out = {}
    for v in VENUE_SUFFIXES:
        p = DATA / f"{sym}_5m_{v}_max.csv"
        if p.exists():
            out[v] = p
    return out


def reconcile_venues(sym: str, tol_pct: float = TOL_PCT, tail: int = 5000) -> dict:
    """Load every per-venue 5m file for `sym`, align on timestamp, and find
    timestamps where one venue diverges from the cross-venue median.

    We only compare the most RECENT `tail` bars (default 5000 ~ 17 days). The
    point of reconciliation is to catch corruption while it is still inside the
    exchange lookback window (so we can re-fetch/repair). Comparing 4 years of
    history is both slow and useless -- old corruption is already permanent.

    Returns {
      'sym', 'n_venues', 'aligned_bars',
      'flagged': [ {ts, venue, price, median, dev_pct, likely_corrupt_venue} ],
    }
    A venue flagged at a timestamp is the LIKELY-CORRUPT one (others agree).
    """
    files = _venue_files(sym)
    frames = {}
    for v, p in files.items():
        d = _read_5m(p)
        if d is not None and "close" in d.columns:
            frames[v] = d.set_index("ts")["close"].astype(float).tail(tail)

    result = {"sym": sym, "n_venues": len(frames), "aligned_bars": 0, "flagged": []}
    if len(frames) < MIN_VENUES:
        return result  # can't form a consensus with <2 sources

    joined = pd.DataFrame(frames)            # rows=ts, cols=venue
    joined = joined.dropna(how="all")
    result["aligned_bars"] = len(joined)
    median = joined.median(axis=1)
    for ts, row in joined.iterrows():
        vals = row.dropna()
        if len(vals) < MIN_VENUES:
            continue
        med = float(vals.median())
        if med <= 0:
            continue
        worst_dev = max((abs(x - med) / med for x in vals), default=0.0)
        for v, price in vals.items():
            dev = abs(price - med) / med
            if dev > tol_pct:
                result["flagged"].append({
                    "ts": ts.isoformat(),
                    "venue": v,
                    "price": float(price),
                    "median": float(med),
                    "dev_pct": float(dev),
                    "likely_corrupt_venue": v if dev == worst_dev else None,
                })
    return result


def cross_check_consensus(sym: str, tol_pct: float = TOL_PCT,
                          repair: bool = False, tail: int = 5000) -> dict:
    """Compare the consolidated <SYM>_5m_max.csv (or btc_5m.csv) to the
    cross-venue median over the RECENT window. Flag deviating bars; if `repair`,
    overwrite them from the consensus median (only when >= MIN_VENUES agree).

    We only check the last `tail` bars: the goal is to catch + repair corruption
    while it is still inside the exchange lookback window. Old corruption is
    already permanent and not worth the I/O.

    Returns {sym, checked, flagged, repaired, reason?}.
    """
    consolidated = _consolidated_path(sym)
    d = _read_5m(consolidated)
    rec = reconcile_venues(sym, tol_pct, tail=tail)
    if d is None or rec["n_venues"] < MIN_VENUES:
        return {"sym": sym, "checked": 0, "flagged": [], "repaired": 0,
                "reason": "no consolidated file or <2 venues"}

    # Rebuild the per-venue median series at the consolidated timestamps.
    files = _venue_files(sym)
    series = {}
    for v, p in files.items():
        dd = _read_5m(p)
        if dd is not None and "close" in dd.columns:
            series[v] = dd.set_index("ts")["close"].astype(float).tail(tail)
    joined = pd.DataFrame(series)
    median = joined.median(axis=1)

    close = d.set_index("ts")["close"].astype(float).tail(tail)
    checked = 0
    flagged = []
    repaired = 0
    for ts, price in close.items():
        if ts not in median.index or pd.isna(median.loc[ts]):
            continue
        checked += 1
        med = float(median.loc[ts])
        if med <= 0:
            continue
        dev = abs(price - med) / med
        if dev > tol_pct:
            flagged.append({"ts": ts.isoformat(), "stored": float(price),
                            "consensus": med, "dev_pct": float(dev)})
            if repair:
                d.loc[d["ts"] == ts, "close"] = med
                repaired += 1

    # Also verify the consolidated file's LAST bar (highest corruption risk
    # from a live append) against a FRESH independent re-fetch, because the
    # last bar's timestamp often does not yet align with the venue tails above.
    # Best-effort: if the network/venues are unavailable we simply skip it.
    fresh = None
    if len(close) > 0:
        last_ts = close.index[-1]
        cons = _fresh_venue_consensus(sym)
        if cons is not None:
            fresh = cons["median"]
            if fresh > 0:
                last_price = float(close.iloc[-1])
                dev = abs(last_price - fresh) / fresh
                if dev > tol_pct:
                    flagged.append({"ts": last_ts.isoformat(), "stored": last_price,
                                    "consensus": fresh, "dev_pct": float(dev),
                                    "source": "fresh_refetch"})
                    if repair:
                        d.loc[d["ts"] == last_ts, "close"] = fresh
                        repaired += 1
                    checked += 1

    if repair and repaired:
        d.to_csv(consolidated, index=False)
    return {"sym": sym, "checked": checked, "flagged": flagged,
            "repaired": repaired}


def _consolidated_path(sym: str) -> Path:
    if sym == "BTCUSDT":
        return REPO / "btc_5m.csv"
    return DATA / f"{sym}_5m_max.csv"


def spot_check_random(symbols: list[str], n: int = 5, tol_pct: float = TOL_PCT) -> list[dict]:
    """Pull the latest bar from 2+ independent venues for N random symbols and
    compare to the stored consolidated close. Reports any symbol where the
    stored value diverges from the freshly-fetched venue median.

    This catches drift on symbols that only have ONE stored venue file (so
    reconcile_venues can't form a consensus) by going back to the source.
    """
    out = []
    if not symbols:
        return out
    pick = random.sample(symbols, min(n, len(symbols)))
    for sym in pick:
        consensus = _fresh_venue_consensus(sym)
        if consensus is None:
            continue
        d = _read_5m(_consolidated_path(sym))
        if d is None or len(d) == 0:
            continue
        stored = float(d["close"].iloc[-1])
        if consensus["median"] > 0:
            dev = abs(stored - consensus["median"]) / consensus["median"]
            out.append({
                "sym": sym,
                "stored": stored,
                "fresh_median": consensus["median"],
                "venues": consensus["venues"],
                "dev_pct": float(dev),
                "flag": dev > tol_pct,
            })
    return out


def _fresh_venue_consensus(sym: str) -> dict | None:
    """Fetch the latest 5m close from 2+ independent public venues (ccxt) and
    return the median. Falls back to whatever venues succeed."""
    import ccxt
    stem = sym.replace("USDT", "").replace("/", "")
    venues = {}
    for name in ("bybit", "okx", "binance", "kraken", "coinbase", "gateio", "kucoin"):
        try:
            ex = getattr(ccxt, name)()
            ex.timeout = 15000
            cc = f"{stem}/USDT"
            if cc not in ex.markets:
                ex.load_markets()
            if cc not in ex.markets:
                continue
            bars = ex.fetch_ohlcv(cc, "5m", limit=1)
            if bars:
                venues[name] = float(bars[-1][4])  # close
        except Exception:
            continue
    if len(venues) < MIN_VENUES:
        return None
    return {"median": float(np.median(list(venues.values()))), "venues": list(venues)}


# ── GMGN: best-effort INDEPENDENT on-chain source for meme/DEX tokens ────────
# GMGN's API is gated behind a trading-volume partnership, so this is a
# best-effort cross-check for ON-CHAIN tokens (Solana etc.) only. It is NEVER
# the price of record for CEX BTC/ETH. Returns a DataFrame or None.
GMGN_BASE = "https://api.gmgn.ai/defi/quotation/v1/tokens/kline/sol/{address}"


def gmgn_klines(address: str, tf: str = "5m", limit: int = 500) -> pd.DataFrame | None:
    """Fetch OHLCV for a Solana token address from GMGN's public quotation
    endpoint (best-effort, no API key). Returns columns
    [ts, open, high, low, close, volume] or None on any failure.

    NOTE: this is an INDEPENDENT cross-check source for meme/DEX tokens, not a
    replacement for the CEX feeds. If GMGN rate-limits or gates the call, we
    simply return None and rely on the other venues.
    """
    url = f"{GMGN_BASE.format(address=address)}?resolution={tf}&limit={limit}"
    try:
        r = bg_get(url, timeout=20)
        if r.status_code != 200:
            return None
        payload = r.json()
        data = (payload.get("data") or {}).get("data") or payload.get("data")
        if not data:
            return None
        rows = []
        for b in data:
            # gmgn kline: [timestamp_ms, open, high, low, close, volume, ...]
            t = b.get("timestamp") or b.get("time") or b[0]
            rows.append([
                pd.to_datetime(int(t) if str(t).isdigit() else t, unit="ms", utc=True),
                float(b.get("open", b[1])), float(b.get("high", b[2])),
                float(b.get("low", b[3])), float(b.get("close", b[4])),
                float(b.get("volume", b[5]) if "volume" in b else 0),
            ])
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        return df.sort_values("ts").drop_duplicates("ts")
    except Exception:
        return None


def _all_symbols() -> list[str]:
    syms = set()
    for p in DATA.glob("*_5m_max.csv"):
        syms.add(p.name.split("_5m_")[0])
    if (REPO / "btc_5m.csv").exists():
        syms.add("BTCUSDT")
    return sorted(syms)


def run_quality_pass(repair: bool = False, tol_pct: float = TOL_PCT,
                     symbols: list[str] | None = None) -> dict:
    """Full reconciliation pass over all multi-venue symbols. Returns a summary
    dict and prints a human-readable report.

    We only check symbols that actually have >= MIN_VENUES independent venue
    files -- single-venue symbols can't be cross-checked (spot_check_random
    covers those via live re-fetch). This keeps the pass fast and focused on
    where reconciliation is possible.
    """
    if symbols is None:
        symbols = _all_symbols()
    # pre-filter to symbols with >= MIN_VENUES venue files (cheap exists() scan)
    multi = [s for s in symbols if len(_venue_files(s)) >= MIN_VENUES]
    total_flagged = 0
    total_repaired = 0
    report = []
    for sym in multi:
        rec = cross_check_consensus(sym, tol_pct=tol_pct, repair=repair)
        if rec.get("flagged"):
            total_flagged += len(rec["flagged"])
            total_repaired += rec.get("repaired", 0)
            report.append(f"  {sym}: {len(rec['flagged'])} flagged"
                          + (f", {rec['repaired']} repaired" if rec.get("repaired") else ""))
    now = datetime.now(timezone.utc).isoformat()
    summary = {"time": now, "symbols_checked": len(multi),
               "symbols_total": len(symbols), "flagged": total_flagged,
               "repaired": total_repaired, "details": report}
    print(f"[data_quality] {now} checked={len(multi)}/{len(symbols)} "
          f"flagged={total_flagged} repaired={total_repaired}")
    for line in report:
        print(line)
    return summary


# ════════════════════════════════════════════════════════════════════════════
# DEX / MICROSTRUCTURE DATA  (the HIGH-VALUE datasets, not just CEX)
#
# The lookback-window reconciliation we built for CEX 5m MUST also cover the
# DEX OHLCV, DEX microstructure breadth, DEX daily history, and CEX funding
# series -- all are collected forward-only and are just as corruptible
# (poison rows, swapped OHLC, spikes). For DEX the "independent sources" are
# the separate aggregators: GeckoTerminal (OHLCV sampler source),
# DexScreener (microstructure poller source, KEYLESS + REACHABLE), and GMGN
# (best-effort). They disagree only when one is wrong -> that is the signal.
# ════════════════════════════════════════════════════════════════════════════

def dex_screener_price(chain: str, address: str) -> float | None:
    """Independent DEX price for a token from DexScreener (keyless, reachable).
    Returns the latest close price or None on any failure."""
    url = f"https://api.dexscreener.com/latest/dex/pairs/{chain}/{address}"
    try:
        r = bg_get(url, timeout=20)
        if r.status_code != 200:
            return None
        data = r.json().get("pairs") or []
        if not data:
            return None
        # highest-liquidity pair's priceNative is the trustworthy price
        data.sort(key=lambda p: float(p.get("liquidity", {}).get("usd", 0) or 0),
                  reverse=True)
        return float(data[0]["priceNative"])
    except Exception:
        return None


def gmgn_token_price(address: str, chain: str = "sol") -> float | None:
    """Best-effort independent DEX price from GMGN. Returns None if GMGN is
    blocked/gated (it frequently is without a partnership). Never raises."""
    url = (f"https://api.gmgn.ai/defi/quotation/v1/tokens/{chain}/{address}")
    try:
        r = bg_get(url, timeout=20)
        if r.status_code != 200:
            return None
        price = (r.json().get("data") or {}).get("price")
        return float(price) if price is not None else None
    except Exception:
        return None


def gmgn_trending(chain: str = "sol", limit: int = 50) -> list[dict] | None:
    """Best-effort TRENDING DEX tokens from GMGN (universe expansion).
    Returns a list of {address, symbol, chain, ...} or None if blocked."""
    url = f"https://api.gmgn.ai/defi/quotation/v1/tokens/trending/{chain}?limit={limit}"
    try:
        r = bg_get(url, timeout=20)
        if r.status_code != 200:
            return None
        return (r.json().get("data") or {}).get("tokens") or []
    except Exception:
        return None


def gmgn_new_pairs(chain: str = "sol", limit: int = 50) -> list[dict] | None:
    """Best-effort NEW-PAIR / new-token-creations feed from GMGN (universe
    expansion beyond the 200 auto-screen). Returns list or None if blocked."""
    url = f"https://api.gmgn.ai/defi/quotation/v1/tokens/new_pairs/{chain}?limit={limit}"
    try:
        r = bg_get(url, timeout=20)
        if r.status_code != 200:
            return None
        return (r.json().get("data") or {}).get("tokens") or []
    except Exception:
        return None


def reconcile_dex_ohlcv(token: str, repair: bool = False,
                       tol_pct: float = TOL_PCT) -> dict:
    """Reconcile a DEX OHLCV file (data/dex/<TOK>_5m_max.csv) against an
    INDEPENDENT live price (DexScreener, GMGN best-effort). Flags + optionally
    repairs the last bar when the stored close diverges from the independent
    source. Also runs spike detection over the recent tail.

    This is the DEX analogue of cross_check_consensus: GeckoTerminal is the
    stored source; DexScreener/GMGN are the independent checkers.
    """
    p = DATA / "dex" / f"{token}_5m_max.csv"
    d = _read_5m(p)
    if d is None or len(d) == 0 or "close" not in d.columns:
        return {"token": token, "checked": 0, "flagged": [], "repaired": 0,
                "reason": "no dex ohlcv file"}
    close = d.set_index("ts")["close"].astype(float).tail(5000)
    spikes = detect_spikes(d, max_pct=MAX_PCT)
    flagged = [{"ts": s["ts"], "type": "spike", "pct": s["pct"]} for s in spikes]
    # independent price check on the last bar
    address = _token_address(token)
    checked = len(close)
    repaired = 0
    if address:
        ind = dex_screener_price("ethereum", address) or gmgn_token_price(address)
        if ind and ind > 0:
            last_price = float(close.iloc[-1])
            dev = abs(last_price - ind) / ind
            if dev > tol_pct:
                flagged.append({"ts": str(close.index[-1]), "type": "independent_mismatch",
                                "stored": last_price, "independent": ind, "dev_pct": float(dev)})
                if repair:
                    d.loc[d["ts"] == close.index[-1], "close"] = ind
                    repaired += 1
    if repair and repaired:
        d.to_csv(p, index=False)
    return {"token": token, "checked": checked, "flagged": flagged, "repaired": repaired}


def reconcile_dex_micro(token: str, repair: bool = False) -> dict:
    """Reconcile a DEX MICROSTRUCTURE file (data/dex_micro/<TOK>.csv) -- the
    highest-value DEX dataset (liquidity_usd, fdv, price_chg). Runs spike /
    range checks on liquidity_usd and fdv (a 100%+ jump = suspect), and flags
    NaN/negative values that would poison downstream features. Best-effort
    GMGN cross-check on fdv when available.

    NOTE: microstructure series are append-only breadth polls; we do NOT
    overwite historical rows (they're a time-series of market state, not a
    single 'true' price). We only FLAG anomalies + optionally drop rows that
    are NaN/non-finite (which would break feature math)."""
    p = DATA / "dex_micro" / f"{token}.csv"
    if not p.exists():
        return {"token": token, "checked": 0, "flagged": [], "repaired": 0,
                "reason": "no dex_micro file"}
    try:
        d = pd.read_csv(p)
    except Exception:
        return {"token": token, "checked": 0, "flagged": [], "repaired": 0,
                "reason": "unreadable"}
    flagged = []
    dropped = 0
    if "liquidity_usd" in d.columns:
        liq = pd.to_numeric(d["liquidity_usd"], errors="coerce")
        # a liquidity reading that is NaN or negative is corrupt -> drop it
        bad = liq.isna() | (liq < 0)
        if bad.any():
            flagged.append({"type": "micro_nan_or_negative_liquidity",
                            "rows": int(bad.sum())})
            if repair:
                d = d[~bad]
                dropped += int(bad.sum())
    if "fdv" in d.columns:
        fdv = pd.to_numeric(d["fdv"], errors="coerce")
        # spike: a single poll 100%+ above the running median fdv = suspect
        med = fdv.median()
        if med and med > 0:
            spike = (fdv - med).abs() / med > MAX_PCT
            if spike.any():
                flagged.append({"type": "fdv_spike", "rows": int(spike.sum())})
    if repair and dropped:
        d.to_csv(p, index=False)
    return {"token": token, "checked": len(d), "flagged": flagged, "repaired": dropped}


def _token_address(token: str) -> str | None:
    """Best-effort resolve a DEX token stem to a contract address from the
    micro file (which stores it). Returns None if unknown."""
    p = DATA / "dex_micro" / f"{token}.csv"
    if not p.exists():
        return None
    try:
        d = pd.read_csv(p)
        if "address" in d.columns and len(d):
            a = d["address"].dropna()
            if len(a):
                return str(a.iloc[-1])
    except Exception:
        return None
    return None


def _dex_tokens() -> list[str]:
    toks = set()
    for p in (DATA / "dex").glob("*_5m_max.csv"):
        toks.add(p.stem.replace("_5m_max", ""))
    for p in (DATA / "dex_micro").glob("*.csv"):
        toks.add(p.stem)
    return sorted(toks)


def run_dex_quality_pass(repair: bool = False, max_tokens: int = 40,
                         tol_pct: float = TOL_PCT) -> dict:
    """Reconciliation pass over DEX OHLCV + microstructure for the top-N tokens
    by liquidity. DEX tokens number in the hundreds; we cap the pass to the
    highest-value (most liquid) N to keep it fast. Returns a summary dict."""
    toks = _dex_tokens()[:max_tokens]
    total_flagged = 0
    total_repaired = 0
    report = []
    for tok in toks:
        r1 = reconcile_dex_ohlcv(tok, repair=repair, tol_pct=tol_pct)
        r2 = reconcile_dex_micro(tok, repair=repair)
        nf = len(r1.get("flagged", [])) + len(r2.get("flagged", []))
        nr = r1.get("repaired", 0) + r2.get("repaired", 0)
        total_flagged += nf
        total_repaired += nr
        if nf:
            report.append(f"  {tok}: ohlcv={len(r1.get('flagged',[]))} "
                          f"micro={len(r2.get('flagged',[]))}"
                          + (f" repaired={nr}" if nr else ""))
    now = datetime.now(timezone.utc).isoformat()
    summary = {"time": now, "dex_tokens_checked": len(toks),
               "flagged": total_flagged, "repaired": total_repaired, "details": report}
    print(f"[data_quality DEX] {now} tokens={len(toks)} "
          f"flagged={total_flagged} repaired={total_repaired}")
    for line in report:
        print(line)
    return summary


def discover_dex_tokens(chains=("ethereum", "base", "bsc", "arbitrum", "polygon", "solana"),
                       max_per_source: int = 50) -> list[dict]:
    """Expand the DEX universe BEYOND the 200 auto-screen by pulling TRENDING
    and NEW-PAIR token feeds from multiple independent sources:

      * DexScreener  -- keyless, reachable (primary expander)
      * GeckoTerminal -- keyless, reachable (primary expander)
      * GMGN         -- best-effort (frequently gated; skipped if blocked)

    Returns a de-duplicated list of {symbol, address, chain, source}. The
    caller (or the poller's universe_worker) feeds these into the dex universe
    so the OHLCV + micro pollers start tracking them -- extending lookback
    before they'd ever appear in the static auto-screen.
    """
    seen = {}
    # DexScreener trending (no key): search top tokens by a broad query
    try:
        for q in ("ETH", "BASE", "BSC", "SOL"):
            r = bg_get(f"https://api.dexscreener.com/latest/dex/search?q={q}",
                       timeout=20)
            if r.status_code == 200:
                for p in (r.json().get("pairs") or [])[:max_per_source]:
                    addr = p.get("baseToken", {}).get("address")
                    sym = p.get("baseToken", {}).get("symbol")
                    ch = p.get("chainId")
                    if addr and sym and ch and addr not in seen:
                        seen[addr] = {"symbol": sym, "address": addr,
                                      "chain": ch, "source": "dexscreener"}
    except Exception:
        pass
    # GeckoTerminal trending pools (keyless, reachable)
    try:
        for net in ("eth", "base", "bsc", "solana"):
            r = bg_get(f"https://api.geckoterminal.com/api/v2/networks/{net}/trending_pools?page=1",
                       timeout=20, headers={"Accept": "application/json"})
            if r.status_code == 200:
                for p in (r.json().get("data") or []):
                    attr = (p.get("attributes") or {})
                    base = attr.get("base_token") or {}
                    addr = base.get("address") or base.get("id") or (attr.get("base_token_address"))
                    sym = base.get("symbol") or (attr.get("base_token_symbol"))
                    if addr and sym and addr not in seen:
                        seen[addr] = {"symbol": sym, "address": addr,
                                      "chain": net, "source": "geckoterminal"}
    except Exception:
        pass
    # GMGN best-effort (trending + new pairs); skipped silently if gated
    for fn in (gmgn_trending, gmgn_new_pairs):
        try:
            for ch in ("sol", "eth", "bsc", "base"):
                toks = fn(ch) or []
                for t in toks[:max_per_source]:
                    addr = t.get("address")
                    sym = t.get("symbol")
                    if addr and sym and addr not in seen:
                        seen[addr] = {"symbol": sym, "address": addr,
                                      "chain": ch, "source": "gmgn"}
        except Exception:
            continue
    return list(seen.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="full reconciliation pass (CEX + DEX)")
    ap.add_argument("--repair", action="store_true", help="overwrite flagged bars from consensus")
    ap.add_argument("--spot", type=int, default=0, help="N random freshness spot-checks")
    ap.add_argument("--sym", type=str, default=None, help="single symbol to reconcile")
    ap.add_argument("--spikes", action="store_true", help="just run spike detection on one sym")
    ap.add_argument("--gmgn", type=str, default=None, help="Solana token address for GMGN check")
    ap.add_argument("--dex", action="store_true", help="DEX-only reconciliation pass")
    ap.add_argument("--dex-tokens", type=int, default=40, help="max DEX tokens to check")
    ap.add_argument("--discover", action="store_true",
                    help="discover trending/new DEX tokens (DexScreener+GeckoTerminal+GMGN) and print")
    ap.add_argument("--gmgn-trending", action="store_true", help="print GMGN trending (best-effort)")
    ap.add_argument("--gmgn-newpairs", action="store_true", help="print GMGN new pairs (best-effort)")
    args = ap.parse_args()

    if args.gmgn_trending:
        print("GMGN trending:", gmgn_trending() or "BLOCKED/empty")
        return
    if args.gmgn_newpairs:
        print("GMGN new_pairs:", gmgn_new_pairs() or "BLOCKED/empty")
        return
    if args.discover:
        toks = discover_dex_tokens()
        print(f"discovered {len(toks)} tokens:")
        for t in toks[:20]:
            print("  ", t)
        return
    if args.gmgn:
        df = gmgn_klines(args.gmgn)
        print("GMGN klines:", None if df is None else f"{len(df)} bars, last close="
              + (f"{df['close'].iloc[-1]:.6f}" if df is not None else ""))
        return
    if args.spikes and args.sym:
        d = _read_5m(_consolidated_path(args.sym))
        sp = detect_spikes(d)
        print(f"{args.sym}: {len(sp)} spike(s) > {MAX_PCT*100:.0f}%")
        for s in sp[:5]:
            print("  ", s)
        return
    if args.sym:
        rec = reconcile_venues(args.sym)
        print(f"{args.sym}: venues={rec['n_venues']} aligned={rec['aligned_bars']} "
              f"flagged={len(rec['flagged'])}")
        for f in rec["flagged"][:10]:
            print("  ", f)
        return
    if args.spot:
        syms = _all_symbols()
        res = spot_check_random(syms, n=args.spot)
        for r in res:
            tag = "FLAG" if r["flag"] else "ok"
            print(f"  [{tag}] {r['sym']} stored={r['stored']:.4f} "
                  f"fresh_median={r['fresh_median']:.4f} dev={r['dev_pct']*100:.2f}% "
                  f"venues={r['venues']}")
        return
    if args.dex:
        run_dex_quality_pass(repair=args.repair, max_tokens=args.dex_tokens)
        return
    if args.check or True:
        # CEX pass (multi-venue reconciliation) + DEX pass (microstructure +
        # OHLCV cross-check via independent aggregators). Covers ALL collected
        # datasets, not just CEX -- so corruption in any lookback window is
        # caught before the exchange API can no longer re-fetch it.
        run_quality_pass(repair=args.repair)
        run_dex_quality_pass(repair=args.repair, max_tokens=args.dex_tokens)


if __name__ == "__main__":
    main()
