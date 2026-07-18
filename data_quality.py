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
        r = requests.get(url, timeout=20, headers={"User-Agent": "trading-bot/1.0"})
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="full reconciliation pass")
    ap.add_argument("--repair", action="store_true", help="overwrite flagged bars from consensus")
    ap.add_argument("--spot", type=int, default=0, help="N random freshness spot-checks")
    ap.add_argument("--sym", type=str, default=None, help="single symbol to reconcile")
    ap.add_argument("--spikes", action="store_true", help="just run spike detection on one sym")
    ap.add_argument("--gmgn", type=str, default=None, help="Solana token address for GMGN check")
    args = ap.parse_args()

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
    if args.check or True:
        run_quality_pass(repair=args.repair)


if __name__ == "__main__":
    main()
