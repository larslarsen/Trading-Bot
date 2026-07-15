#!/usr/bin/env python3
"""
DEX 5m FORWARD collector — DexScreener (free, no key).

Polls the live price of every token in dex_universe.csv every 5m and appends a
5m bar to data/<SYM>_5m_dex_max.csv — the SAME store + naming scheme the CEX
collector uses (data/<SYM>_<tf>_<src>_max.csv). This unifies CEX + DEX into one
place with one method per timeframe.

Rate discipline (priority data first, never lock ourselves out):
- Pair addresses are resolved ONCE per day and cached (dex_pairs_cache.json),
  so the hot 5m loop never does discovery searches.
- PAUSE between price polls; a full 223-token cycle is ~1 min, well under any
  sane budget. If a cycle runs long we just pick up next time.

Usage:
    python dex_forward_collector.py            # one 5m snapshot, append
    python dex_forward_collector.py --loop     # run forever, snap every 5m
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
CACHE = ROOT / "dex_pairs_cache.json"
UNIVERSE = ROOT / "dex_universe.csv"
BASE = "https://api.dexscreener.com/latest/dex"

PAUSE = 0.25          # seconds between price polls
RESOLVE_EVERY_S = 86400  # re-resolve pair addresses daily


def _get(url, timeout=15, **kwargs):
    r = requests.get(url, timeout=timeout, **kwargs)
    r.raise_for_status()
    return r.json()


def resolve_pairs(force=False):
    """Map symbol -> (chain, pairAddress) for the whole universe. Cached daily."""
    now = time.time()
    if not force and CACHE.exists():
        try:
            blob = json.loads(CACHE.read_text())
            # Never trust an empty cache (a failed resolve would otherwise
            # short-circuit forever) — force re-resolve if it has no pairs.
            if now - blob.get("_ts", 0) < RESOLVE_EVERY_S and blob.get("pairs"):
                return blob["pairs"]
        except Exception:
            pass
    if not UNIVERSE.exists():
        print(f"ERROR: {UNIVERSE} missing. Run build_dex_universe.py first.")
        return {}
    syms = pd.read_csv(UNIVERSE)["symbol"].astype(str).tolist()
    pairs = {}
    print(f"Resolving {len(syms)} DEX pair addresses (DexScreener)...")
    for i, sym in enumerate(syms):
        try:
            j = _get(f"{BASE}/search", params={"q": sym})
            cand = [p for p in j.get("pairs", []) if p.get("baseToken", {}).get("symbol", "").upper() == sym.upper()]
            if not cand:
                continue
            # highest-liquidity pair for that token
            cand.sort(key=lambda p: float(p.get("liquidity", {}).get("usd", 0) or 0), reverse=True)
            p = cand[0]
            pairs[sym.upper()] = (p["chainId"], p["pairAddress"])
        except Exception as e:
            print(f"  resolve {sym}: {e}")
        if (i + 1) % 25 == 0:
            print(f"  resolved {i+1}/{len(syms)}")
        time.sleep(PAUSE)
    CACHE.write_text(json.dumps({"_ts": now, "pairs": pairs}, indent=2))
    print(f"Resolved {len(pairs)} pairs -> {CACHE}")
    return pairs


def snapshot(pairs):
    """Poll every pair's live price once; append a 5m bar to its CSV."""
    ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    stamp = ts.strftime("%Y-%m-%d %H:%M:%S%z")
    added = skipped = 0
    for sym, (chain, addr) in pairs.items():
        try:
            j = _get(f"{BASE}/pairs/{chain}/{addr}")
            p = j.get("pairs", [{}])[0] if j.get("pairs") else {}
            price = float(p.get("priceUsd") or 0)
            vol = float(p.get("volume", {}).get("h24", 0) or 0)
            if price <= 0:
                skipped += 1
                continue
            bar = pd.DataFrame([{
                "ts": stamp, "open": price, "high": price,
                "low": price, "close": price, "volume": vol,
            }])
            out = DATA / f"{sym.upper()}_5m_dex_max.csv"
            if out.exists():
                bar.to_csv(out, mode="a", header=False, index=False)
            else:
                bar.to_csv(out, index=False)
            added += 1
        except Exception as e:
            skipped += 1
        time.sleep(PAUSE)
    print(f"[{stamp}] DEX 5m snapshot: +{added} bars, {skipped} skipped ({len(pairs)} pairs)")
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", action="store_true", help="run forever, snap every 5m")
    ap.add_argument("--force-resolve", action="store_true", help="re-resolve pair addresses now")
    args = ap.parse_args()
    pairs = resolve_pairs(force=args.force_resolve)
    if not pairs:
        return
    if args.loop:
        print("DEX forward collector: looping every 5m (Ctrl-C to stop)")
        while True:
            snapshot(pairs)
            time.sleep(300)
    else:
        snapshot(pairs)


if __name__ == "__main__":
    main()
