#!/usr/bin/env python3
"""
DEX MTF backfill (1h / 4h / 1d) — GeckoTerminal POOL OHLCV, FREE, no key.

Unifies DEX with CEX: writes data/<SYM>_{1h,4h,1d}_dex_max.csv, the SAME store
+ naming the CEX collector uses. DEX was previously 1d-only in dex_data/ — this
adds 1h/4h and moves everything into data/ so CEX + DEX share one place, one
method per timeframe (5m via dex_forward_collector, 1h/4h/1d here).

For each symbol in dex_universe.csv we resolve its top GeckoTerminal pool
(network + pool_address) on the fly, then fetch the three timeframes. Self-
contained: does not depend on extra columns in dex_universe.csv.

Rate discipline: 1s sleep between fetches (backfill is occasional, not hot).
Usage:
    python backfill_dex_mtf.py                 # all TFs, all universe tokens
    python backfill_dex_mtf.py --tf 1h --limit 500
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import urllib.request
import urllib.parse
import json

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
UNIVERSE = ROOT / "dex_universe.csv"
API = "https://api.geckoterminal.com/api/v2"
BASE = "https://api.dexscreener.com/latest/dex"
UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
DEEP_ENOUGH = pd.Timestamp("2023-01-01")
TFS = {"1d": ("day", 1), "1h": ("hour", 1), "4h": ("hour", 4)}
SLEEP = 1.0


def _get(url, tries=5):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                wait = 5 * (2 ** i)
                print(f"    HTTP {e.code} -> backoff {wait}s")
                time.sleep(wait)
                continue
            raise
        except Exception as e:
            last = e
            time.sleep(5)
    raise last if last else RuntimeError("get failed")


import requests as _req

# DexScreener chainId -> GeckoTerminal network id
NET_MAP = {
    "ethereum": "eth", "polygon": "polygon_pos", "bsc": "bsc",
    "base": "base", "arbitrum": "arbitrum", "avalanche": "avax",
    "fantom": "fantom", "solana": "solana",
}


def resolve_pool(sym):
    """Top GeckoTerminal pool for a symbol. Uses DexScreener (free) to get the
    contract + chain, maps to Gecko's network id, then fetches its top pool.
    Returns (gecko_net, pool_address) or None."""
    try:
        j = _req.get(f"{BASE}/search", params={"q": sym}, timeout=15).json()
    except Exception:
        return None
    cand = [p for p in j.get("pairs", [])
            if p.get("baseToken", {}).get("symbol", "").upper() == sym.upper()]
    if not cand:
        return None
    cand.sort(key=lambda p: float(p.get("liquidity", {}).get("usd", 0) or 0),
              reverse=True)
    p = cand[0]
    chain = p.get("chainId")
    addr = p.get("baseToken", {}).get("address")
    gnet = NET_MAP.get(chain)
    if not gnet or not addr:
        return None
    try:
        tp = _get(f"{API}/networks/{gnet}/tokens/{addr}/pools?page=1")
        tpools = tp.get("data", [])
        if not tpools:
            return None
        return gnet, tpools[0]["attributes"]["address"]
    except Exception:
        return None


def fetch_ohlcv(net, pool, tf, agg, limit):
    url = f"{API}/networks/{net}/pools/{pool}/ohlcv/{tf}?limit={limit}&aggregate={agg}"
    d = _get(url)
    rows = d.get("data", {}).get("attributes", {}).get("ohlcv_list", [])
    if not rows:
        return None
    return pd.DataFrame(
        [{"ts": pd.to_datetime(r[0], unit="s").strftime("%Y-%m-%d"),
           "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]}
          for r in rows]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", choices=list(TFS), default=None, help="single TF (default: all)")
    ap.add_argument("--limit", type=int, default=1000, help="bars per TF (~3yr at 1d)")
    ap.add_argument("--sleep", type=float, default=SLEEP)
    ap.add_argument("--universe", default=str(UNIVERSE))
    args = ap.parse_args()
    if not Path(args.universe).exists():
        print(f"ERROR: {args.universe} missing. Run build_dex_universe.py first.")
        return
    syms = pd.read_csv(args.universe)["symbol"].astype(str).tolist()
    tfs = [args.tf] if args.tf else list(TFS)
    print(f"DEX MTF backfill (GeckoTerminal free): {len(syms)} tokens -> data/ "
          f"TFs={tfs}")
    for sym in syms:
        sym = sym.upper()
        resolved = resolve_pool(sym)
        if not resolved:
            print(f"  {sym}: no Gecko pool")
            time.sleep(args.sleep)
            continue
        net, pool = resolved
        for tf in tfs:
            kind, agg = TFS[tf]
            out = DATA / f"{sym}_{tf}_dex_max.csv"
            if out.exists():
                try:
                    if pd.Timestamp(pd.read_csv(out)["ts"].min()) <= DEEP_ENOUGH:
                        continue
                except Exception:
                    pass
            df = fetch_ohlcv(net, pool, kind, agg, args.limit)
            if df is None or len(df) == 0:
                print(f"  {sym} {tf}: no OHLC")
                time.sleep(args.sleep)
                continue
            df.to_csv(out, index=False)
            print(f"  {sym} {tf}: {len(df)} bars -> {out.name}")
            time.sleep(args.sleep)
        time.sleep(args.sleep)
    print("DEX MTF backfill complete.")


if __name__ == "__main__":
    main()
