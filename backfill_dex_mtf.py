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

MEMORY SAFETY (added): 
  - Resolved pools are cached (in-memory + resolved_pool_cache.json) so a token
    is never re-resolved on a re-run.
  - gc.collect() runs after every token so no DataFrame/object accumulates.
  - A hard RSS cap (--mem-limit-mb, default 1536) ABORTS the run well before it
    could threaten the machine. This makes an OOM-kill-by-this-script impossible.

Usage:
    python backfill_dex_mtf.py                 # all TFs, all universe tokens
    python backfill_dex_mtf.py --tf 1h --limit 500
"""
import argparse
import gc
import json
import time
from pathlib import Path

import pandas as pd
import urllib.request
import urllib.parse
import json as _json  # noqa: F401 (kept for compatibility)

from mem_guard import guard as _mem_guard

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
CACHE_FILE = ROOT / "resolved_pool_cache.json"
DEFAULT_MEM_LIMIT_MB = 1536


# ---- resolved-pool cache (persisted so re-runs never re-resolve) ----
_resolved_cache: dict = {}


def _load_cache() -> None:
    if CACHE_FILE.exists():
        try:
            _resolved_cache.update(json.loads(CACHE_FILE.read_text()))
        except Exception:
            pass


def _save_cache() -> None:
    try:
        CACHE_FILE.write_text(json.dumps(_resolved_cache))
    except Exception:
        pass


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


def _resolve_pool_real(sym):
    """Top GeckoTerminal pool for a symbol. Returns (gecko_net, pool_address) or None."""
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


def resolve_pool(sym):
    """Cached wrapper: never re-resolves a token we already resolved this run
    or in a previous run (persisted cache)."""
    s = sym.upper()
    if s in _resolved_cache:
        v = _resolved_cache[s]
        return tuple(v) if v else None
    res = _resolve_pool_real(s)
    _resolved_cache[s] = list(res) if res else None
    _save_cache()
    return res


def fetch_ohlcv(net, pool, tf, agg, limit):
    url = f"{API}/networks/{net}/pools/{pool}/ohlcv/{tf}?limit={limit}&aggregate={agg}"
    d = _get(url)
    rows = d.get("data", {}).get("attributes", {}).get("ohlcv_list", [])
    if not rows:
        return None
    # 1d -> one bar per date (date-only ts). 1h/4h -> full timestamp, else
    # every bar in a day collapses to the same date string and overwrites.
    fmt = "%Y-%m-%d" if tf == "day" else "%Y-%m-%d %H:%M:%S+0000"
    return pd.DataFrame(
        [{"ts": pd.to_datetime(r[0], unit="s").strftime(fmt),
           "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]}
          for r in rows]
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", choices=list(TFS), default=None, help="single TF (default: all)")
    ap.add_argument("--limit", type=int, default=1000, help="bars per TF (~3yr at 1d)")
    ap.add_argument("--sleep", type=float, default=SLEEP)
    ap.add_argument("--universe", default=str(UNIVERSE))
    ap.add_argument("--mem-limit-mb", type=int, default=DEFAULT_MEM_LIMIT_MB,
                    help="hard RSS cap; run aborts safely above this (prevents OOM)")
    args = ap.parse_args()

    _load_cache()
    _mem_guard(args.mem_limit_mb)  # trip early if already over cap at start

    if not Path(args.universe).exists():
        print(f"ERROR: {args.universe} missing. Run build_dex_universe.py first.")
        return
    syms = pd.read_csv(args.universe)["symbol"].astype(str).tolist()
    tfs = [args.tf] if args.tf else list(TFS)
    print(f"DEX MTF backfill (GeckoTerminal free): {len(syms)} tokens -> data/ "
          f"TFs={tfs} (mem cap={args.mem_limit_mb}MB)")
    for sym in syms:
        sym = sym.upper()
        _mem_guard(args.mem_limit_mb)  # checked before every token
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
        gc.collect()  # reclaim this token's DataFrame/objects immediately
        time.sleep(args.sleep)
    gc.collect()
    print("DEX MTF backfill complete.")


if __name__ == "__main__":
    main()
