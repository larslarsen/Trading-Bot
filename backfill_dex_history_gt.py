#!/usr/bin/env python3
"""
GeckoTerminal historical DEX pool OHLCV backfill (FREE, NO KEY).

We already collect LIVE DEX micro prices (dex_micro_poller.py) but NOT
historical candles. GeckoTerminal serves per-pool OHLCV history:
  GET /api/v2/networks/{network}/pools/{pool_address}/ohlcv/{timeframe}
  -> data.attributes.ohlcv_list = [[ts, o, h, l, c, volume], ...]
Paginated via ?page=N (oldest first? newest first? we walk pages backward).

Pipeline per token:
  1. Resolve token -> highest-liquidity pool via DexScreener (reuse logic from
     dex_micro_poller.resolve_address + grab top pair's pool address + GT net).
  2. Backfill OHLCV/day (and optionally /hour) for that pool.
  3. Write data/dex_history/<TOKEN>_1d_max.csv (ts-indexed OHLCV).

Resumable: skips tokens already done (file exists with >= N rows).

Usage:
  python backfill_dex_history_gt.py [--tokens TOK1,TOK2,...] [--limit N]
                                   [--tf day] [--net-pref eth,bsc,solana]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests

REPO = Path(__file__).parent
DATA = REPO / "data" / "dex_history"
DATA.mkdir(parents=True, exist_ok=True)
UA = {"User-Agent": "Mozilla/5.0 (research backfill)"}
GT_API = "https://api.geckoterminal.com/api/v2"
DEX_API = "https://api.dexscreener.com/latest/dex"


def fetch_json(url, timeout=20, tries=3):
    for i in range(tries):
        try:
            r = requests.get(url, headers=UA, timeout=timeout)
            if r.status_code == 200:
                return r.json(), None
            if r.status_code == 429:
                time.sleep(min(2 ** i * 3, 30)); continue
            return None, f"HTTP {r.status_code}"
        except Exception as e:
            time.sleep(min(2 ** i * 2, 15)); continue
    return None, "exhausted retries"


def resolve_top_pool(tok):
    """Resolve token -> (gt_network, pool_address, dex) for highest-LIQ pool.
    Uses DexScreener token search; returns top pair by liquidity."""
    # token may be a contract address or a ticker; search both
    url = f"{DEX_API}/search?q={tok}"
    d, err = fetch_json(url)
    if err or not d:
        return None, err or "no response"
    pairs = d.get("pairs") or []
    if not pairs:
        return None, "no pairs"
    # network map: dexscreener chain -> geckoterminal network slug
    NETMAP = {
        "ethereum": "eth", "bsc": "bsc", "solana": "solana", "base": "base",
        "arbitrum": "arbitrum", "polygon": "polygon", "avax": "avax",
        "optimism": "optimism", "fantom": "fantom", "cronos": "cronos",
    }
    best = None
    for p in pairs:
        chain = (p.get("chainId") or "").lower()
        gtnet = NETMAP.get(chain)
        if not gtnet:
            continue
        liq = float(p.get("liquidity", {}).get("usd") or 0)
        pair_addr = p.get("pairAddress")
        if not pair_addr:
            continue
        if best is None or liq > best[3]:
            best = (gtnet, pair_addr, p.get("dexId"), liq)
    if not best:
        return None, "no mapped network/pool"
    return (best[0], best[1], best[2]), None


def gt_ohlcv(net, pool, tf="day", page=1, limit=1000, aggregate=1):
    url = f"{GT_API}/networks/{net}/pools/{pool}/ohlcv/{tf}?limit={limit}&page={page}&aggregate={aggregate}"
    d, err = fetch_json(url)
    if err:
        return None, err
    if "data" not in d:
        return None, "no data"
    ohlc = d["data"].get("attributes", {}).get("ohlcv_list") or []
    return ohlc, None


def backfill_token(tok, tf="day", max_pages=200):
    res, err = resolve_top_pool(tok)
    if err:
        return 0, f"resolve: {err}"
    net, pool, dex = res
    rows = []
    seen = set()
    page = 1
    while page <= max_pages:
        ohlc, e = gt_ohlcv(net, pool, tf, page)
        if e:
            return len(rows), f"page{page}: {e}"
        if not ohlc:
            break
        for r in ohlc:
            # r = [ts, o, h, l, c, v]
            t = int(r[0])
            if t in seen:
                continue
            seen.add(t)
            rows.append([t, float(r[1]), float(r[2]), float(r[3]),
                         float(r[4]), float(r[5])])
        # GeckoTerminal returns newest-first per page; if page returned < limit, done
        if len(ohlc) < 1000:
            break
        page += 1
        time.sleep(0.4)
    if not rows:
        return 0, "no rows"
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)
    df = df.sort_values("ts").set_index("ts")
    tgt = DATA / f"{tok}_{tf}_max.csv"
    if tgt.exists():
        old = pd.read_csv(tgt, parse_dates=["ts"]).set_index("ts")
        df = pd.concat([old, df]).sort_index()
        df = df[~df.index.duplicated(keep="last")]
    df.to_csv(tgt)
    return len(df), f"{net}/{dex} pool={pool[:10]}.."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tf", default="day")
    ap.add_argument("--sleep", type=float, default=3.0,
                    help="seconds between tokens (throttle so the 1m daemon keeps API budget)")
    args = ap.parse_args()
    if args.tokens:
        toks = [t.strip() for t in args.tokens.split(",") if t.strip()]
    else:
        # default: all dex_data token stems
        toks = [p.stem for p in (REPO / "dex_data").glob("*.csv")]
    if args.limit:
        toks = toks[:args.limit]
    print(f"dex history backfill: {len(toks)} tokens, tf={args.tf}", flush=True)
    done = 0
    for tok in toks:
        tgt = DATA / f"{tok}_{args.tf}_max.csv"
        if tgt.exists():
            print(f"  skip {tok} (exists)", flush=True); done += 1; continue
        try:
            n, info = backfill_token(tok, args.tf)
            print(f"  {tok}: {n} rows ({info})", flush=True)
            if n: done += 1
        except Exception as ex:
            print(f"  {tok}: EXC {str(ex)[:80]}", flush=True)
        time.sleep(args.sleep)
    print(f"dex history backfill done. {done}/{len(toks)} succeeded.", flush=True)


if __name__ == "__main__":
    main()
