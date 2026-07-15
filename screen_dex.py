#!/usr/bin/env python3
"""Free DEX screener: find liquid low-cap alts tradable on DEXes.

No API key. Two free sources:
  1. Token address seed list: CoinGecko's Uniswap tokenlist (MIT, ~4800 ETH
     tokens) -- https://tokens.coingecko.com/uniswap/all.json
  2. Live DEX metrics per token: dex-screener API (free, no key) --
     /latest/dex/tokens/{address} -> pairs with price/liquidity/volume/age.

For each token we aggregate its DEX pairs into one screen row (best liquidity,
sum volume, oldest pair age, chains/dexes). Then filter for tradeable low-cap
DEX alts (liquid + some volume, exclude stables/wrapped). Output a screen CSV
compatible with the pipeline (backtest_output/screen_dex_<date>.csv).

This is DEX SCREENING (what's tradeable, how liquid, what chain) -- the DEX
counterpart to screen_liquidity_idiosyncratic.py. It does NOT give historical
OHLCV (free deep DEX price history needs a key: The Graph free tier / CoinGecko
paid) -- that gap is noted, not solved here.

Usage:
    python screen_dex.py [--limit N] [--sleep 0.15]
"""
import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import urllib.request

TOKENLIST_URL = "https://tokens.coingecko.com/uniswap/all.json"
DEX_API = "https://api.dexscreener.com/latest/dex/tokens/"
OUT = Path("backtest_output")
# filter thresholds (liquid low-cap DEX alt)
MIN_LIQUIDITY = 50_000      # USD
MIN_VOLUME24H = 10_000      # USD
EXCLUDE = {"USDT", "USDC", "DAI", "WETH", "WBNB", "WMATIC", "ETH", "WBTC",
           "stETH", "WEETH", "WSTETH", "FRAX", "LUSD", "USDE", "SUSDE"}


def load_tokens(limit):
    import json
    req = urllib.request.Request(TOKENLIST_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        toks = json.loads(r.read())["tokens"]
    eth = [t for t in toks if t.get("chainId") == 1]
    eth = [t for t in eth if str(t.get("symbol", "")).upper() not in EXCLUDE]
    if limit:
        eth = eth[:limit]
    return eth


def screen_token(addr, sleep):
    url = f"{DEX_API}{addr}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as r:
            import json
            pairs = json.loads(r.read()).get("pairs") or []
    except Exception:
        time.sleep(sleep)
        return None
    if not pairs:
        return None
    sym = pairs[0]["baseToken"].get("symbol", "")
    name = pairs[0]["baseToken"].get("name", "")
    liq = max((p.get("liquidity") or {}).get("usd", 0) or 0 for p in pairs)
    vol = sum((p.get("volume") or {}).get("h24", 0) or 0 for p in pairs)
    if liq < MIN_LIQUIDITY or vol < MIN_VOLUME24H:
        return None
    ages = [p.get("pairCreatedAt", 0) for p in pairs if p.get("pairCreatedAt")]
    oldest = min(ages) if ages else 0
    chains = sorted({p.get("chainId") for p in pairs})
    dexes = sorted({p.get("dexId") for p in pairs})
    price = pairs[0].get("priceUsd")
    return {"address": addr, "symbol": sym, "name": name, "priceUsd": price,
            "liquidityUsd": liq, "volume24h": vol, "oldestPairMs": oldest,
            "chains": ",".join(chains), "dexes": ",".join(dexes),
            "pairCount": len(pairs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="max tokens to screen (0=all)")
    ap.add_argument("--sleep", type=float, default=0.15)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    toks = load_tokens(args.limit)
    print(f"DEX screen: {len(toks)} ETH tokens -> backtest_output/screen_dex_*.csv")
    rows = []
    for i, t in enumerate(toks, 1):
        row = screen_token(t["address"], args.sleep)
        if row:
            rows.append(row)
            print(f"[{i}/{len(toks)}] {row['symbol']}: liq=${row['liquidityUsd']:,.0f} vol=${row['volume24h']:,.0f}")
        time.sleep(args.sleep)
    df = pd.DataFrame(rows)
    if len(df):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
        out = OUT / f"screen_dex_{stamp}.csv"
        df.to_csv(out, index=False)
        print(f"\nWrote {len(df)} liquid DEX tokens -> {out}")
    else:
        print("\nNo liquid DEX tokens found (raise thresholds or check connectivity).")


if __name__ == "__main__":
    main()
