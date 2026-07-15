#!/usr/bin/env python3
"""Build a LIVE DEX universe from GeckoTerminal's free pool rankings (no key).

GeckoTerminal ranks DEX pools by 24h volume per network. We paginate the top
pools across major networks, parse each pool's "BASE / QUOTE" name, keep the
non-stable token side, and DEDUPE to the highest-volume pool per token. Result
is a real, data-driven DEX universe (low-caps included -- they rank by volume)
-- not a hand-picked list. Written to dex_universe.csv for the backfill + the
forward collector to consume.

Re-run daily to refresh the universe (ALL THE DATA ALL THE TIME).

Usage:
    python build_dex_universe.py [--top-per-network 500] [--networks eth,base,bsc,arbitrum,polygon,solana]
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import backfill_dex_history as b  # reuse _get + UA (retry/backoff)

OUT = Path("dex_universe.csv")
# Stable / wrapped assets to skip as the "token" side (compared case-insensitively)
STABLES = {"WETH", "WBNB", "ETH", "BNB", "USDC", "USDT", "DAI", "WBTC", "BTC",
           "BUSD", "TUSD", "USDE", "FDUSD", "PYUSD", "SUSD", "LUSD", "FRAX",
           "WEETH", "WSTETH", "STETH", "GHO", "CLE", "USDG", "EURC",
           "USDD", "USDP", "USDN", "GUSD", "USDB", "USD1", "USDL",
           "SFRXETH", "CBBTC", "CBETH", "RETH", "METH", "CRVUSD", "USDS",
           "USD0", "FXD", "USDX", "DOLA", "MIM", "USDP", "AGEUR", "EURS",
           "XAUT", "PAXG", "GOLD", "USDL", "YIELDBOSTON", "WSUPER", "SUPER"}


def parse_pair(name):
    """'BLINK / WETH 0.3%' -> ('BLINK', 'WETH')."""
    base = name.split(" / ")[0].strip()
    quote = name.split(" / ")[1].split(" ")[0].strip() if " / " in name else ""
    return base, quote


def build(networks, top_per_network, sleep):
    rows = []
    seen = {}  # (network, token) -> vol24h (keep max)
    for net in networks:
        collected = 0
        page = 1
        while collected < top_per_network:
            url = f"{b.API}/networks/{net}/pools?page={page}"
            try:
                d = b._get(url)
            except Exception as e:
                print(f"  {net} page{page} ERR: {e!r}")
                break
            pools = d.get("data", [])
            if not pools:
                break
            for p in pools:
                a = p["attributes"]
                name = a.get("name", "")
                base, quote = parse_pair(name)
                if not base:
                    continue
                # token side = the non-stable; skip if BOTH sides are stables.
                # compare case-insensitively (GeckoTerminal returns e.g. "cbBTC").
                base_st = base.upper() in STABLES
                quote_st = quote.upper() in STABLES
                if base_st and quote_st:
                    continue
                token = quote if base_st else base
                if not token or token.upper() in STABLES:
                    continue
                vol = (a.get("volume_usd") or {}).get("h24") or 0.0
                key = (net, token)
                if key not in seen or vol > seen[key]["vol24h"]:
                    seen[key] = {"symbol": token, "network": net,
                                 "pool_address": a.get("address"),
                                 "quote": quote, "vol24h": vol}
                collected += 1
            page += 1
            time.sleep(sleep)
            if page > 50:  # safety: ~25000 pools max
                break
        print(f"  {net}: {len([k for k in seen if k[0]==net])} tokens so far")
    df = pd.DataFrame(list(seen.values()))
    if len(df):
        df = df.sort_values("vol24h", ascending=False)
    else:
        df = pd.DataFrame(columns=["symbol", "network", "pool_address", "quote", "vol24h"])
    df.to_csv(OUT, index=False)
    print(f"\nDEX universe: {len(df)} unique tokens -> {OUT}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-per-network", type=int, default=500)
    ap.add_argument("--networks", default="eth,base,bsc,arbitrum,polygon,solana")
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()
    nets = [n.strip() for n in args.networks.split(",") if n.strip()]
    print(f"Building DEX universe from GeckoTerminal: networks={nets} top={args.top_per_network}/net")
    build(nets, args.top_per_network, args.sleep)


if __name__ == "__main__":
    main()
