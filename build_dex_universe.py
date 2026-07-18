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
    # --- seed tokenized-stock / equities tokens (Robinhood-on-Arbitrum etc.)
    # These rarely surface in volume-ranked pool scans but are high-signal
    # retail-driven markets. Force-include via the same resolver the micro
    # poller uses, so each lands on its real chain (robinhood/eth/bsc/...).
    SEED_EQUITIES = ["NVDA", "AAPL", "TSLA", "GOOGL", "MSFT", "AMZN", "META",
                     "HOOD", "COIN", "SPY", "QQQ", "DIA", "NVDL", "TSLL",
                     "GOOGL", "NFLX", "AMD", "INTC", "BABA", "ORCL"]
    try:
        import dex_micro_poller as mpol
        for tok in SEED_EQUITIES:
            try:
                res = mpol.resolve_address(tok)
                if not res:
                    continue
                chain, addr = res
                # pick a representative pool-less entry; micro poller resolves per-token
                seen.setdefault((chain, tok), {"symbol": tok, "network": chain,
                                                "pool_address": addr, "quote": "USDC",
                                                "vol24h": 0.0})
            except Exception:
                pass
        print(f"  seeded {len(SEED_EQUITIES)} equities tokens")
    except Exception as e:
        print(f"  equities seed ERR: {e!r}")
    df = pd.DataFrame(list(seen.values()))
    if len(df):
        df = df.sort_values("vol24h", ascending=False)
    else:
        df = pd.DataFrame(columns=["symbol", "network", "pool_address", "quote", "vol24h"])
    df.to_csv(OUT, index=False)
    print(f"\nDEX universe: {len(df)} unique tokens -> {OUT}")
    return df


def discover_and_merge(top_n=80, existing=None):
    """Expand the universe beyond the volume-ranked 200 by merging in TRENDING
    and NEW-PAIR DEX tokens discovered from DexScreener + GeckoTerminal + GMGN
    (best-effort). Each discovered token is resolved to its real chain/pool via
    the same resolver the equities seed uses, then appended to dex_universe.csv
    (deduped on (network, symbol)). Best-effort: any failure is skipped, the
    weekly rebuild never breaks.

    Returns the number of NEW tokens added.
    """
    try:
        import data_quality as dq
        import dex_micro_poller as mpol
    except Exception as e:
        print(f"  [discover] import ERR: {e!r}")
        return 0
    try:
        toks = dq.discover_dex_tokens() or []
    except Exception as e:
        print(f"  [discover] discovery ERR: {e!r}")
        return 0
    if not toks:
        print("  [discover] no tokens discovered (sources gated/offline)")
        return 0
    existing = set(existing or [])
    added = 0
    new_rows = []
    for t in toks:
        sym = (t.get("symbol") or "").upper()
        if not sym or sym in existing or sym in STABLES:
            continue
        try:
            res = mpol.resolve_address(sym)
            if not res:
                continue
            chain, addr = res
            new_rows.append({"symbol": sym, "network": chain,
                             "pool_address": addr, "quote": "USDC", "vol24h": 0.0})
            existing.add(sym)
            added += 1
            if added >= top_n:
                break
        except Exception:
            continue
    if new_rows:
        df_new = pd.DataFrame(new_rows)
        if OUT.exists():
            df_old = pd.read_csv(OUT)
            df = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df = df_new
        # keep first occurrence of each (network, symbol)
        df = df.drop_duplicates(subset=["network", "symbol"], keep="first")
        df.to_csv(OUT, index=False)
        print(f"  [discover] +{added} trending/new tokens merged -> {OUT} "
              f"(now {len(df)} total)")
    else:
        print("  [discover] no new tokens to add")
    return added


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top-per-network", type=int, default=500)
    ap.add_argument("--networks", default="eth,base,bsc,arbitrum,polygon,solana,robinhood")
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--discover", type=int, default=0,
                    help="after building, merge up to N trending/new DEX tokens "
                         "from DexScreener+GeckoTerminal+GMGN")
    args = ap.parse_args()
    nets = [n.strip() for n in args.networks.split(",") if n.strip()]
    print(f"Building DEX universe from GeckoTerminal: networks={nets} top={args.top_per_network}/net")
    build(nets, args.top_per_network, args.sleep)
    if args.discover:
        discover_and_merge(top_n=args.discover)


if __name__ == "__main__":
    main()
