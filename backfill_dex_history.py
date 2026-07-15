#!/usr/bin/env python3
"""Backfill DEX token daily OHLCV from GeckoTerminal's FREE public API (no key).

GeckoTerminal (CoinGecko's DEX terminal) serves DEX pool OHLC history for free:
  GET /networks/{chain}/tokens/{contract}/pools  -> token's pools (highest vol first)
  GET /networks/{chain}/pools/{pool}/ohlcv/day?limit=N -> daily OHLC [ts,o,h,l,c,vol]
No API key. Needs a browser User-Agent (default urllib UA gets 403).

For each token (symbol, contract) in DEX_TOKENS:
  - resolve its top pool on `chain` (highest volume)
  - fetch daily OHLC (limit=1000 ~ 3yr; pagination via limit/before)
  - normalize to ts,open,high,low,close,volume -> dex_data/<SYM>_1d_max.csv
Resume: skip tokens already <= 2023-01-01. Rate-limit: 1s between calls.

Usage:
    python backfill_dex_history.py [--sleep 1.0] [--chain eth] [--limit 1000]
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import urllib.request

DEX = Path("dex_data")
DEX.mkdir(exist_ok=True)
API = "https://api.geckoterminal.com/api/v2"
UA = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}
# Curated major ERC20 DEX tokens (verified contract addresses). Grow as needed.
DEX_TOKENS = {
    "UNI":   "0x1f9840a85d5aF5bf1D1762F925BDADdC4201F984",
    "LINK":  "0x514910771AF9Ca656af840dff83e8264EcF986CA",
    "MATIC": "0x7D1AfA7B718fb893dB30A3aBc0Cfc608AaCfeBB0",
    "AAVE":  "0x7Fc66500c84A76Ad7e9c93437bFc5Ac33E2DDaE9",
    "SHIB":  "0x95aD61b0a150d79219dCF64E1E6Cc01f0B64C4cE",
    "PEPE":  "0x6982508145454Ce325dDbE47a25d4ec3d2311933",
    "WETH":  "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",
    "DAI":   "0x6B175474E89094C44Da98b954EedeAC495271d0F",
    "USDC":  "0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48",
    "WBTC":  "0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599",
    "ARB":   "0x912CE59144191C1204E64559FE8253a0e49E6548",
    "OP":    "0x4200000000000000000000000000000000000042",
    "MKR":   "0x9f8F72aA9304c8B593d555F12eF6589cC3A579A2",
    "LDO":   "0x5A98FcBEA516Cf06857215779Fd812CA3beF1B32",
    "CRV":   "0xD533a949740bb3306d119CC777fa900bA034cd52",
    "GRT":   "0xc944E90C64B2c07662A292be6244BDF05Cda44a7",
    "ENS":   "0x186fA132d286004c53D42f351B8a4a8b3fC6bD45",
    "COMP":  "0xc00e94Cb662C3520282E6f5717214004A7f26888",
    "SNX":   "0xC011a73ee8576Fb46F5E1c5751cA2979411eC121",
    "SUSHI": "0x6B3595068838A9727E1c270bD3E9d8E6c3aC3719",
    "1INCH": "0x111111111117dC0aa78b770fA6A738034120C302",
    "INJ":   "0xe28b3B32B6c345A34FfD01b1A30B23075236aB1b",
    "FET":   "0xaea46A60368A7BdA2Dc983F13072d32Bb0DA14e3",
    "RNDR":  "0x6B2452aD3fF6e106F8439570b7e186c7e41A4E1d",
    "LRC":   "0xBBbbCA6A901c926F240b89EacB641d8Aec7AEafD",
    "BAL":   "0xba100000625a3754423978a60c9317c58a424e3D",
    "ALCX":  "0xdbdb4d16eda451d0503b854cf79d55697f90c8df",
}
DEEP_ENOUGH = pd.Timestamp("2023-01-01")


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
                wait = 5 * (2 ** i)  # 5,10,20,40,80s backoff
                print(f"    HTTP {e.code} -> backoff {wait}s")
                time.sleep(wait)
                continue
            raise
        except Exception as e:
            last = e
            time.sleep(5)
    raise last if last else RuntimeError("get failed")


def resolve_top_pool(chain, contract):
    url = f"{API}/networks/{chain}/tokens/{contract}/pools?page=1"
    try:
        d = _get(url)
        pools = d.get("data", [])
        if not pools:
            return None
        return pools[0]["attributes"]["address"]
    except Exception:
        return None


def fetch_ohlcv(chain, pool, limit):
    url = f"{API}/networks/{chain}/pools/{pool}/ohlcv/day?limit={limit}"
    d = _get(url)
    rows = d.get("data", {}).get("attributes", {}).get("ohlcv_list", [])
    if not rows:
        return None
    # GeckoTerminal: [timestamp, open, high, low, close, volume]
    out = [{"ts": pd.to_datetime(r[0], unit="s").strftime("%Y-%m-%d"),
            "open": r[1], "high": r[2], "low": r[3], "close": r[4], "volume": r[5]}
           for r in rows]
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--chain", default="eth")
    ap.add_argument("--limit", type=int, default=1000, help="daily bars per token (~3yr)")
    args = ap.parse_args()
    print(f"DEX history backfill (GeckoTerminal, free): {len(DEX_TOKENS)} tokens -> dex_data/")
    done = skipped = 0
    for sym, contract in DEX_TOKENS.items():
        out = DEX / f"{sym}_1d_max.csv"
        if out.exists():
            try:
                if pd.Timestamp(pd.read_csv(out)["ts"].min()) <= DEEP_ENOUGH:
                    skipped += 1
                    continue
            except Exception:
                pass
        pool = resolve_top_pool(args.chain, contract)
        if not pool:
            print(f"  {sym}: no DEX pool found")
            time.sleep(args.sleep)
            continue
        df = fetch_ohlcv(args.chain, pool, args.limit)
        if df is None or len(df) == 0:
            print(f"  {sym}: no OHLC")
            time.sleep(args.sleep)
            continue
        df.to_csv(out, index=False)
        done += 1
        print(f"  {sym}: {len(df)} bars -> {out.name} earliest={df['ts'].min()}")
        time.sleep(args.sleep)
    print(f"\nDEX backfill complete: {done} tokens, {skipped} already deep (skipped).")


if __name__ == "__main__":
    main()
