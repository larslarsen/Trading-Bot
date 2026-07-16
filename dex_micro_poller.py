#!/usr/bin/env python3
"""
DEX microstructure breadth poller (DexScreener free API, no key).

For every token in dex_data/*.csv (the DEX universe we already hold), poll
DexScreener's token endpoint for live per-token liquidity + volume + price
change, and append a timestamped row to data/dex_micro/<TOKEN>.csv. This
builds a TIME-SERIES of DEX-wide breadth/flow that later becomes cross-venue
features for the CEX models (total DEX vol, % gainers, idiosyncratic vol).

Rate-limited (gentle). Resumable: appends new polls, dedups by timestamp.

Usage:
  python dex_micro_poller.py            # poll all 426 tokens once
  python dex_micro_poller.py --loop 900 # poll every 900s (background daemon)
"""
import argparse
import time
import requests
import pandas as pd
from pathlib import Path

REPO = Path(__file__).parent
DEX_DIR = REPO / "dex_data"  # existing DEX price-bar universe (token stems = addresses/tickers)
OUT = REPO / "data" / "dex_micro"
OUT.mkdir(parents=True, exist_ok=True)
API = "https://api.dexscreener.com/latest/dex/tokens"
SLEEP = 0.4
MIN_INTERVAL = 60  # floor between polls per token


def tokens():
    toks = []
    for p in sorted(DEX_DIR.glob("*.csv")):
        # filenames are like OMI_1d_max.csv -> token stem is OMI
        stem = p.stem
        for suf in ("_1d_max", "_1d", "_1h_max", "_1h", "_5m_max", "_5m", "_4h_max", "_4h"):
            if stem.endswith(suf):
                stem = stem[: -len(suf)]
                break
        toks.append(stem)
    return toks


def resolve_address(tok):
    """Resolve a ticker to a contract address via DexScreener search.
    Prefer credible chains (ethereum/polygon/bsc/base) over copycat chains
    (solana/etc. often host symbol-collision scam pairs). Returns
    (chainId, address) for the highest-liquidity credible match, else any."""
    try:
        r = requests.get(f"{API.rsplit('/tokens',1)[0]}/search", params={"q": tok}, timeout=20)
    except Exception:
        return None
    if r.status_code != 200:
        return None
    pairs = (r.json() or {}).get("pairs") or []
    PREF = {"ethereum": 0, "polygon": 1, "bsc": 2, "base": 3, "arbitrum": 4}
    norm = lambda s: (s or "").upper().lstrip("$").strip()
    q = norm(tok)
    exact = None      # best credible-chain EXACT-symbol match
    fallback = None   # best credible-chain pair by liquidity (any symbol)
    for p in pairs:
        base = (p.get("baseToken") or {}).get("symbol", "")
        chain = p.get("chainId", "")
        liq = float(p.get("liquidity", {}).get("usd") or 0)
        pref = PREF.get(chain, 9)
        if norm(base) == q:
            score = (pref, -liq)
            if exact is None or score < exact[0]:
                exact = (score, (chain, p.get("baseToken", {}).get("address")))
        if fallback is None or (pref, -liq) < fallback[0]:
            fallback = ((pref, -liq), (chain, p.get("baseToken", {}).get("address")))
    # exact symbol match on a credible chain is authoritative; otherwise
    # best-effort capture the top credible-chain pair so breadth isn't dropped
    return (exact or fallback)[1] if (exact or fallback) else None


def poll_token(tok):
    addr = resolve_address(tok)
    if addr is None:
        return None, "no addr"
    chainId, address = addr
    try:
        r = requests.get(f"{API}/{address}", timeout=20)
    except Exception as e:
        return None, f"req {e}"
    if r.status_code != 200:
        return None, f"HTTP {r.status_code}"
    try:
        data = r.json()
    except Exception:
        return None, "bad json"
    pairs = data.get("pairs") or []
    if not pairs:
        return None, "no pairs"
    vol = sum(float(p.get("volume", {}).get("h24") or 0) for p in pairs)
    liq = sum(float(p.get("liquidity", {}).get("usd") or 0) for p in pairs)
    chg = [float(p.get("priceChange", {}).get("h24") or 0) for p in pairs if p.get("priceChange", {}).get("h24") is not None]
    chg_med = sorted(chg)[len(chg)//2] if chg else 0.0
    fdv = max((float(p.get("fdv") or 0) for p in pairs), default=0.0)
    return {"volume_h24": vol, "liquidity_usd": liq, "price_chg_h24_med": chg_med, "fdv": fdv,
            "chain": chainId, "address": address}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loop", type=int, default=0, help="seconds between polls; 0=once")
    ap.add_argument("--limit", type=int, default=0, help="max tokens this run")
    args = ap.parse_args()
    toks = tokens()
    if args.limit:
        toks = toks[:args.limit]
    print(f"DEX micro poller: {len(toks)} tokens -> {OUT}")
    while True:
        done = 0
        for tok in toks:
            path = OUT / f"{tok}.csv"
            # skip if polled very recently
            if path.exists():
                try:
                    last = pd.read_csv(path, usecols=["ts"], parse_dates=["ts"]).iloc[-1]["ts"]
                    if (pd.Timestamp.now("UTC") - pd.to_datetime(last, utc=True)).total_seconds() < MIN_INTERVAL:
                        continue
                except Exception:
                    pass
            rec, err = poll_token(tok)
            if rec is None:
                if err and "no pairs" not in err:
                    print(f"  {tok}: {err}")
                time.sleep(SLEEP)
                continue
            row = {"ts": pd.Timestamp.now("UTC"), **rec}
            df = pd.DataFrame([row])
            if path.exists():
                old = pd.read_csv(path, parse_dates=["ts"])
                df = pd.concat([old, df]).drop_duplicates(subset=["ts"]).sort_values("ts")
            df.to_csv(path, index=False)
            done += 1
            if done % 50 == 0:
                print(f"  polled {done}/{len(toks)}")
            time.sleep(SLEEP)
        print(f"cycle done: {done} tokens updated at {pd.Timestamp.now('UTC')}")
        if args.loop <= 0:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
