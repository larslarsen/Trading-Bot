#!/usr/bin/env python3
"""Backfill point-in-time (PIT) universe snapshots for survivorship-free backtests.

For each weekly date from DATA_START to today, fetch CoinGecko's top coins
*as of that date* (via /coins/markets?date=DD-MM-YYYY) and screen them into a
dated screen_liqu_idio_<YYYYMMDD>_*.csv. load_screened_universe(as_of=...)
already consumes these, so this populates the history the PIT fix needs.

Rate-limited for the free CoinGecko tier (sleep between calls). Approx
(number of weeks) calls — trivial for free tier.

Usage:
    python backfill_universe_history.py [--start YYYY-MM-DD] [--step 7]
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd
import urllib.request

import screen_liquidity_idiosyncratic as sl

ROOT = Path("data")
OUT = Path("backtest_output")
OUT.mkdir(exist_ok=True)

# Our local 1d price history starts here (earliest bar in data/*_1d_max.csv).
DEFAULT_START = "2025-03-02"
CG_BASE = "https://api.coingecko.com/api/v3/coins/markets"


def fetch_markets_asof(date_ddmmyyyy: str, per_page: int = 250):
    """Fetch top coins by volume AS OF a date (DD-MM-YYYY). Free-tier safe:
    ONE page (top `per_page` by volume) — that's the liquid alt universe we
    actually backtest (only coins with local bars qualify anyway). Returns [] on
    any error so the caller can back off / skip."""
    url = (f"{CG_BASE}?vs_currency=usd&order=volume_desc&per_page={per_page}"
           f"&page=1&sparkline=false&date={date_ddmmyyyy}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except Exception as e:
        print(f"  fetch error ({date_ddmmyyyy}): {e!r}")
        return []


def to_universe_csv(rows, date_iso):
    """Mirror fetch_coingecko_universe.py's filter, write universe_broad_<date>.csv."""
    stable = ["usdt","usdc","busd","tusd","dai","frax","usdd","usde","pyusd",
              "rlusd","fdusd","euroc","eurc"]
    filtered = []
    for c in rows:
        vol = c.get("total_volume") or 0
        if vol < 100000:
            continue
        sym = (c.get("symbol") or "").lower()
        name = (c.get("name") or "").lower()
        if any(k in sym or k in name for k in stable):
            continue
        filtered.append({
            "id": c.get("id"),
            "symbol": c.get("symbol", "").upper(),
            "name": c.get("name", ""),
            "tier": "",  # tier assigned by screen script from mcap_rank proxy below
            "market_cap_rank": c.get("market_cap_rank"),
            "market_cap_usd": c.get("market_cap"),
            "volume_24h_usd": c.get("total_volume"),
            "current_price_usd": c.get("current_price"),
        })
    # derive a coarse tier from market_cap_rank (screen script reads 'tier')
    for r in filtered:
        rk = r.get("market_cap_rank") or 9999
        r["tier"] = "large" if rk <= 100 else ("mid" if rk <= 400 else "tail")
    df = pd.DataFrame(filtered)
    path = ROOT / f"universe_broad_{date_iso.replace('-', '')}.csv"
    df.to_csv(path, index=False)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=DEFAULT_START)
    ap.add_argument("--step", type=int, default=7)
    ap.add_argument("--end", default=pd.Timestamp.now().strftime("%Y-%m-%d"))
    ap.add_argument("--sleep", type=int, default=15, help="seconds between weekly fetches (free-tier safe)")
    args = ap.parse_args()

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    weeks = pd.date_range(start, end, freq=f"{args.step}D")
    print(f"Backfilling PIT universe: {len(weeks)} weekly snapshots {start.date()} .. {end.date()}")

    done = 0
    for d in weeks:
        iso = d.strftime("%Y-%m-%d")
        out = OUT / f"screen_liqu_idio_{iso.replace('-', '')}_000000.csv"
        if out.exists():
            print(f"[{iso}] already have snapshot, skipping")
            continue
        ddmmyyyy = d.strftime("%d-%m-%Y")   # CoinGecko date format
        print(f"[{iso}] fetching CoinGecko universe as-of {ddmmyyyy} ...", flush=True)
        rows = fetch_markets_asof(ddmmyyyy)
        if not rows:
            # 429 or empty: back off once, then skip this date
            print(f"  no data for {iso}, backing off {args.sleep*2}s")
            time.sleep(args.sleep * 2)
            continue
        u_path = to_universe_csv(rows, iso)
        sl.run_screen(universe_csv=str(u_path), out_csv=str(out))
        done += 1
        time.sleep(args.sleep)   # free-tier politeness between weekly snapshots

    print(f"\nBackfill complete: {done} PIT snapshots written to {OUT}")


if __name__ == "__main__":
    main()
