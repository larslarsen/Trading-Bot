#!/usr/bin/env python3
"""
Live data feed via CCXT.
Supports Kraken spot (US accessible) and prepares for MEXC futures.
Fetches latest OHLCV, merges with local history, and emits rolling windows.
"""
import ccxt
import pandas as pd
import time
from datetime import datetime, timezone
from pathlib import Path

# ── CONFIG ──────────────────────────────────────────────────────────
SYMBOL    = "BTC/USDT"
TIMEFRAME = "5m"
EXCHANGE  = "kraken"           # US-accessible
LOCAL_CSV = Path(__file__).parent / "btc_5m.csv"
HISTORY_CSV = Path(__file__).parent / "live_history.csv"
FETCH_LIMIT = 500              # max bars per REST call
LONG_POLL_SEC = 270            # ~4.5 min to stay under 5m bar boundary


def get_exchange(name=EXCHANGE):
    """Return configured CCXT exchange instance."""
    ex_class = getattr(ccxt, name)
    return ex_class({"enableRateLimit": True})


def fetch_latest(exchange=None):
    """Fetch most recent OHLCV candles from exchange."""
    if exchange is None:
        exchange = get_exchange()

    bars = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=FETCH_LIMIT)
    df = pd.DataFrame(bars, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return df


def append_to_history(new_bars):
    """Append new bars to live_history.csv, deduplicate."""
    cols = ["ts", "open", "high", "low", "close", "volume"]
    if HISTORY_CSV.exists():
        hist = pd.read_csv(HISTORY_CSV, parse_dates=["ts"])
        # Ensure UTC
        if hist["ts"].dt.tz is None:
            hist["ts"] = hist["ts"].dt.tz_localize("UTC")
        combined = pd.concat([hist[cols], new_bars[cols]], ignore_index=True)
        combined = combined.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    else:
        combined = new_bars[cols].copy()

    combined.to_csv(HISTORY_CSV, index=False)
    return combined


def backfill_since(since_dt):
    """Fetch bars from exchange since a datetime."""
    exchange = get_exchange()
    since_ms = int(since_dt.timestamp() * 1000)
    all_bars = []
    while True:
        bars = exchange.fetch_ohlcv(
            SYMBOL, timeframe=TIMEFRAME,
            since=since_ms, limit=FETCH_LIMIT
        )
        if not bars:
            break
        all_bars.extend(bars)
        since_ms = bars[-1][0] + 1
        if len(bars) < FETCH_LIMIT:
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_bars, columns=["ts", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.drop_duplicates("ts").sort_values("ts").reset_index(drop=True)
    return df


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live data feed")
    parser.add_argument("--backfill", action="store_true", help="Backfill from local CSV last timestamp")
    parser.add_argument("--latest", action="store_true", help="Fetch latest bar and append")
    parser.add_argument("--run", action="store_true", help=f"Poll every {LONG_POLL_SEC}s")
    args = parser.parse_args()

    if args.backfill:
        print("Backfilling from exchange since local CSV end...")
        if LOCAL_CSV.exists():
            local = pd.read_csv(LOCAL_CSV, parse_dates=["ts"])
            if local["ts"].dt.tz is None:
                local["ts"] = local["ts"].dt.tz_localize("UTC")
            last_ts = local["ts"].max()
        else:
            last_ts = pd.Timestamp("2024-01-01", tz="UTC")
        new = backfill_since(last_ts)
        print(f"Fetched {len(new)} bars since {last_ts}")
        append_to_history(new)

    elif args.latest:
        print("Fetching latest bar...")
        df = fetch_latest()
        append_to_history(df)
        print(f"Latest: {df.iloc[-1]['ts']} close={df.iloc[-1]['close']}")

    elif args.run:
        print(f"Live polling every {LONG_POLL_SEC}s (Ctrl+C to stop)")
        while True:
            try:
                df = fetch_latest()
                append_to_history(df)
                print(f"[{datetime.now(timezone.utc).strftime('%H:%M:%S')}] "
                      f"bars={len(df)}, latest={df.iloc[-1]['ts']}, close={df.iloc[-1]['close']}")
            except Exception as e:
                print(f"Fetch error: {e}")
            time.sleep(LONG_POLL_SEC)

    else:
        parser.print_help()
