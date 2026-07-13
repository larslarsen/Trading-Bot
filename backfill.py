#!/usr/bin/env python3
"""
Backfill historical BTC/USDT 5m bars from public exchange APIs via CCXT.
No API keys required. Merges new data with existing btc_5m.csv.
"""
import time
import pandas as pd
import ccxt
from pathlib import Path
from datetime import datetime, timezone

EXCHANGES = [
    ('binance', 'BTC/USDT', 'since_2018'),
    ('bybit',  'BTC/USDT', 'since_2021'),
]

OUT = Path(__file__).parent / 'btc_5m.csv'
STOP_AT = datetime.now(timezone.utc)

BATCH = 1000  # CCXT limit per request


def fetch_exchange(name, symbol, label):
    ex = getattr(ccxt, name)({'enableRateLimit': True})
    # Historical starting points
    if label == 'since_2018':
        since = ex.parse8601('2018-01-01T00:00:00Z')
    else:
        since = ex.parse8601('2021-07-01T00:00:00Z')

    print(f'[{name}] fetching {symbol} 5m bars since {datetime.fromtimestamp(since/1000, tz=timezone.utc).date()}...')
    all_rows = []
    last_fetched = None
    fails = 0
    while since < STOP_AT.timestamp() * 1000:
        try:
            batch = ex.fetch_ohlcv(symbol, timeframe='5m', since=since, limit=BATCH)
        except Exception as e:
            fails += 1
            print(f'  fetch error: {e}')
            if fails >= 5:
                break
            time.sleep(2)
            continue
        if not batch:
            break
        all_rows.extend(batch)
        since = batch[-1][0] + 1
        last_fetched = datetime.fromtimestamp(batch[-1][0]/1000, tz=timezone.utc)
        if len(batch) < BATCH:
            break
        if len(all_rows) % 50000 == 0:
            print(f'  ... {len(all_rows)} bars')
        time.sleep(0.2)

    print(f'[{name}] got {len(all_rows)} bars through {last_fetched}')
    if not all_rows:
        return []
    df = pd.DataFrame(all_rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    return df


def main():
    existing = None
    if OUT.exists():
        existing = pd.read_csv(OUT, parse_dates=['ts'])
        if 'ts' in existing.columns:
            existing = existing.rename(columns={'ts': 'timestamp'})
        if 'timestamp' in existing.columns and existing['timestamp'].dt.tz is None:
            existing['timestamp'] = existing['timestamp'].dt.tz_localize('UTC')
        existing = existing.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        print(f'Existing local file: {len(existing)} bars from {existing["timestamp"].iloc[0]} to {existing["timestamp"].iloc[-1]}')

    frames = []
    for name, symbol, label in EXCHANGES:
        try:
            df = fetch_exchange(name, symbol, label)
            if len(df):
                frames.append(df)
        except Exception as e:
            print(f'[{name}] failed: {e}')

    if not frames:
        print('No new data fetched. Exiting.')
        return

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    if existing is not None and len(existing):
        merged = pd.concat([existing, merged], ignore_index=True)
        merged = merged.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)

    merged.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    merged = merged.sort_values('timestamp').reset_index(drop=True)
    if 'timestamp' in merged.columns:
        merged = merged.rename(columns={'timestamp': 'ts'})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ['ts','open','high','low','close','volume']
    merged = merged[cols]
    merged.to_csv(OUT, index=False)
    print(f'Wrote {len(merged)} bars to {OUT}')
    print(f'Range: {merged["ts"].iloc[0]} -> {merged["ts"].iloc[-1]}')


if __name__ == '__main__':
    main()
