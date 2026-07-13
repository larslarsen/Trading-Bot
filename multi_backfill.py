#!/usr/bin/env python3
"""
Backfill BTC + ETH 5m bars from Binance and Bybit into one local CSV.
Schema: ts, open, high, low, close, volume, symbol
"""
import time
import pandas as pd
import ccxt
from datetime import datetime, timezone
from pathlib import Path

EXCHANGES = [
    ('binance', [('SOL/USDT', 'since_2018')]),
]

OUT = Path(__file__).parent / 'multi_5m.csv'
STOP_AT = datetime.now(timezone.utc)
BATCH = 1000


def fetch_exchange(name, symbols):
    ex = getattr(ccxt, name)({'enableRateLimit': True})
    ex.load_markets()
    frames = []
    for symbol, label in symbols:
        if symbol not in ex.markets:
            print(f'[{name}] {symbol} not available')
            continue
        since = ex.parse8601('2018-01-01T00:00:00Z') if label == 'since_2018' else ex.parse8601('2021-07-01T00:00:00Z')
        print(f'[{name}] {symbol} 5m since {datetime.fromtimestamp(since/1000, tz=timezone.utc).date()}...')
        rows, last_fetched, fails = [], None, 0
        while since < STOP_AT.timestamp() * 1000:
            try:
                batch = ex.fetch_ohlcv(symbol, timeframe='5m', since=since, limit=BATCH)
            except Exception as e:
                fails += 1
                print(f'  error: {e}')
                if fails >= 5:
                    break
                time.sleep(2)
                continue
            if not batch:
                break
            rows.extend(batch)
            since = batch[-1][0] + 1
            last_fetched = datetime.fromtimestamp(batch[-1][0]/1000, tz=timezone.utc)
            if len(batch) < BATCH:
                break
            if len(rows) % 50000 == 0:
                print(f'  ... {len(rows)} bars')
            time.sleep(0.2)
        print(f'[{name}] {symbol}: {len(rows)} bars through {last_fetched}')
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        df['symbol'] = symbol.replace('/', '')
        df = df.drop_duplicates(subset=['timestamp', 'symbol']).sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        frames.append(df)
    return frames


def main():
    existing = None
    if OUT.exists():
        existing = pd.read_csv(OUT, parse_dates=['ts'])
        rename_cols = {'ts': 'timestamp', 'sym': 'symbol'}
        existing = existing.rename(columns={k: v for k, v in rename_cols.items() if k in existing.columns})
        if 'timestamp' in existing.columns and existing['timestamp'].dt.tz is None:
            existing['timestamp'] = existing['timestamp'].dt.tz_localize('UTC')
        existing = existing.drop_duplicates(subset=['timestamp', 'symbol']).sort_values(['timestamp', 'symbol']).reset_index(drop=True)
        print(f'Existing: {len(existing)} rows, symbols={sorted(existing["symbol"].unique()) if "symbol" in existing.columns else "N/A"}')

    frames = []
    for name, symbols in EXCHANGES:
        try:
            frames.extend(fetch_exchange(name, symbols))
        except Exception as e:
            print(f'[{name}] failed: {e}')

    if not frames:
        print('No data fetched.')
        return

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=['timestamp', 'symbol']).sort_values(['timestamp', 'symbol']).reset_index(drop=True)
    if existing is not None and len(existing):
        merged = pd.concat([existing, merged], ignore_index=True)
        merged = merged.drop_duplicates(subset=['timestamp', 'symbol']).sort_values(['timestamp', 'symbol']).reset_index(drop=True)

    merged.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
    if 'timestamp' in merged.columns:
        merged = merged.rename(columns={'timestamp': 'ts'})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cols = ['ts', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    merged = merged[cols]
    merged.to_csv(OUT, index=False)
    print(f'Wrote {len(merged)} rows to {OUT}')
    print(f'Symbols: {sorted(merged["symbol"].unique())}')
    print(f'Range: {merged["ts"].iloc[0]} -> {merged["ts"].iloc[-1]}')


if __name__ == '__main__':
    main()
