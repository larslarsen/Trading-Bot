#!/usr/bin/env python3
"""
Backfill multi-asset 5m bars into one local CSV (multi_5m.csv).
Schema: ts, symbol, open, high, low, close, volume

Seeds from the deep local 5m history we already have (BTC via btc_5m.csv,
DOGE via data/DOGEUSDT_5m_max.csv) so the cross-asset file always covers the
target's trading era — then optionally appends any additional symbols fetched
from the network (e.g. SOL via ccxt). The local seed makes the file correct
without network access and prevents stale/asset-wrong regeneration.
"""
import time
import pandas as pd
import ccxt
from datetime import datetime, timezone
from pathlib import Path

EXCHANGES = [
    ('binance', [('SOL/USDT', 'since_2018')]),
]

ROOT = Path(__file__).parent
OUT = ROOT / 'multi_5m.csv'
STOP_AT = datetime.now(timezone.utc)
BATCH = 1000

# Local deep sources seeded first so the file is always era-correct.
LOCAL_SEEDS = [
    (ROOT / 'btc_5m.csv', 'BTCUSDT'),
    (ROOT / 'data' / 'DOGEUSDT_5m_max.csv', 'DOGEUSDT'),
]


def seed_from_local():
    frames = []
    for path, sym in LOCAL_SEEDS:
        if not path.exists():
            continue
        d = pd.read_csv(path, parse_dates=['ts'])
        if d['ts'].dt.tz is None:
            d['ts'] = d['ts'].dt.tz_localize('UTC')
        d['symbol'] = sym
        frames.append(d[['ts', 'symbol', 'open', 'high', 'low', 'close', 'volume']])
        print(f"[local] seeded {sym}: {len(d)} bars")
    return frames


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
    frames = seed_from_local()
    for name, symbols in EXCHANGES:
        try:
            frames.extend(fetch_exchange(name, symbols))
        except Exception as e:
            print(f'[{name}] failed: {e}')

    if not frames:
        print('No data.')
        return

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=['ts', 'symbol']).sort_values(['symbol', 'ts']).reset_index(drop=True)
    merged = merged.dropna(subset=['open', 'high', 'low', 'close'])
    cols = ['ts', 'symbol', 'open', 'high', 'low', 'close', 'volume']
    merged = merged[cols]
    merged.to_csv(OUT, index=False)
    print(f'Wrote {len(merged)} rows to {OUT}')
    print(f'Symbols: {sorted(merged["symbol"].unique())}')
    print(f'Range: {merged["ts"].iloc[0]} -> {merged["ts"].iloc[-1]}')


if __name__ == '__main__':
    main()
