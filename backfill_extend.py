"""
Extend 1d history for all *_1d_max.csv coins by pulling OLDER bars via MEXC
klines using BOTH startTime+endTime (MEXC ignores startTime alone). Prepends
to existing files. Idempotent: only fetches bars older than current earliest.

Confirmed: api.mexc.com/api/v3/klines honors startTime+endTime together.
Strategy: window = [cursor_end - LIMIT*1d, cursor_end), fetch, prepend, move
cursor_end back, repeat until empty or MAX_PAGES reached.

Run in background; prints progress. Safe to re-run.
"""
import time
import json
import urllib.request
import pandas as pd
from pathlib import Path

ROOT = Path('data')
LIMIT = 1000
DAY_MS = 86_400_000
BATCH_SLEEP = 0.3
MAX_PAGES = 8  # ~8000 bars back (~22y, far beyond any listing)


def fetch_window(symbol, start_ms, end_ms, limit=LIMIT):
    url = (f'https://api.mexc.com/api/v3/klines?symbol={symbol}'
           f'&interval=1d&startTime={int(start_ms)}&endTime={int(end_ms)}&limit={limit}')
    req = urllib.request.Request(url, headers={'User-Agent': 'curl/7.88.1'})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def main():
    files = sorted(ROOT.glob('*_1d_max.csv'))
    print(f'Extending history for {len(files)} coins (MEXC startTime+endTime paging)')
    total_new = 0
    for p in files:
        try:
            df = pd.read_csv(p)
            if len(df) == 0:
                continue
            df['ts'] = pd.to_datetime(df['ts'], utc=True).dt.tz_localize(None) if str(df['ts'].dtype).startswith('datetime64') and df['ts'].dt.tz is not None else pd.to_datetime(df['ts'])
            df = df.sort_values('ts').reset_index(drop=True)
            sym = p.name.replace('_1d_max.csv', '')
            cursor_end = int(pd.Timestamp(df['ts'].iloc[0]).timestamp() * 1000) - DAY_MS
            all_rows = []
            for _ in range(MAX_PAGES):
                start_ms = cursor_end - LIMIT * DAY_MS
                raw = fetch_window(sym, start_ms, cursor_end, LIMIT)
                if not raw:
                    break
                rows = []
                for k in raw:
                    ts = pd.to_datetime(int(k[0]), unit='ms', utc=True).tz_localize(None)
                    rows.append({'ts': ts, 'open': float(k[1]), 'high': float(k[2]),
                                 'low': float(k[3]), 'close': float(k[4]), 'volume': float(k[5])})
                newdf = pd.DataFrame(rows).sort_values('ts').reset_index(drop=True)
                newdf = newdf[newdf['ts'] < df['ts'].iloc[0]]
                if len(newdf) == 0:
                    break
                all_rows.append(newdf)
                total_new += len(newdf)
                cursor_end = int(pd.Timestamp(newdf['ts'].iloc[0]).timestamp() * 1000) - DAY_MS
                if len(newdf) < LIMIT:
                    break
                time.sleep(BATCH_SLEEP)
            if all_rows:
                combined = pd.concat(all_rows + [df], ignore_index=True).drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
                combined.to_csv(p, index=False)
                print(f'  {p.name:22s} +{len(combined)-len(df)} bars -> {combined["ts"].min().date()}..{combined["ts"].max().date()}')
            else:
                print(f'  {p.name:22s} no older data')
        except Exception as e:
            print(f'  {p.name:22s} ERROR {e}')
        time.sleep(BATCH_SLEEP)
    print(f'DONE. Total new bars added: {total_new}')


if __name__ == '__main__':
    main()
