"""
Kraken deep-history backfill. Pull daily OHLC for all pairs in kraken_pairs.csv
(787 pairs) via Kraken public OHLC, paging backward with `since` to get multi-year
history. Save to data/<SYMBOL>_1d_max.csv. Runs in background; logs progress.

Kraken OHLC row: [time, open, high, low, close, vwap, volume, count].
Caps ~720 bars per call; page with since = first_ts_ms - 1.
"""
import time
import json
import urllib.request
import pandas as pd
from pathlib import Path

ROOT = Path('data')
PAUSE = 0.4
PAGE_BARS = 700
MAX_PAGES = 30  # ~30 * 700 = 21000 bars ~ 57y cap (plenty)


def fetch(pair, since_ms):
    url = f'https://api.kraken.com/0/public/OHLC?pair={pair}&interval=1440&since={int(since_ms)}'
    req = urllib.request.Request(url, headers={'User-Agent': 'curl/7.88.1'})
    with urllib.request.urlopen(req, timeout=30) as r:
        d = json.load(r)
    if d.get('error'):
        return None
    res = d['result']
    key = [k for k in res if k != 'last'][0]
    return res[key]


def main():
    kp = pd.read_csv(ROOT / 'kraken_pairs.csv')
    # kraken symbol is like 'XBTUSD' (pair_id). Map to a clean filename symbol.
    pairs = kp['pair_id'].dropna().unique().tolist()
    print(f'Kraken backfill: {len(pairs)} pairs')
    done = 0
    for pair in pairs:
        try:
            sym = str(pair)
            out = ROOT / f'{sym}_1d_max.csv'
            if out.exists():
                continue
            # find earliest tradable date by paging back
            rows = []
            since = int(pd.Timestamp('2026-07-14').timestamp())
            for _ in range(MAX_PAGES):
                raw = fetch(sym, since * 1000)
                if not raw:
                    break
                # convert: row[0]=time(s)
                page = []
                for r in raw:
                    ts = pd.to_datetime(int(float(r[0])), unit='s')
                    page.append({'ts': ts, 'open': float(r[1]), 'high': float(r[2]),
                                 'low': float(r[3]), 'close': float(r[4]), 'volume': float(r[6])})
                if not page:
                    break
                rows = page + rows
                earliest = int(float(raw[0][0]))
                since = earliest - PAGE_BARS * 86400
                if len(raw) < PAGE_BARS:
                    break
                time.sleep(PAUSE)
            if rows:
                df = pd.DataFrame(rows).drop_duplicates('ts').sort_values('ts').reset_index(drop=True)
                df.to_csv(out, index=False)
                done += 1
                print(f'  {sym:12s} {len(df)} bars {df["ts"].min().date()}..{df["ts"].max().date()}')
        except Exception as e:
            print(f'  {sym} ERR {e}')
        time.sleep(PAUSE)
    print(f'Kraken backfill DONE. New files: {done}')


if __name__ == '__main__':
    main()
