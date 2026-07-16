#!/usr/bin/env python3
"""
One-shot parallel CEX 5m backfill via the Binance bulk CDN (no rate limit).

The poller sweeps one symbol per pass (slow). The CDN has no per-IP throttle,
so we can pull ALL missing symbols concurrently and land full history fast.
Writes canonical paths (btc_5m.csv / data/<SYM>USDT_5m_max.csv), derives
1h/4h/1d, then the poller keeps them topped up live.

Usage: python bulk_backfill_all_5m.py [--workers 12] [--min-bars 1000]
"""
import argparse
import concurrent.futures as cf
import time
from pathlib import Path

import pandas as pd

import backfill_cex_all as cex
import derive_cex_tf as derive

REPO = Path(__file__).parent
START_MS = 1262304000000  # 2010-01-01; CDN 404s before listing, harmless


def cex_5m_path(sym):
    if sym == "BTCUSDT":
        return REPO / "btc_5m.csv"
    return REPO / "data" / f"{sym.replace('USDT', '')}USDT_5m_max.csv"


def do_symbol(sym, min_bars):
    path = cex_5m_path(sym)
    last = cex.existing_last_ms(path)
    now_ms = int(time.time() * 1000)
    if last is not None and last >= cex.floor_ts(now_ms, "5m") - 5 * 60 * 1000:
        # already current; count rows for reporting
        try:
            n = sum(1 for _ in open(path)) - 1
        except Exception:
            n = 0
        if n >= min_bars:
            return sym, "skip", n
    start = START_MS if last is None else cex.floor_ts(last + 1, "5m")
    try:
        rows = cex.pull_bulk(sym, "5m", start)
        if not rows:
            return sym, "nodata", 0
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume",
                                         "close_time", "qav", "trades", "tbav", "tqav", "ignore"])
        df = df[["ts", "open", "high", "low", "close", "volume"]].copy()
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        if path.exists():
            old = pd.read_csv(path)
            old["ts"] = pd.to_datetime(old["ts"], utc=True)
            df = pd.concat([old, df]).drop_duplicates(subset=["ts"]).sort_values("ts")
        df.to_csv(path, index=False)
        try:
            derive.derive_sym(sym, ["1h", "4h", "1d"])
        except Exception:
            pass
        return sym, "ok", len(df)
    except Exception as e:
        return sym, f"err:{e}", 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--min-bars", type=int, default=1000)
    args = ap.parse_args()
    syms = cex.get_syms()
    print(f"bulk backfilling {len(syms)} symbols via CDN, {args.workers} workers", flush=True)
    done = ok = skip = nodata = err = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(do_symbol, s, args.min_bars): s for s in syms}
        for fut in cf.as_completed(futs):
            sym, status, n = fut.result()
            done += 1
            if status == "ok":
                ok += 1
            elif status == "skip":
                skip += 1
            elif status == "nodata":
                nodata += 1
            else:
                err += 1
                print(f"  {sym}: {status}", flush=True)
            if done % 20 == 0 or status.startswith("err"):
                print(f"  [{done}/{len(syms)}] ok={ok} skip={skip} nodata={nodata} err={err} "
                      f"({time.time()-t0:.0f}s)", flush=True)
    print(f"DONE: {done} symbols in {time.time()-t0:.0f}s -- ok={ok} skip={skip} "
          f"nodata={nodata} err={err}", flush=True)


if __name__ == "__main__":
    main()
