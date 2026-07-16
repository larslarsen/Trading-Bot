#!/usr/bin/env python3
"""
One-shot parallel CEX 5m backfill via the Binance bulk CDN (no rate limit).

Memory-safe: for a fresh symbol, history is STREAMED month-by-month straight to
the CSV (months arrive in chronological order from the CDN, so we append without
ever holding full history in RAM -- this avoids the OOM that killed the naive
accumulate-then-concat version). Resumable: skips symbols already current, and
a killed run simply continues on restart.

Writes canonical paths (btc_5m.csv / data/<SYM>USDT_5m_max.csv), derives
1h/4h/1d, then the poller keeps them topped up live.

Usage: python bulk_backfill_all_5m.py [--workers 4] [--min-bars 1000]
"""
import argparse
import concurrent.futures as cf
import csv as _csv
import gc
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import backfill_cex_all as cex
import derive_cex_tf as derive

REPO = Path(__file__).parent
START_MS = 1262304000000  # 2010-01-01; CDN 404s before listing, skipped via probe
COLS = ["ts", "open", "high", "low", "close", "volume"]


def cex_5m_path(sym):
    if sym == "BTCUSDT":
        return REPO / "btc_5m.csv"
    return REPO / "data" / f"{sym.replace('USDT', '')}USDT_5m_max.csv"


def _rows_to_ms(rows):
    """12-field kline rows -> list of [ms_ts, o,h,l,c,v]."""
    return [[r[0], r[1], r[2], r[3], r[4], r[5]] for r in rows]


def stream_fresh(sym, tf, path):
    """Fresh symbol: fetch month-by-month, append to CSV as we go (flat memory)."""
    fam = cex.first_available_month(sym, tf, START_MS)
    if fam is None:
        return 0
    now = datetime.now(timezone.utc)
    total = 0
    tmp = path.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="") as fh:
        w = _csv.writer(fh)
        w.writerow(COLS)
        d = datetime(fam[0], fam[1], 1, tzinfo=timezone.utc)
        while (d.year, d.month) <= (now.year, now.month):
            if (d.year, d.month) == (now.year, now.month):
                for day in range(1, now.day + 1):
                    rows = cex._fetch_zip_csv(
                        f"{cex.CDN}/daily/klines/{sym}/{tf}/{sym}-{tf}-{d.year:04d}-{d.month:02d}-{day:02d}.zip")
                    if rows:
                        for r in _rows_to_ms(rows):
                            ms = r[0] if r[0] < 10_000_000_000_000 else r[0] // 1000
                            w.writerow([pd.Timestamp(ms, unit="ms", tz="UTC")] + r[1:])
                            total += 1
            else:
                rows = cex._fetch_zip_csv(
                    f"{cex.CDN}/monthly/klines/{sym}/{tf}/{sym}-{tf}-{d.year:04d}-{d.month:02d}.zip")
                if rows:
                    for r in _rows_to_ms(rows):
                        ms = r[0] if r[0] < 10_000_000_000_000 else r[0] // 1000
                        w.writerow([pd.Timestamp(ms, unit="ms", tz="UTC")] + r[1:])
                        total += 1
            d = (d.replace(day=28) + pd.Timedelta(days=7)).replace(day=1)
    if total:
        tmp.replace(path)
    else:
        tmp.unlink(missing_ok=True)
    return total


def do_symbol(sym, min_bars):
    path = cex_5m_path(sym)
    last = cex.existing_last_ms(path)
    now_ms = int(time.time() * 1000)
    if last is not None and last >= cex.floor_ts(now_ms, "5m") - 10 * 60 * 1000:
        try:
            n = sum(1 for _ in open(path)) - 1
        except Exception:
            n = 0
        if n >= min_bars:
            return sym, "skip", n
    try:
        if last is None:
            n = stream_fresh(sym, "5m", path)
            status = "ok" if n else "nodata"
        else:
            # incremental top-up (small) -> safe to hold in memory
            rows = cex.pull_bulk(sym, "5m", cex.floor_ts(last + 1, "5m"))
            if not rows:
                return sym, "skip", (sum(1 for _ in open(path)) - 1)
            df = pd.DataFrame(rows, columns=cex_cols())
            df = df[COLS].copy()
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            old = pd.read_csv(path)
            old["ts"] = cex.parse_ts(old["ts"])
            df = pd.concat([old, df]).drop_duplicates(subset=["ts"]).sort_values("ts")
            df.to_csv(path, index=False)
            n = len(df)
            status = "ok"
        try:
            derive.derive_sym(sym, ["1h", "4h", "1d"])
        except Exception:
            pass
        gc.collect()
        return sym, status, n
    except Exception as e:
        return sym, f"err:{e}", 0


def cex_cols():
    return ["ts", "open", "high", "low", "close", "volume",
            "close_time", "qav", "trades", "tbav", "tqav", "ignore"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
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
            if done % 10 == 0 or status.startswith("err"):
                print(f"  [{done}/{len(syms)}] ok={ok} skip={skip} nodata={nodata} "
                      f"err={err} ({time.time()-t0:.0f}s)", flush=True)
    print(f"DONE: {done} symbols in {time.time()-t0:.0f}s -- ok={ok} skip={skip} "
          f"nodata={nodata} err={err}", flush=True)


if __name__ == "__main__":
    main()
