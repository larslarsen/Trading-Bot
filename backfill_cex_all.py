#!/usr/bin/env python3
"""
Backfill ALL Binance USDT spot pairs across all timeframes (5m/1h/4h/1d) from
the free Binance klines mirror (no key). Writes to data/cex/<SYM>_<tf>.csv and
registers each file in MANIFEST.json.

Resumable: skips (tf, symbol) already present with sufficient depth.
Multi-symbol + multi-tf. Rate-limited (0.25s/symbol-tf). Logs progress.

Usage:
  python backfill_cex_all.py            # all 459 symbols, all tfs
  python backfill_cex_all.py --syms BTCUSDT,ETHUSDT
  python backfill_cex_all.py --tfs 1d,1h
"""
import argparse
import time
import requests
import pandas as pd
from pathlib import Path

REPO = Path(__file__).parent
CEX = REPO / "data" / "cex"
CEX.mkdir(parents=True, exist_ok=True)
SYMS_FILE = REPO / "all_binance_usdt.txt"
TFS = ["5m"]  # only 5m from API; 1h/4h/1d derived locally (derive_cex_tf.py)
# (previously ["5m","1h","4h","1d"]; pulling higher TFs directly is redundant
#  once full-depth 5m exists -- resampling is exact and costs zero API calls)
BASE = "https://data-api.binance.vision/api/v3/klines"
LIMIT = 1000
SLEEP = 0.5          # per-page pause; mirror has a per-IP limit, so 2 workers
BACKOFF = 10         # seconds on HTTP 429 (multiplied by retry count, capped 60s)
MAX_429_RETRIES = 8  # cap so a persistently-throttled symbol yields instead of hanging forever
MAX_NEW_PER_PULL = 1_000_000  # safety cap to avoid runaway


def get_syms():
    if SYMS_FILE.exists():
        return [s.strip() for s in SYMS_FILE.read_text().splitlines() if s.strip()]
    info = requests.get(f"{BASE.rsplit('/klines',1)[0]}/exchangeInfo", timeout=30).json()
    syms = [s["symbol"] for s in info["symbols"]
            if s["symbol"].endswith("USDT") and s["status"] == "TRADING"
            and s.get("isSpotTradingAllowed")]
    SYMS_FILE.write_text("\n".join(syms))
    return syms


def floor_ts(ts_ms, tf):
    # align to tf boundary so appends never duplicate
    mins = {"5m": 5, "1h": 60, "4h": 240, "1d": 1440}[tf]
    bar_ms = mins * 60 * 1000
    return (ts_ms // bar_ms) * bar_ms


def pull(sym, tf, start_ms):
    out = []
    nxt = start_ms
    retries = 0
    while True:
        try:
            r = requests.get(BASE, params={"symbol": sym, "interval": tf,
                                           "startTime": nxt, "limit": LIMIT}, timeout=30)
        except Exception as e:
            retries += 1
            if retries > MAX_429_RETRIES:
                print(f"  [{sym} {tf}] giving up after {retries} network errors: {e}", flush=True)
                break
            time.sleep(min(BACKOFF * retries, 60))
            continue
        if r.status_code != 200:
            if r.status_code == 429:
                retries += 1
                if retries > MAX_429_RETRIES:
                    print(f"  [{sym} {tf}] 429 cap hit ({retries}); yielding", flush=True)
                    break
                time.sleep(min(BACKOFF * retries, 60))
                continue
            print(f"  [{sym} {tf}] HTTP {r.status_code}: {r.text[:120]}")
            break
        retries = 0
        rows = r.json()
        if not rows:
            break
        out.extend(rows)
        nxt = rows[-1][0] + 1
        if len(rows) < LIMIT:
            break
        if len(out) >= MAX_NEW_PER_PULL:
            print(f"  [{sym} {tf}] hit new-cap, stopping")
            break
        time.sleep(SLEEP)
    return out


def existing_last_ms(path):
    if not path.exists():
        return None
    try:
        d = pd.read_csv(path, usecols=["ts"])
        return int(pd.to_datetime(d["ts"]).max().timestamp() * 1000)
    except Exception:
        return None


# ── BULK CDN FETCH (data.binance.vision static ZIPs, NOT rate-limited) ────────
# The REST klines API rate-limits per-IP (429 storms). The bulk data CDN serves
# the SAME klines as static monthly/daily ZIP files with no per-IP throttle, so
# deep history for hundreds of symbols downloads fast. We fetch monthly ZIPs for
# full history, daily ZIPs for the current partial month, and leave the last few
# live bars to the REST pull(). Kline schema in the ZIPs matches the API exactly
# (12 cols), so rows are drop-in compatible.
import io
import zipfile
import csv as _csv
from datetime import datetime, timezone, timedelta

CDN = "https://data.binance.vision/data/spot"


def _fetch_zip_csv(url):
    """Download a Binance bulk ZIP; return list of 12-field kline rows, or None
    if the file doesn't exist (404) -- 404 is normal (symbol not listed then)."""
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=60)
        except Exception:
            time.sleep(2 * (attempt + 1))
            continue
        if r.status_code == 404:
            return None
        if r.status_code != 200:
            time.sleep(2 * (attempt + 1))
            continue
        try:
            zf = zipfile.ZipFile(io.BytesIO(r.content))
            name = zf.namelist()[0]
            rows = []
            with zf.open(name) as fh:
                text = io.TextIOWrapper(fh, encoding="utf-8")
                for rec in _csv.reader(text):
                    if not rec or not rec[0].lstrip("-").isdigit():
                        continue  # skip header row present in newer dumps
                    # newer dumps use microsecond open-time (16 digits); older use
                    # milliseconds (13). Normalize to ms so schema matches the API.
                    ot = int(rec[0])
                    if ot > 10_000_000_000_000:  # > ~year 2286 in ms => actually µs
                        ot //= 1000
                    rows.append([ot] + rec[1:6] + rec[6:])
            return rows
        except Exception:
            return None
    return None


def _months_since(start_ms):
    """Yield (year, month) from start month through the current month, UTC."""
    d = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).replace(day=1)
    now = datetime.now(timezone.utc)
    while (d.year, d.month) <= (now.year, now.month):
        yield d.year, d.month
        d = (d.replace(day=28) + timedelta(days=7)).replace(day=1)


def _zip_exists(sym, tf, yy, mm):
    """Cheap HEAD-like check: does this monthly ZIP exist? (avoids downloading)."""
    url = f"{CDN}/monthly/klines/{sym}/{tf}/{sym}-{tf}-{yy:04d}-{mm:02d}.zip"
    try:
        r = requests.head(url, timeout=20)
        return r.status_code == 200
    except Exception:
        return False


def first_available_month(sym, tf, start_ms):
    """Find the earliest monthly ZIP that exists, so we skip the dead pre-listing
    404 span (2010->listing). Probe January of each year forward from start; once
    a year has data, narrow to the first month in that year."""
    start_year = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).year
    now = datetime.now(timezone.utc)
    for yy in range(start_year, now.year + 1):
        # coarse: does any of this year exist? probe a few months
        hit_year = any(_zip_exists(sym, tf, yy, mm) for mm in (1, 4, 7, 10))
        if not hit_year and yy < now.year:
            continue
        for mm in range(1, 13):
            if (yy, mm) > (now.year, now.month):
                break
            if _zip_exists(sym, tf, yy, mm):
                return yy, mm
    return None


def pull_bulk(sym, tf, start_ms):
    """Deep history via monthly ZIPs + current month via daily ZIPs. Returns
    12-field kline rows >= start_ms (dedup/sort handled by caller). Skips the
    dead pre-listing 404 span by locating the first available month first."""
    out = []
    now = datetime.now(timezone.utc)
    fam = first_available_month(sym, tf, start_ms)
    if fam is None:
        # no monthly data at all; try current-month dailies only
        first = (now.year, now.month)
    else:
        first = fam
    for (yy, mm) in _months_since(start_ms):
        if (yy, mm) < first:
            continue
        if (yy, mm) == (now.year, now.month):
            for day in range(1, now.day + 1):
                rows = _fetch_zip_csv(
                    f"{CDN}/daily/klines/{sym}/{tf}/{sym}-{tf}-{yy:04d}-{mm:02d}-{day:02d}.zip")
                if rows:
                    out.extend(rows)
        else:
            rows = _fetch_zip_csv(
                f"{CDN}/monthly/klines/{sym}/{tf}/{sym}-{tf}-{yy:04d}-{mm:02d}.zip")
            if rows:
                out.extend(rows)
    return [r for r in out if r[0] >= start_ms]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--syms", default=None, help="comma list; default all")
    ap.add_argument("--tfs", default=",".join(TFS), help="comma list of tfs")
    ap.add_argument("--shard", default=None, help="i/n sharding, e.g. 1/6")
    args = ap.parse_args()
    tfs = [t for t in args.tfs.split(",") if t in TFS]
    # Process fast lookback TFs first so data is usable immediately; 5m last
    # (it is the heaviest: deep history spans 1000s of pages per symbol).
    tfs.sort(key=lambda t: {"1d": 0, "1h": 1, "4h": 2, "5m": 3}[t])
    syms = args.syms.split(",") if args.syms else get_syms()
    tag = ""
    if args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        syms = syms[i - 1::n]   # round-robin slice across shards
        tag = f"[shard {args.shard}] "
    print(f"{tag}Backfilling {len(syms)} symbols x {tfs} tfs -> {CEX}")
    total = 0
    for tf in tfs:
        for sym in syms:
            path = CEX / f"{sym}_{tf}.csv"
            # skip if this tf already complete (deep enough) -> lets shards
            # resume without re-pulling finished tfs
            if path.exists() and existing_last_ms(path) is not None and \
               existing_last_ms(path) >= floor_ts(int(time.time() * 1000), tf):
                continue
            last = existing_last_ms(path)
            start = 1262304000000 if last is None else floor_ts(last + 1, tf)
            rows = pull(sym, tf, start)
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume",
                                             "close_time", "qav", "trades", "tbav", "tqav", "ignore"])
            df = df[["ts", "open", "high", "low", "close", "volume"]].copy()
            df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            if path.exists():
                old = pd.read_csv(path)
                old["ts"] = pd.to_datetime(old["ts"], utc=True)
                df = pd.concat([old, df]).drop_duplicates(subset=["ts"]).sort_values("ts")
            df.to_csv(path, index=False)
            total += 1
            if total % 20 == 0:
                print(f"{tag}wrote {total} files; latest {sym} {tf} -> {df['ts'].max()}")
        print(f"{tag}completed tf={tf} ({total} files so far)")


if __name__ == "__main__":
    main()
