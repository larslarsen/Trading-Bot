#!/usr/bin/env python3
"""
Backfill deep CEX 5m OHLCV for the whole universe from FREE sources.

WHY: the ML pipeline (model_trainer.py) is 5m-native and needs ~20k bars per
walk-forward fold. BTC had deep 5m (btc_5m.csv) but every other pair only had
shallow BloFin 5m (~9 days). This script closes that gap: pull deep 5m for
every universe symbol from Binance's FREE public data mirror (no API key, no
geo-block from this box) with MEXC as fallback.

SOURCE: https://data-api.binance.vision/api/v3/klines  (free, no key)
        falls back to https://api.mexc.com/api/v3/klines  (free, no key)
Binance klines cap 1000 bars/call; we paginate via startTime/endTime back to
the earliest available (~2020 for liquid pairs).

OUTPUT: data/<SYM>USDT_5m_max.csv  (matches <SYM>_<tf>_<venue>_max.csv naming)

SAFETY (same profile as backfill_dex_mtf.py): per-symbol gc.collect(), hard RSS
cap that aborts before any OOM, 1s sleep + exponential backoff on 429, skip
symbols already deep (<= 2023) so re-runs are cheap.

Usage:
    python backfill_cex_5m.py                 # full universe
    python backfill_cex_5m.py --symbol DOGE   # one symbol
    python backfill_cex_5m.py --limit 50      # cap symbols (smoke test)
"""
import argparse
import gc
import json
import time
import urllib.request
import urllib.error
from pathlib import Path

import pandas as pd
from mem_guard import guard as _mem_guard

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
DEFAULT_MEM_LIMIT_MB = 1536

BINANCE = "https://data-api.binance.vision/api/v3/klines"
MEXC = "https://api.mexc.com/api/v3/klines"
UA = {"User-Agent": "curl/7.88.1"}
SLEEP = 1.0
# one klines window worth of milliseconds (1000 bars x 5 min) — used to skip
# over gaps in history without re-requesting the same range.
WINDOW_MS = 1000 * 5 * 60 * 1000


def _get_json(url: str, tries: int = 6):
    last = None
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers=UA)
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = 5 * (2 ** i)
                print(f"    429 -> backoff {wait}s")
                time.sleep(wait)
                continue
            raise
        except Exception as e:
            last = e
            time.sleep(SLEEP * (i + 1))
    if last is not None:
        raise last
    return {}


def backfill_symbol(stem: str, mem_limit: int) -> int:
    """Pull deep 5m for one stem (e.g. DOGE). Returns NEW bars written.
    Writes INCREMENTALLY (append per 1000-bar window) so a kill/resume is safe
    and progress persists. Resumes from the file's last ts if partially done."""
    _mem_guard(mem_limit)
    sym = stem.upper()
    bin_sym = f"{sym}USDT"
    out = DATA / f"{bin_sym}_5m_max.csv"

    # resume / skip logic:
    #  - if file exists AND its last bar is recent (within 2 days of now) ->
    #    it's a complete, up-to-date pull -> skip (don't re-fetch).
    #  - if file exists but its last bar is old -> it's a partial/killed run ->
    #    resume from just after its last ts.
    #  - else (no file) -> start from 2020.
    start_dt = None
    floor_ts = None  # newest bar already on disk; appended windows are filtered to ts > floor_ts
    if out.exists():
        try:
            prev = pd.read_csv(out)
            if not prev.empty:
                last_ts = pd.to_datetime(prev["ts"].iloc[-1], utc=True)
                now = pd.Timestamp.now(tz="UTC")
                if last_ts >= now - pd.Timedelta(days=2):
                    print(f"  {sym}: up to date (last {last_ts.date()}), skip")
                    return 0
                start_dt = last_ts + pd.Timedelta(minutes=5)
                floor_ts = last_ts
        except Exception:
            pass
    if start_dt is None:
        start_dt = pd.Timestamp("2020-01-01", tz="UTC")
    start_ms = int(start_dt.timestamp() * 1000)
    now_ms = int(time.time() * 1000)
    written = 0
    first = not out.exists() or out.stat().st_size == 0  # header written once
    while start_ms < now_ms:
        _mem_guard(mem_limit)
        raw = _fetch_with_fallback(bin_sym, start_ms)
        if not raw:
            # empty window (no trades / data hole): skip one full window and
            # keep going — do NOT break, a gap mid-history isn't the end.
            start_ms += WINDOW_MS
            time.sleep(0.25)
            continue
        df = _raw_to_df(raw)
        if df.empty:
            start_ms += WINDOW_MS
            time.sleep(0.25)
            continue
        # never re-append bars already on disk (prevents duplicate accumulation
        # across resume runs, which previously duplicated ~63% of DOGE).
        if floor_ts is not None:
            df = df[df["ts"] > floor_ts.strftime("%Y-%m-%d %H:%M:%S+0000")]
        if df.empty:
            start_ms += WINDOW_MS
            time.sleep(0.25)
            continue
        # append incrementally
        df.to_csv(out, mode="a", header=first, index=False)
        first = False
        written += len(df)
        # advance by a full 5m bar (not +1ms) so the next window's first bar
        # does not collide with this window's last on whole-second ts formatting
        start_ms = int(pd.to_datetime(df["ts"].iloc[-1], utc=True).timestamp() * 1000) + 300000
        floor_ts = pd.to_datetime(df["ts"].iloc[-1], utc=True)  # track newest written bar
        time.sleep(0.25)
        if written > 1_000_000:
            break
    gc.collect()
    if written:
        print(f"  {sym}: +{written} bars -> {out.name}")
    return written


def _fetch_binance_limit(symbol: str, start_ms: int):
    url = f"{BINANCE}?symbol={symbol}&interval=5m&startTime={start_ms}&limit=1000"
    return _get_json(url)


def _fetch_mexc_limit(symbol: str, start_ms: int):
    url = f"{MEXC}?symbol={symbol}&interval=5m&startTime={start_ms}&limit=1000"
    return _get_json(url)


def _fetch_with_fallback(symbol: str, start_ms: int):
    """Fetch one klines window, trying Binance then MEXC.

    Returns the raw JSON (a list of OHLCV rows), or [] if both sources fail
    (e.g. pair does not exist on either venue). Callers treat [] as a gap to
    skip, not an error.
    """
    try:
        return _fetch_binance_limit(symbol, start_ms)
    except Exception:
        try:
            return _fetch_mexc_limit(symbol, start_ms)
        except Exception:
            return []


def _raw_to_df(raw) -> pd.DataFrame:
    if not raw:
        return pd.DataFrame()
    df = pd.DataFrame(raw, columns=[
        "ts_raw", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trades", "taker_buy_base",
        "taker_buy_quote", "ignore"])
    df = df[["ts_raw", "open", "high", "low", "close", "volume"]].copy()
    df["ts"] = pd.to_datetime(df["ts_raw"], unit="ms", utc=True).dt.strftime("%Y-%m-%d %H:%M:%S+0000")
    df = df[["ts", "open", "high", "low", "close", "volume"]].astype(
        {"open": float, "high": float, "low": float, "close": float, "volume": float})
    return df.drop_duplicates(subset="ts").sort_values("ts")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=None)
    ap.add_argument("--limit", type=int, default=0, help="cap symbols (smoke test)")
    ap.add_argument("--mem-limit-mb", type=int, default=DEFAULT_MEM_LIMIT_MB)
    args = ap.parse_args()

    _mem_guard(args.mem_limit_mb)
    if args.symbol:
        syms = [args.symbol.upper()]
    else:
        u = pd.read_csv(ROOT / "dex_universe.csv")["symbol"].astype(str).str.upper()
        # keep only clean USDT-pairable stems (drop $ and weird suffixes)
        syms = sorted({s.replace("$", "") for s in u if "$" not in s and "USDT" not in s})
    if args.limit:
        syms = syms[:args.limit]

    print(f"CEX 5m backfill (free Binance mirror + MEXC): {len(syms)} symbols")
    total = 0
    for s in syms:
        n = backfill_symbol(s, args.mem_limit_mb)
        total += n
    gc.collect()
    print(f"Backfill complete. Total bars written: {total}")


if __name__ == "__main__":
    main()
