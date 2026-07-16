#!/usr/bin/env python3
"""
Two more free, no-key data sources:

1. Bybit funding RATE history (perp carry / sentiment) for major perp symbols.
   /v5/market/funding/history  -> [{symbol, fundingRate, fundingRateTimestamp}]
   Paginated via startTime/endTime, 8h funding interval, limit<=200.
   Writes data/funding/<SYM>USDT_funding.csv (ts-indexed, rate + interval-hours).

2. MEXC 5m klines (5th CEX venue, strong on low-cap/retail alts).
   /api/v3/klines  -> [openTime,o,h,l,c,vol,closeTime,quoteVol]
   Paginated via startTime/endTime, limit<=1000. Walks forward.
   Writes data/<SYM>USDT_5m_max.csv (matches ONE-place convention).

Both resumable via state (last ts per symbol). Rate-limit safe (curl + backoff).

Usage:
  python backfill_funding_mexc.py funding [--symbols BTCUSDT,ETHUSDT,...]
  python backfill_funding_mexc.py mexc    [--symbols BTCUSDT,ETHUSDT,...]
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO = Path(__file__).parent
DATA = REPO / "data"
UA = {"User-Agent": "Mozilla/5.0 (research backfill)"}


def fetch_json(url, timeout=30, tries=4):
    for i in range(tries):
        try:
            out = subprocess.run(["curl", "-s", "--max-time", str(timeout),
                                  "-A", UA["User-Agent"], url],
                                 capture_output=True, text=True, timeout=timeout + 5)
            if out.returncode != 0:
                time.sleep(min(2 ** i * 2, 30)); continue
            txt = out.stdout
            if not txt.strip():
                time.sleep(min(2 ** i * 2, 30)); continue
            return json.loads(txt), None
        except subprocess.TimeoutExpired:
            time.sleep(min(2 ** i * 5, 30)); continue
        except Exception as e:
            return None, str(e)[:120]
    return None, "exhausted retries"


# ---------------- Bybit funding rate history ----------------
def bybit_funding(symbol, start_ms, end_ms, limit=200, win_ms=None):
    """Walk funding history forward in bounded windows (Bybit requires BOTH
    startTime and endTime). 8h funding interval. Returns [[ts_ms, rate], ...]."""
    if win_ms is None:
        win_ms = 365 * 2 * 86400 * 1000  # 2y windows
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        win_end = min(cursor + win_ms, end_ms)
        u = (f"https://api.bybit.com/v5/market/funding/history?category=linear"
             f"&symbol={symbol}&limit={limit}&startTime={cursor}&endTime={win_end}")
        d, err = fetch_json(u)
        if err:
            return rows, err
        if d.get("retCode") != 0:
            return rows, d.get("retMsg")
        lst = d.get("result", {}).get("list", [])
        if not lst:
            # no data in this window; advance
            cursor = win_end + 1
            time.sleep(0.2)
            continue
        for r in lst:
            t = int(r["fundingRateTimestamp"])
            if t >= start_ms:
                rows.append([t, float(r["fundingRate"])])
        last = int(lst[-1]["fundingRateTimestamp"])
        cursor = last + 1
        time.sleep(0.3)
    return rows, None


def run_funding(symbols):
    outdir = DATA / "funding"
    outdir.mkdir(parents=True, exist_ok=True)
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    # Bybit perp funding started ~2019; default deep start
    start_default = int(pd.Timestamp("2019-01-01", tz="UTC").timestamp() * 1000)
    for sym in symbols:
        tgt = outdir / f"{sym}_funding.csv"
        cursor = start_default
        if tgt.exists():
            old = pd.read_csv(tgt, parse_dates=["ts"])
            if len(old):
                cursor = int(old["ts"].max().timestamp() * 1000) + 1
        print(f"[funding] {sym}: from {pd.to_datetime(cursor,unit='ms',utc=True)}", flush=True)
        rows, err = bybit_funding(sym, cursor, end_ms)
        if err:
            print(f"  ERR {err}", flush=True); continue
        if not rows:
            print(f"  no new rows", flush=True); continue
        df = pd.DataFrame(rows, columns=["ts", "funding_rate"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        if tgt.exists():
            old = pd.read_csv(tgt, parse_dates=["ts"]).set_index("ts")
            df = pd.concat([old, df.set_index("ts")]).sort_index()
            df = df[~df.index.duplicated(keep="last")]
        else:
            df = df.set_index("ts")
        df["interval_hours"] = 8
        df.to_csv(tgt)
        print(f"  wrote {len(df)} rows ({df.index[0]} .. {df.index[-1]})", flush=True)
    print("funding backfill done.", flush=True)


# ---------------- MEXC 5m klines ----------------
def mexc_klines(symbol, start_ms, end_ms=None, limit=1000, win_ms=None):
    """MEXC requires BOTH startTime and endTime (startTime alone is ignored and
    returns recent bars). Walk forward in bounded windows (default 90d)."""
    if end_ms is None:
        end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    if win_ms is None:
        win_ms = 90 * 86400 * 1000
    rows = []
    cursor = start_ms
    while cursor < end_ms:
        win_end = min(cursor + win_ms, end_ms)
        u = (f"https://api.mexc.com/api/v3/klines?symbol={symbol}&interval=5m"
             f"&limit={limit}&startTime={cursor}&endTime={win_end}")
        out, err = fetch_json(u)
        if err:
            return rows, err
        if not out or not isinstance(out, list):
            return rows, "bad response"
        if not out:
            cursor = win_end + 1
            time.sleep(0.2)
            continue
        for b in out:
            rows.append([int(b[0]), float(b[1]), float(b[2]),
                         float(b[3]), float(b[4]), float(b[5])])
        last = int(out[-1][0])
        cursor = last + 1
        time.sleep(0.3)
    return rows, None


def run_mexc(symbols):
    start_ms = int(pd.Timestamp("2021-01-01", tz="UTC").timestamp() * 1000)
    for sym in symbols:
        tgt = DATA / f"{sym}_5m_max.csv"
        cursor = start_ms
        if tgt.exists():
            old = pd.read_csv(tgt, parse_dates=["ts"])
            if len(old):
                cursor = int(old["ts"].max().timestamp() * 1000) + 1
        print(f"[mexc] {sym}: from {pd.to_datetime(cursor,unit='ms',utc=True)}", flush=True)
        rows, err = mexc_klines(sym, cursor)
        if err:
            print(f"  ERR {err}", flush=True); continue
        if not rows:
            print(f"  no new bars", flush=True); continue
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
        if tgt.exists():
            old = pd.read_csv(tgt, parse_dates=["ts"]).set_index("ts")
            df = pd.concat([old, df.set_index("ts")]).sort_index()
            df = df[~df.index.duplicated(keep="last")]
        else:
            df = df.set_index("ts")
        df.to_csv(tgt)
        print(f"  wrote {len(df)} rows ({df.index[0]} .. {df.index[-1]})", flush=True)
    print("mexc backfill done.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["funding", "mexc"])
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT,MATICUSDT")
    args = ap.parse_args()
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if args.mode == "funding":
        run_funding(syms)
    else:
        run_mexc(syms)


if __name__ == "__main__":
    main()
