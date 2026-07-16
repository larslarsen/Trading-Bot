#!/usr/bin/env python3
"""
Free CEX 5m backfill for ALL no-key venues: binance(bybit/okx/gate/kucoin/
bitget/coinbase). Cross-venue confirmation for ML. No API key anywhere.

Each venue writes SUFFIXED files: data/<SYM>_5m_<venue>_max.csv
(so they never clobber each other or the Binance bulk file). Resumable
(via state: skips if local file already reaches ~now).

Timeframes derived locally from 5m (exact) per the ONE-place rule.

Rate limits: each venue is a SEPARATE host from GeckoTerminal, so this does
NOT compete with the DEX 1m daemon. Within CEX, pace with --sleep.

Usage:
  python backfill_cex_all.py --venue okx --sleep 1.0
  python backfill_cex_all.py --venue gateio --symbols BTCUSDT,ETHUSDT
  python backfill_cex_all.py --venue all --sleep 1.5   # loops venues
"""
import argparse
import time
import subprocess
import json
import urllib.parse
from pathlib import Path

import pandas as pd

REPO = Path(__file__).parent
OUT = REPO / "data"
UA = {"User-Agent": "Mozilla/5.0 (research backfill)"}
SYMS = "BTCUSDT,ETHUSDT,DOGEUSDT,SOLUSDT,BNBUSDT,XRPUSDT,ADAUSDT,AVAXUSDT,LINKUSDT,MATICUSDT"


def curl_json(url, timeout=20, tries=4):
    for i in range(tries):
        try:
            out = subprocess.run(["curl", "-s", "--max-time", str(timeout),
                                  "-A", UA["User-Agent"], url],
                                 capture_output=True, text=True, timeout=timeout + 5)
            if out.returncode != 0:
                time.sleep(min(2 ** i * 2, 30)); continue
            if not out.stdout.strip():
                time.sleep(min(2 ** i * 2, 30)); continue
            return json.loads(out.stdout), None
        except Exception as e:
            return None, str(e)[:120]
    return None, "exhausted retries"


def api(url, timeout=20, tries=4):
    """curl_json that tolerates non-JSON (Coinbase returns raw JSON array)."""
    for i in range(tries):
        try:
            out = subprocess.run(["curl", "-s", "--max-time", str(timeout),
                                  "-A", UA["User-Agent"], url],
                                 capture_output=True, text=True, timeout=timeout + 5)
            if out.returncode != 0 or not out.stdout.strip():
                time.sleep(min(2 ** i * 2, 30)); continue
            return json.loads(out.stdout), None
        except Exception as e:
            return None, str(e)[:120]
    return None, "exhausted retries"


def norm(rows, venue, ts_unit="s"):
    """rows -> df[ts,open,high,low,close,volume] (5m)."""
    out = []
    for r in rows:
        try:
            if venue == "gateio":
                # [ts_s, vol, o, h, l, c, quote_vol, completed]
                ts, o, h, l, c, v = int(r[0]), float(r[2]), float(r[3]), float(r[4]), float(r[5]), float(r[1])
            elif venue == "kucoin":
                # [ts_s, o, h, l, c, vol, quote_vol]
                ts, o, h, l, c, v = int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])
            elif venue == "bitget":
                # [ts_ms, o, h, l, c, vol, quote_vol]
                ts, o, h, l, c, v = int(r[0]) // 1000, float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])
            elif venue == "coinbase":
                # [ts_s, o, h, l, c, vol]
                ts, o, h, l, c, v = int(r[0]), float(r[1]), float(r[2]), float(r[3]), float(r[4]), float(r[5])
            else:
                continue
            out.append([pd.to_datetime(ts, unit="s", utc=True), o, h, l, c, v])
        except Exception:
            continue
    return pd.DataFrame(out, columns=["ts", "open", "high", "low", "close", "volume"]).drop_duplicates("ts").sort_values("ts")


def fetch_pages(venue, sym, limit=1000):
    """Walk full 5m history per venue. Returns list of row-lists."""
    rows = []
    if venue == "gateio":
        # newest-first; after=ts_s cursor, walk back
        pair = sym.replace("USDT", "_USDT")
        after = int(pd.Timestamp.now(tz="UTC").timestamp())
        while True:
            u = f"https://api.gateio.ws/api/v4/spot/candlesticks?currency_pair={pair}&interval=5m&limit={min(limit,1000)}&after={after}"
            d, e = api(u)
            if e or not isinstance(d, list) or not d:
                break
            rows.extend(d)
            oldest = int(d[-1][0])
            if oldest >= after:
                break
            after = oldest
            time.sleep(0.3)
    elif venue == "kucoin":
        # walk backward in ~300d chunks (KuCoin needs real start/end range)
        end = int(pd.Timestamp.now(tz="UTC").timestamp())
        start0 = int(pd.Timestamp("2021-01-01", tz="UTC").timestamp())
        cur_end = end
        while cur_end > start0:
            cur_start = max(start0, cur_end - 300 * 86400)
            u = f"https://api.kucoin.com/api/v1/market/candles?symbol={sym[:-4]}-USDT&type=5min&startAt={cur_start}&endAt={cur_end}"
            d, e = api(u)
            if e or not d or d.get("code") != "200000":
                break
            page = d["data"]
            if page:
                rows.extend(page)
            cur_end = cur_start
            time.sleep(0.3)
    elif venue == "bitget":
        # after=ts_ms (exclusive upper bound), walk backward; start floor 2021
        sym_b = sym
        after = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        floor = int(pd.Timestamp("2021-01-01", tz="UTC").timestamp() * 1000)
        while after > floor:
            u = f"https://api.bitget.com/api/v2/spot/market/candles?symbol={sym_b}&granularity=5min&limit={min(limit,1000)}&after={after}"
            d, e = api(u)
            if e or not isinstance(d, dict) or d.get("code") != "00000":
                break
            page = d.get("data", [])
            if not page:
                break
            rows.extend(page)
            oldest = int(page[-1][0])
            if oldest <= floor or oldest >= after:
                break
            after = oldest
            time.sleep(0.3)
    elif venue == "coinbase":
        # Coinbase public candles cap at 300 candles/request (= 25h at 5m).
        # Walk backward in ~24h chunks.
        end = pd.Timestamp.now(tz="UTC")
        start0 = pd.Timestamp("2021-01-01", tz="UTC")
        while end > start0:
            chunk_start = max(start0, end - pd.Timedelta(hours=24))
            u = f"https://api.exchange.coinbase.com/products/{sym[:-4]}-USDT/candles?granularity=300&start={chunk_start.isoformat()}&end={end.isoformat()}"
            d, e = api(u)
            if e or not isinstance(d, list) or not d:
                time.sleep(1.0)
                break
            rows.extend(d)
            oldest = pd.to_datetime(int(d[-1][0]), unit="s", utc=True)
            if oldest <= chunk_start:
                break
            end = oldest
            time.sleep(0.3)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", required=True, choices=["okx", "bybit", "gateio", "kucoin", "bitget", "coinbase", "all"])
    ap.add_argument("--symbols", default=SYMS)
    ap.add_argument("--sleep", type=float, default=1.0)
    ap.add_argument("--start", default="2021-01-01")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    venues = ["okx", "bybit", "gateio", "kucoin", "bitget", "coinbase"] if args.venue == "all" else [args.venue]
    syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]

    for venue in venues:
        if venue in ("okx", "bybit"):
            # delegate to existing backfill_cex_others.py (fix suffix there)
            import subprocess as sp
            print(f"[delegating {venue} to backfill_cex_others.py]", flush=True)
            sp.run(["python", "backfill_cex_others.py", "--venue", venue,
                    "--symbols", args.symbols, "--start", args.start], check=False)
            continue
        for sym in syms:
            tgt = OUT / f"{sym}_5m_{venue}_max.csv"
            # resume: skip if file already reaches ~now (within 10m)
            if tgt.exists():
                old = pd.read_csv(tgt, parse_dates=["ts"])
                if len(old) and (pd.Timestamp.now(tz="UTC") - old["ts"].max()).total_seconds() < 600:
                    print(f"  skip {sym} {venue} (up to date)", flush=True)
                    continue
            print(f"[{venue}] {sym}: fetching 5m...", flush=True)
            rows = fetch_pages(venue, sym)
            if not rows:
                print(f"  {sym} {venue}: no data", flush=True)
                time.sleep(args.sleep); continue
            df = norm(rows, venue)
            if tgt.exists():
                old = pd.read_csv(tgt, parse_dates=["ts"]).set_index("ts")
                df = pd.concat([old, df.set_index("ts")]).sort_index()
                df = df[~df.index.duplicated(keep="last")]
            else:
                df = df.set_index("ts")
            df.to_csv(tgt)
            print(f"  wrote {len(df)} rows ({df.index[0]} .. {df.index[-1]})", flush=True)
            time.sleep(args.sleep)
        print(f"{venue} done.", flush=True)


if __name__ == "__main__":
    main()
