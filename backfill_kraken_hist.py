#!/usr/bin/env python3
"""
Kraken historical OHLCVT backfill (FREE, NO API KEY).

Kraken publishes per-quarter OHLCVT ZIPs (1-min bars, ALL pairs) in a public
Google Drive folder. No account, no key. Each quarter ZIP's inner CSV has a
'pair' column covering every Kraken pair that quarter:

    pair,time,open,high,low,close,volume,vwap,count

We split by pair, normalize to our USDT convention (XBT->BTC, XXXUSD->XXXUSDT),
resample 1m->5m, and append to data/kraken/<SYM>USDT_5m_max.csv.

GDrive throttles shared folders ("too many users downloaded recently") -- this
can last hours. So this is a PERSISTENT retry loop: state file tracks which
quarters landed, it retries failed quarters with exponential backoff, resleeps,
and rescans until everything is down or --max-wall-h is exceeded. It survives
both the throttle and process restarts. "All the data, all the time."

Usage:  python backfill_kraken_hist.py [--max-retry 40] [--max-wall-h 168]
"""
import argparse
import io
import json
import re
import time
import zipfile
from pathlib import Path

import pandas as pd
import gdown

REPO = Path(__file__).parent
OUT = REPO / "data" / "kraken"
STATE = OUT / ".kraken_state.json"
FOLDER = "https://drive.google.com/drive/folders/15RSlNuW_h0kVM8or8McOGOMfHeBFvFGI"
STEM_MAP = {"XBT": "BTC"}


def pair_to_stem(pair):
    """XBTUSD->BTCUSDT, ETHUSD->ETHUSDT, DOGEUSD->DOGEUSDT; skip non-USD."""
    p = pair.upper()
    if p.endswith("USD"):
        base = p[:-3]
    elif p.endswith("USDT"):
        base = p[:-4]
    else:
        return None  # non-USD pair; we mirror the Binance USDT universe
    return STEM_MAP.get(base, base) + "USDT"


def list_zips():
    """(gdrive_file_id, name) for OHLCVT/TnS zips in the public folder."""
    try:
        metas = gdown.download_folder(url=FOLDER, output=str(OUT / "_idx"),
                                      quiet=True, skip_download=True)
    except Exception as e:
        print(f"  folder index failed: {e!r}", flush=True)
        return []
    out = []
    for m in metas or []:
        fid = getattr(m, "id", None)
        nm = getattr(m, "path", "") or getattr(m, "local_path", "") or f"{fid}.zip"
        nm = nm.rsplit("/", 1)[-1]
        if nm.endswith(".zip") and ("OHLCVT" in nm.upper() or "time" in nm.lower()
                                    or "sales" in nm.lower()):
            out.append((fid, nm))
    return out


def parse_quarter(zbytes):
    """Yield (stem, 5m_dataframe) for each USDT pair in the quarter ZIP."""
    z = zipfile.ZipFile(io.BytesIO(zbytes))
    name = z.namelist()[0]
    df = pd.read_csv(io.BytesIO(z.read(name)))
    df["ts"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.drop(columns=["time"])
    res = {}
    for pair, g in df.groupby("pair"):
        stem = pair_to_stem(pair)
        if stem is None:
            continue
        g5 = (g.set_index("ts")[["open", "high", "low", "close", "volume"]]
              .sort_index()
              .resample("5min")
              .agg({"open": "first", "high": "max", "low": "min",
                    "close": "last", "volume": "sum"})
              .dropna())
        if len(g5):
            res[stem] = g5
    return res


def load_state():
    try:
        return set(json.loads(STATE.read_text()).get("done", []))
    except Exception:
        return set()


def save_state(done):
    STATE.write_text(json.dumps({"done": sorted(done)}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-retry", type=int, default=40)
    ap.add_argument("--max-wall-h", type=float, default=168.0)
    ap.add_argument("--loop-sleep", type=int, default=300)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    done = load_state()
    started = time.time()
    print(f"Kraken backfill: persistent loop, max-wall={args.max_wall_h}h", flush=True)
    while True:
        zips = list_zips()
        pending = [(fid, nm) for fid, nm in zips if nm not in done]
        if not pending:
            print("ALL Kraken quarters landed.", flush=True)
            break
        print(f"  {len(pending)} quarters pending (done={len(done)})", flush=True)
        for fid, nm in pending:
            for attempt in range(args.max_retry):
                try:
                    p = gdown.download(id=fid, output=str(OUT / "_t.zip"),
                                       quiet=True, use_cookies=False)
                    zbytes = open(p, "rb").read()
                    (OUT / "_t.zip").unlink(missing_ok=True)
                    pairs = parse_quarter(zbytes)
                    for stem, g5 in pairs.items():
                        tgt = OUT / f"{stem}_5m_max.csv"
                        if tgt.exists():
                            old = pd.read_csv(tgt, parse_dates=["ts"]).set_index("ts")
                            g5 = pd.concat([old, g5]).sort_index()
                            g5 = g5[~g5.index.duplicated(keep="last")]
                        g5.to_csv(tgt)
                    done.add(nm)
                    save_state(done)
                    print(f"  DONE {nm}: {len(pairs)} pairs written", flush=True)
                    break
                except Exception as e:
                    msg = str(e)
                    if "Too many users" in msg or "FileURLRetrieval" in msg:
                        w = min(2 ** attempt * 30, 600)
                        print(f"  {nm}: throttle retry {attempt+1}, wait {w}s", flush=True)
                        time.sleep(w)
                    else:
                        print(f"  {nm}: ERR {msg[:150]}", flush=True)
                        break
        if time.time() - started > args.max_wall_h * 3600:
            print("max-wall reached; state saved, rerun to continue.", flush=True)
            break
        remaining = [nm for _, nm in list_zips() if nm not in done]
        if remaining:
            print(f"  {len(remaining)} still pending; rescan in {args.loop_sleep}s", flush=True)
            time.sleep(args.loop_sleep)
        else:
            break
    print("Kraken loop exit.", flush=True)


if __name__ == "__main__":
    main()
