#!/usr/bin/env python3
"""
Kraken historical OHLCVT + time-and-sales backfill (FREE, NO API KEY).

Kraken publishes per-quarter OHLCVT (and time-and-sales) ZIPs in a public
Google Drive folder. No account, no key. We pull them with gdown, handling
Google's "too many users downloaded this recently" per-file throttle via
retry+backoff (one file at a time, resumes on the folder index).

Kraken's native 1-minute bars are the finest free granularity we have for a
second CEX venue -> resample to 5m/1h/4h/1d locally like the Binance data.

Usage:  python backfill_kraken_hist.py [--workers 1] [--max-retry 8]
Writes: data/kraken/<STEM>USDT_5m_max.csv  (stem = XBT->BTC, etc.)
        data/kraken/<STEM>USDT_trades_max.csv  (time-and-sales, if present)
"""
import argparse
import io
import re
import time
import zipfile
from pathlib import Path

import pandas as pd

import gdown

REPO = Path(__file__).parent
OUT = REPO / "data" / "kraken"
FOLDER = "https://drive.google.com/drive/folders/15RSlNuW_h0kVM8or8McOGOMfHeBFvFGI"

# Kraken uses XBT not BTC; normalize stems to our BTC convention.
STEM_MAP = {"XBT": "BTC"}


def norm_stem(stem):
    return STEM_MAP.get(stem, stem)


def list_zips():
    """Return list of (gdrive_file_id, name) for OHLCVT + TnS zips in the folder.
    gdown.download_folder(skip_download=True) returns GoogleDriveFileToDownload
    objects (attrs: id, index, path). Name isn't on the object, so we keep the
    id and a placeholder; the real filename comes from the zip's inner CSV."""
    try:
        metas = gdown.download_folder(url=FOLDER, output=str(OUT / "_index_tmp"),
                                      quiet=True, skip_download=True)
    except Exception as e:
        print(f"  folder index failed: {e!r}", flush=True)
        return []
    out = []
    for m in metas or []:
        fid = getattr(m, "id", None)
        # best-effort name from the local_path gdown would use
        nm = getattr(m, "path", "") or getattr(m, "local_path", "") or f"{fid}.zip"
        nm = nm.rsplit("/", 1)[-1]
        if "OHLCVT" in nm.upper() or "time" in nm.lower() or "sales" in nm.lower() or nm.endswith(".zip"):
            out.append((fid, nm))
    return out


def unzip_to_csv(zbytes, stem, kind):
    """Kraken ZIP contains one CSV. Parse, normalize columns, resample to 5m."""
    z = zipfile.ZipFile(io.BytesIO(zbytes))
    name = z.namelist()[0]
    df = pd.read_csv(io.BytesIO(z.read(name)))
    # Kraken OHLCVT cols: pair,time,open,high,low,close,volume,vwap,count
    # (time is unix seconds). time-and-sales: pair,time,price,volume,side,type,order
    pair = df.iloc[0, 0] if "pair" in df.columns else stem
    if kind == "ohlcvt":
        out = df[["time", "open", "high", "low", "close", "volume"]].copy()
        out["ts"] = pd.to_datetime(out["time"], unit="s", utc=True)
        out = out.drop(columns=["time"]).set_index("ts").sort_index()
        out_5m = out.resample("5min").agg({
            "open": "first", "high": "max", "low": "min",
            "close": "last", "volume": "sum"}).dropna()
        return out_5m
    else:  # trades / time-and-sales
        out = df[["time", "price", "volume"]].copy()
        out["ts"] = pd.to_datetime(out["time"], unit="s", utc=True)
        return out.set_index("ts").sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-retry", type=int, default=8)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    zips = list_zips()
    print(f"Kraken folder: {len(zips)} zip files found", flush=True)

    # aggregate per stem across quarters
    by_stem = {}
    for fid, nm in zips:
        m = re.match(r"Kraken_(OHLCVT|TimeAndSales)_Q(\d)_(\d{4})", nm)
        if not m:
            continue
        kind = "ohlcvt" if m.group(1) == "OHLCVT" else "trades"
        stem = norm_stem(m.group(2))
        by_stem.setdefault((stem, kind), []).append((fid, nm))

    done = 0
    for (stem, kind), files in by_stem.items():
        target = OUT / f"{stem}USDT_{'5m' if kind=='ohlcvt' else 'trades'}_max.csv"
        frames = []
        for fid, nm in files:
            for attempt in range(args.max_retry):
                try:
                    path = gdown.download(id=fid, output=str(OUT / "_tmp.zip"),
                                           quiet=True, use_cookies=False)
                    with open(path, "rb") as fh:
                        zbytes = fh.read()
                    frames.append(unzip_to_csv(zbytes, stem, kind))
                    (OUT / "_tmp.zip").unlink(missing_ok=True)
                    break
                except Exception as e:
                    msg = str(e)
                    if "Too many users" in msg or "FileURLRetrieval" in msg:
                        wait = min(2 ** attempt * 30, 300)
                        print(f"  {nm}: throttled, retry {attempt+1} in {wait}s", flush=True)
                        time.sleep(wait)
                    else:
                        print(f"  {nm}: ERR {msg[:120]}", flush=True)
                        break
        if frames:
            big = pd.concat(frames).sort_index()
            big = big[~big.index.duplicated(keep="last")]
            big.to_csv(target, index=True)
            done += 1
            print(f"  wrote {target.name}: {len(big):,} rows", flush=True)
    print(f"DONE: {done} Kraken series written to {OUT}", flush=True)


if __name__ == "__main__":
    main()
