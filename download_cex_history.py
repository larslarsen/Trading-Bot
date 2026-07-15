#!/usr/bin/env python3
"""Download free deep CEX OHLCV from CryptoDataDownload and merge into our local
commercial dataset (data/<SYM>USDT_1d_max.csv).

Why: CDD offers free daily OHLCV from major exchanges SINCE 2017 (deeper than
our bitfinex 2019 backfill) as one-shot CSV downloads -- no API key, no rate
limits. This complements the API backfill (backfill_cex_history.py): API gives
breadth (coins on shallow exchanges), CDD gives deeper history for Binance-listed
coins. Both merge into the SAME files -> one unified commercial dataset.

LICENSE: CryptoDataDownload data is CC-BY-NC-SA 4.0 (non-commercial). We treat
all data here as COMMERCIAL and accept the obligation: if we end up USING CDD
data in a commercial product, buy a Pro license later. Personal research now is
within NC terms; this script just collects it into the unified store.

For each target symbol (broad universe + all PIT screens):
  - try Binance <SYM>USDT_d.csv, fallback Bitstamp/Gemini/Bitfinex/Kraken <SYM>USD_d.csv
  - normalize to ts,open,high,low,close,volume (ts = date; volume = quote volume)
  - merge with existing local file by ts (union, dedup, sort) -> extends history
Resume: skip symbols already <= 2023-01-01. Rate-limit: 1s between downloads.

Usage:
    python download_cex_history.py [--sleep 1.0]
"""
import argparse
import time
from pathlib import Path

import pandas as pd
import urllib.request

ROOT = Path("data")
SCR = Path("backtest_output")
CDD = "https://www.cryptodatadownload.com/cdd"
# (exchange, quote) tried in order; Binance deepest + most symbols.
SOURCES = [("Binance", "USDT"), ("Bitstamp", "USD"), ("Gemini", "USD"),
           ("Bitfinex", "USD"), ("Kraken", "USD")]
DEEP_ENOUGH = pd.Timestamp("2023-01-01")


def target_symbols():
    syms = set()
    bu = ROOT / "universe_broad.csv"
    if bu.exists():
        df = pd.read_csv(bu)
        if "symbol" in df:
            syms |= set(df["symbol"].dropna().astype(str).str.strip().str.upper())
    for f in SCR.glob("screen_liqu_idio_*.csv"):
        df = pd.read_csv(f)
        if "symbol" in df:
            syms |= set(df["symbol"].dropna().astype(str).str.strip().str.upper())
    return sorted(s for s in syms if s)


def fetch_cdd(sym, ex, quote, sleep):
    url = f"{CDD}/{ex}_{sym}{quote}_d.csv"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as r:
            # CDD CSVs have a title row ("https://www.CryptoDataDownload.com")
            # before the real header -> skip it.
            df = pd.read_csv(r, skiprows=1)
    except Exception:
        time.sleep(sleep)
        return None
    try:
        # normalize columns (defensive: lowercase everything first; CDD mixes
        # cases across exchanges -- Binance=Unix/Date, others=unix/date)
        df = df.rename(columns={c: str(c).lower() for c in df.columns})
        df = df.rename(columns={"date": "ts", "open": "open", "high": "high",
                                "low": "low", "close": "close"})
        if "ts" not in df.columns and "unix" in df.columns:
            df["ts"] = pd.to_datetime(df["unix"].astype("int64"), unit="ms").dt.strftime("%Y-%m-%d")
        df["ts"] = pd.to_datetime(df.get("ts"), errors="coerce").dt.strftime("%Y-%m-%d")
        # volume: prefer a quote-volume column (usdt/usd), else any 'volume*'
        vol_col = None
        for c in df.columns:
            if str(c).startswith("volume") and any(q in str(c) for q in ("usdt", "usd")):
                vol_col = c; break
        if vol_col is None:
            vols = [c for c in df.columns if str(c).startswith("volume")]
            vol_col = vols[0] if vols else None
        df["volume"] = df[vol_col] if vol_col else 0.0
        out = df[["ts", "open", "high", "low", "close", "volume"]].dropna(subset=["close"])
        if len(out) == 0:
            return None
        return out
    except Exception:
        time.sleep(sleep)
        return None


def merge_symbol(sym, sleep):
    out = ROOT / f"{sym}USDT_1d_max.csv"
    existing = None
    for q in ("USDT", "USD", "USDC"):
        p = ROOT / f"{sym}{q}_1d_max.csv"
        if p.exists():
            existing = pd.read_csv(p)
            break
    frames = [existing] if existing is not None else []
    got = False
    for ex, quote in SOURCES:
        df = fetch_cdd(sym, ex, quote, sleep)
        if df is not None and len(df):
            frames.append(df)
            got = True
        time.sleep(sleep)
    if not got:
        return None
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(subset="ts").sort_values("ts")
    merged.to_csv(out, index=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sleep", type=float, default=1.0)
    args = ap.parse_args()
    syms = target_symbols()
    print(f"CDD history download: {len(syms)} symbols -> data/ (unified commercial store)")
    done = skipped = 0
    for i, sym in enumerate(syms, 1):
        out = ROOT / f"{sym}USDT_1d_max.csv"
        if out.exists():
            try:
                if pd.Timestamp(pd.read_csv(out)["ts"].min()) <= DEEP_ENOUGH:
                    skipped += 1
                    continue
            except Exception:
                pass
        res = merge_symbol(sym, args.sleep)
        if res:
            done += 1
            print(f"[{i}/{len(syms)}] {sym}: -> {res.name}")
        else:
            print(f"[{i}/{len(syms)}] {sym}: not on CDD")
    print(f"\nDownload complete: {done} merged, {skipped} already deep (skipped).")


if __name__ == "__main__":
    main()
