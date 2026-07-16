#!/usr/bin/env python3
"""
MASTER DATA MANIFEST — ONE place to see ALL free data we hold.

Scans data/ + dex_data/ + data/onchain/ + data/funding/ + data/dex_history/
and reports, per source, what symbols/TFs we have and their date spans.

This is the single source of truth for "what data do we have and where".
Run: python data_manifest.py
"""
from pathlib import Path
import pandas as pd
import glob
import re

REPO = Path(__file__).parent
OUT = []

def span(f):
    try:
        # cheap: read only head + tail (avoid loading full file)
        head = pd.read_csv(f, nrows=1)
        if "ts" in head.columns:
            col = "ts"
        elif "date" in head.columns:
            col = "date"
        else:
            return None, None, 0
        total = sum(1 for _ in open(f)) - 1  # minus header
        # tail: read last 2 rows
        skip = max(0, total - 1)
        tail = pd.read_csv(f, skiprows=range(1, skip + 1)) if skip else head
        first = pd.to_datetime(head[col].iloc[0], utc=True, errors="coerce")
        last = pd.to_datetime(tail[col].iloc[-1], utc=True, errors="coerce")
        return (first.date() if pd.notna(first) else None,
                last.date() if pd.notna(last) else None, total)
    except Exception:
        return None, None, 0

def add(source, venue, tf, f):
    s, e, n = span(f)
    OUT.append({
        "source": source, "venue": venue, "tf": tf,
        "symbol": f.stem.replace(f"_{tf}_max", "").replace(f"_{tf}", ""),
        "rows": n, "start": str(s) if s else "?", "end": str(e) if e else "?",
        "path": str(f.relative_to(REPO)),
    })

# CEX 5m (Binance bulk = unsuffixed; others suffixed)
for f in glob.glob(str(REPO / "data" / "*USDT_5m_max.csv")):
    add("CEX", "binance", "5m", Path(f))
for v in ["okx", "bybit", "gateio", "kucoin", "bitget", "coinbase", "mexc"]:
    for f in glob.glob(str(REPO / "data" / f"*_5m_{v}_max.csv")):
        add("CEX", v, "5m", Path(f))

# Funding (Bybit deep)
for f in glob.glob(str(REPO / "data" / "funding" / "*_funding.csv")):
    add("FUNDING", "bybit", "live", Path(f))

# On-chain (AWS S3)
for f in glob.glob(str(REPO / "data" / "onchain" / "*_features_daily.csv")):
    add("ONCHAIN", "aws-s3", "1d", Path(f))

# DEX 1m live (daemon) + derived
for f in glob.glob(str(REPO / "data" / "dex" / "*_1m_max.csv")):
    add("DEX", "geckoterminal", "1m", Path(f))
for f in glob.glob(str(REPO / "data" / "dex" / "*_1h_max.csv")):
    add("DEX", "geckoterminal", "1h", Path(f))
for f in glob.glob(str(REPO / "data" / "dex" / "*_4h_max.csv")):
    add("DEX", "geckoterminal", "4h", Path(f))
for f in glob.glob(str(REPO / "data" / "dex" / "*_1d_max.csv")):
    add("DEX", "geckoterminal", "1d", Path(f))

# DEX day history backfill
for f in glob.glob(str(REPO / "data" / "dex_history" / "*.csv")):
    add("DEX-HIST", "geckoterminal", "1d", Path(f))

# DEX universe legacy 1d
for f in glob.glob(str(REPO / "dex_data" / "*_1d_max.csv")):
    add("DEX-UNIV", "dexscreener", "1d", Path(f))

# Kraken (gdrive, throttled)
for f in glob.glob(str(REPO / "data" / "*_kraken_max.csv")):
    p = Path(f)
    add("CEX", "kraken", p.stem.split("_")[-2] if "_" in p.stem else "?", p)

df = pd.DataFrame(OUT)
pd.set_option("display.max_rows", 200)
pd.set_option("display.width", 200)
print(f"=== MASTER DATA MANIFEST ({len(df)} files) ===")
print(df.sort_values(["source", "venue", "tf", "symbol"]).to_string(index=False))
# summary by source/venue
print("\n=== SUMMARY (files per source/venue) ===")
print(df.groupby(["source", "venue"]).size().to_string())
