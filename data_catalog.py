#!/usr/bin/env python3
"""Single source of truth for 'what data we have'.

Registers every data file with metadata (kind, symbol, tf, source, first/last
ts, rows, bytes) and persists to MANIFEST.json so we always know what we have
and where. New data lands in the canonical data/ subtrees; existing files
(BTC/DOGE/DEX) are registered in place (not moved) so nothing breaks.

Layout (NEW data):
  data/cex/<SYM>_<tf>.csv        CEX OHLCV (tf in 5m/1h/4h/1d)
  data/dex/<TOKEN>.csv           DEX 1d price bars
  data/dex_micro/<TOKEN>.csv     DEX microstructure time-series (DexScreener/Graph)
  data/micro/<source>.csv        CEX microstructure (Bybit etc.)
  data/macro/*.csv               macro ETFs (SPY/GLD/UUP)
"""
from pathlib import Path
import json
import pandas as pd

REPO = Path(__file__).parent
MANIFEST = REPO / "MANIFEST.json"

CEX_DIR = REPO / "data" / "cex"
DEX_DIR = REPO / "data" / "dex"
DEX_MICRO_DIR = REPO / "data" / "dex_micro"
MICRO_DIR = REPO / "data" / "micro"
MACRO_DIR = REPO / "data" / "macro"
for d in (CEX_DIR, DEX_DIR, DEX_MICRO_DIR, MICRO_DIR, MACRO_DIR):
    d.mkdir(parents=True, exist_ok=True)


def _load():
    if MANIFEST.exists():
        try:
            return json.loads(MANIFEST.read_text())
        except Exception:
            return {"files": {}}
    return {"files": {}}


def _save(db):
    MANIFEST.write_text(json.dumps(db, indent=2))


def register(path, kind, symbol, tf, source, first=None, last=None, rows=None):
    p = Path(path)
    if not p.is_absolute():
        p = REPO / p
    db = _load()
    key = str(p.relative_to(REPO))
    if rows is None and p.exists():
        try:
            rows = sum(1 for _ in open(p)) - 1
        except Exception:
            rows = None
    db["files"][key] = {
        "kind": kind, "symbol": symbol, "tf": tf, "source": source,
        "first": first, "last": last, "rows": rows,
        "bytes": p.stat().st_size if p.exists() else None,
        "mtime": p.stat().st_mtime if p.exists() else None,
    }
    _save(db)
    return key


def register_csv_auto(path, kind, symbol, tf, source):
    p = Path(path)
    if not p.exists():
        return register(p, kind, symbol, tf, source)
    try:
        d = pd.read_csv(p, usecols=["ts"], nrows=300000)
        first = str(d["ts"].min())
        last = str(d["ts"].max())
    except Exception:
        first = last = None
    return register(p, kind, symbol, tf, source, first, last)


def query(**filters):
    db = _load()
    out = []
    for k, v in db["files"].items():
        if all(v.get(fk) == fv for fk, fv in filters.items()):
            out.append((k, v))
    return out


def summary():
    db = _load()
    files = db.get("files", {})
    by_kind = {}
    for v in files.values():
        by_kind[v["kind"]] = by_kind.get(v["kind"], 0) + 1
    return {"total_files": len(files), "by_kind": by_kind}


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "scan":
        # Auto-register known locations without moving anything.
        # CEX 5m (new tree)
        for p in CEX_DIR.glob("*_5m.csv"):
            sym = p.name.replace("_5m.csv", "")
            register_csv_auto(p, "cex", sym, "5m", "binance")
        # DEX 1d
        for p in DEX_DIR.glob("*.csv"):
            register_csv_auto(p, "dex", p.stem, "1d", "dexscreener")
        # DEX micro
        for p in DEX_MICRO_DIR.glob("*.csv"):
            register_csv_auto(p, "dex_micro", p.stem, "poll", "dexscreener")
        # CEX micro (Bybit)
        for nm, sym in [("funding_history", "BTC"), ("oi_history", "BTC"),
                        ("liquidations_history", "BTC"), ("trade_agg_5m", "BTC"),
                        ("orderbook_5m", "BTC")]:
            p = REPO / f"{nm}.csv"
            if p.exists():
                register_csv_auto(p, "micro", sym, "5m", "bybit")
        # Legacy root BTC + data/ DOGE (registered in place)
        btc = REPO / "btc_5m.csv"
        if btc.exists():
            register_csv_auto(btc, "cex", "BTC", "5m", "binance")
        for p in (REPO / "data").glob("*USDT*5m*.csv"):
            sym = p.name.split("_5m")[0]
            register_csv_auto(p, "cex", sym, "5m", "binance")
        for p in (REPO / "data").glob("*_1d_max.csv"):
            register_csv_auto(p, "dex", p.stem.replace("_1d_max", ""), "1d", "dexscreener")
        print("scan complete")
    print(json.dumps(summary(), indent=2))
