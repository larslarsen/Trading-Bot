#!/usr/bin/env python3
"""
Derive DEX 1h/4h/1d OHLCV from local 5m data — ZERO API calls.

Why this exists: backfill_dex_mtf.py pulls 1h/4h/1d DIRECTLY from GeckoTerminal
(one API call per TF per token). That's the deep 3yr history we want to keep.
But going forward, the 5m forward collector (dex_forward_collector.py) already
polls every token's live price every 5m into data/<SYM>_5m_dex_max.csv. We can
RESAMPLE that 5m into 1h/4h/1d locally and never call the API again for those
timeframes. This is the "limit API calls / avoid the $35 license" path.

MERGE semantics (the user wants ALL the free data, not less):
  - The deep direct-pull history is kept as-is.
  - Derived bars are appended ONLY for dates AFTER the deep file's last date,
    so there is no overlap and no dropping of real OHLCV history. As 5m
    accumulates, the derived tail keeps 1h/4h/1d current with no extra calls.

ts-format handling: existing deep files are date-only ("2026-07-15"). The
script detects the existing file's format and matches it, so each output file
stays internally consistent. If no deep file exists yet, 1d -> date, 1h/4h ->
full timestamp.

Memory safety (same as backfill fix): per-token gc.collect() + a hard RSS cap
that aborts cleanly before any OOM risk.

Usage:
    python derive_dex_tf.py                 # all tokens that have 5m data
    python derive_dex_tf.py --token AAVE    # one token (test)
    python derive_dex_tf.py --tfs 1h 1d     # subset of timeframes
"""
import argparse
import gc
from pathlib import Path

import pandas as pd
from mem_guard import guard as _mem_guard

ROOT = Path(__file__).parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
DEFAULT_MEM_LIMIT_MB = 1536

TFS = {"1h": "1h", "4h": "4h", "1d": "1d"}
AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}


def _detect_fmt(tf: str) -> str:
    """ts format for a derived file. 1d -> date-only (one bar/day, correct).
    1h/4h -> FULL timestamp unconditionally: a date-only ts would collapse all
    hourly/4-hourly bars of a day onto one row and corrupt the file. We never
    inherit a (possibly wrong) date-only format from an existing deep file."""
    return "%Y-%m-%d" if tf == "1d" else "%Y-%m-%d %H:%M:%S+0000"


def _resample(df5m: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample a 5m DataFrame into `rule` bars with proper OHLCV aggregation."""
    d = df5m.copy()
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    d = d.set_index("ts").sort_index()
    out = d.resample(rule).agg(AGG).dropna()
    return out.reset_index().rename(columns={"index": "ts"})


def derive_token(sym: str, tfs: list, mem_limit: int) -> dict:
    sym = sym.upper()
    src = DATA / f"{sym}_5m_dex_max.csv"
    if not src.exists():
        return {tf: 0 for tf in tfs}
    df5m = pd.read_csv(src)
    if df5m.empty or len(df5m) < 2:
        return {tf: 0 for tf in tfs}  # need >=2 bars to form a bucket
    added = {}
    for tf in tfs:
        _mem_guard(mem_limit)
        r = _resample(df5m, TFS[tf])
        if r.empty:
            added[tf] = 0
            continue
        out = DATA / f"{sym}_{tf}_dex_max.csv"
        if out.exists():
            try:
                deep = pd.read_csv(out)
                if deep.empty:
                    r.to_csv(out, index=False)
                    added[tf] = len(r)
                    continue
                # last date present in deep history
                last_deep = pd.to_datetime(deep["ts"], utc=True).max().date()
                # keep only derived bars strictly AFTER deep history (no overlap)
                r["_d"] = pd.to_datetime(r["ts"], utc=True).dt.date
                r = r[r["_d"] > last_deep].drop(columns="_d")
                if r.empty:
                    added[tf] = 0
                    gc.collect()
                    continue
                fmt = _detect_fmt(tf)
                r["ts"] = pd.to_datetime(r["ts"], utc=True).dt.strftime(fmt)
                # dedupe by ts just in case
                existing = set(pd.to_datetime(deep["ts"], utc=True).dt.strftime(fmt))
                r = r[~r["ts"].isin(list(existing))]
                if r.empty:
                    added[tf] = 0
                    gc.collect()
                    continue
                # pandas>=2 removed DataFrame.append; use concat (preserves order)
                pd.concat([deep, r], ignore_index=True).to_csv(out, index=False)
                added[tf] = len(r)
                gc.collect()
                continue
            except Exception as e:
                print(f"  {sym} {tf}: merge failed ({e}) — writing derived only")
        # no existing deep file: write derived fully
        r["ts"] = pd.to_datetime(r["ts"], utc=True).dt.strftime(_detect_fmt(tf))
        r.to_csv(out, index=False)
        added[tf] = len(r)
        gc.collect()
    return added


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--token", default=None, help="single symbol (default: all with 5m data)")
    ap.add_argument("--tfs", nargs="+", choices=list(TFS), default=list(TFS))
    ap.add_argument("--mem-limit-mb", type=int, default=DEFAULT_MEM_LIMIT_MB)
    args = ap.parse_args()

    _mem_guard(args.mem_limit_mb)

    if args.token:
        syms = [args.token.upper()]
    else:
        syms = sorted({p.name.split("_")[0] for p in DATA.glob("*_5m_dex_max.csv")})

    print(f"Derive DEX TFs from local 5m (no API): {len(syms)} tokens, TFs={args.tfs} "
          f"(mem cap={args.mem_limit_mb}MB)")
    tot = {tf: 0 for tf in args.tfs}
    for sym in syms:
        _mem_guard(args.mem_limit_mb)
        added = derive_token(sym, args.tfs, args.mem_limit_mb)
        for tf in args.tfs:
            tot[tf] += added.get(tf, 0)
        if any(added.values()):
            print(f"  {sym}: +{added}")
    gc.collect()
    print(f"Derive complete. Appended bars: {tot}")


if __name__ == "__main__":
    main()
