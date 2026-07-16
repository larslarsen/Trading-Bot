#!/usr/bin/env python3
"""DEX-wide cross-venue features for CEX ML models.

TWO sources, combined:
1. HISTORICAL (trainable): aggregate the 426 DEX token 1d price bars in
   dex_data/*_1d_max.csv into a daily DEX breadth/risk-appetite series:
     dex_total_mcap_proxy  sum of close (size of the DEX complex)
     dex_n_tokens          tokens with a bar that day
     dex_pct_gainers       fraction of tokens up on the day
     dex_med_ret           cross-sectional median daily return
     dex_mean_ret          cross-sectional mean daily return
     dex_idio_vol          cross-sectional std of daily return (dispersion)
   This has full history (2020->now) so it fills the model's training era.
2. LIVE (forward-only): DexScreener microstructure poller
   (data/dex_micro/<TOKEN>.csv) adds dex_total_volume / dex_total_liquidity /
   dex_med_fdv per poll cycle. These are NaN before "now" (live context only).

Both are resampled to a regular grid and forward-filled onto any CEX bar
index (5m/1h) as cross-venue context.
"""
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).parent
DEX_DATA = REPO / "dex_data"
DEX_MICRO = REPO / "data" / "dex_micro"
GRID = "1h"


def _strip_token(stem):
    for suf in ("_1d_max", "_1d", "_1h_max", "_1h", "_5m_max", "_5m", "_4h_max", "_4h"):
        if stem.endswith(suf):
            return stem[: -len(suf)]
    return stem


def build_dex_breadth(grid=GRID):
    """Return DataFrame indexed by regular grid (DatetimeIndex UTC) with DEX
    features. Combines historical price breadth + live micro breadth."""
    frames = []

    # --- 1. Historical price breadth from dex_data 1d bars ---
    if DEX_DATA.exists():
        files = list(DEX_DATA.glob("*_1d_max.csv")) + list(DEX_DATA.glob("*_1d.csv"))
        series_by_tok = {}
        for p in files:
            try:
                d = pd.read_csv(p, parse_dates=["ts"])
            except Exception:
                continue
            if d.empty or "ts" not in d.columns or "close" not in d.columns:
                continue
            d["ts"] = pd.to_datetime(d["ts"], utc=True).dt.normalize()
            tok = _strip_token(p.stem)
            s = d.set_index("ts")["close"]
            s = s[~s.index.duplicated(keep="last")]
            # keep the longest / most-recent series per token
            if tok not in series_by_tok or len(s) > len(series_by_tok[tok]):
                series_by_tok[tok] = s
        frames = list(series_by_tok.values())

    if frames:
        price = pd.concat(frames, axis=1).sort_index()
        rets = price.pct_change(1)
        breadth_hist = pd.DataFrame(index=price.index)
        breadth_hist["dex_total_mcap_proxy"] = price.sum(axis=1, min_count=1)
        breadth_hist["dex_n_tokens"] = price.notna().sum(axis=1)
        breadth_hist["dex_pct_gainers"] = (rets > 0).mean(axis=1)
        breadth_hist["dex_med_ret"] = rets.median(axis=1)
        breadth_hist["dex_mean_ret"] = rets.mean(axis=1)
        breadth_hist["dex_idio_vol"] = rets.std(axis=1)
        breadth_hist = breadth_hist.ffill()
    else:
        breadth_hist = pd.DataFrame()

    # --- 2. Live micro breadth from DexScreener poller ---
    if DEX_MICRO.exists():
        mframes = []
        for p in DEX_MICRO.glob("*.csv"):
            try:
                d = pd.read_csv(p, parse_dates=["ts"])
            except Exception:
                continue
            if d.empty or "ts" not in d.columns:
                continue
            d["ts"] = pd.to_datetime(d["ts"], utc=True)
            d["token"] = _strip_token(p.stem)
            mframes.append(d)
        if mframes:
            wide = pd.concat(mframes, ignore_index=True).sort_values("ts")
            # bin polls into ~10-min cycles so a full sweep collapses to one row
            wide["cycle"] = wide["ts"].dt.floor("10min")
            def agg(g):
                chg = g["price_chg_h24_med"].dropna()
                return pd.Series({
                    "dex_total_volume": g["volume_h24"].sum(min_count=1),
                    "dex_total_liquidity": g["liquidity_usd"].sum(min_count=1),
                    "dex_med_fdv": g["fdv"].median(),
                })
            micro = wide.groupby("cycle").apply(agg, include_groups=False)
            micro.index = micro.index.tz_convert("UTC") if micro.index.tz else micro.index.tz_localize("UTC")
            micro = micro.ffill()
            # merge onto historical index if it exists, else use micro index
            if breadth_hist.empty:
                breadth_hist = micro
            else:
                breadth_hist = breadth_hist.join(micro, how="outer").sort_index().ffill()

    if breadth_hist.empty:
        return pd.DataFrame()

    # Resample to regular grid, forward-fill
    gi = pd.date_range(breadth_hist.index.min(), breadth_hist.index.max(), freq=grid, tz="UTC")
    out = breadth_hist.reindex(gi).ffill().bfill()
    return out


def add_dex_features(df, grid=GRID):
    """Join DEX breadth onto a CEX frame (df indexed by ts/UTC)."""
    breadth = build_dex_breadth(grid)
    if breadth.empty:
        return df, []
    idx = df.index
    if getattr(idx, "tz", None) is None:
        idx = idx.tz_localize("UTC")
    b = breadth.reindex(idx, method="ffill")
    added = list(breadth.columns)
    for c in added:
        df[c] = b[c].values
    return df, added


if __name__ == "__main__":
    b = build_dex_breadth()
    print(f"DEX breadth rows: {len(b)}  cols: {list(b.columns)}")
    if not b.empty:
        print(b.tail(3).to_string())
