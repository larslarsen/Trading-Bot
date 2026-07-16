#!/usr/bin/env python3
"""
Microstructure feature engineering.
Loads Bybit funding, OI, insurance, taker trade history, and orderbook snapshots;
aligns them to the 5m index with lag-safe forward-fill.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent

FUND_FILE = OUT_DIR / "funding_history.csv"
FUND_DIR = OUT_DIR / "data" / "funding"  # new deep-history Bybit funding (backfill_funding_mexc.py)
OI_FILE = OUT_DIR / "oi_history.csv"
LIQ_FILE = OUT_DIR / "liquidations_history.csv"
TRADE_AGG_FILE = OUT_DIR / "trade_agg_5m.csv"
ORDERBOOK_FILE = OUT_DIR / "orderbook_5m.csv"


def load_funding(symbol, dbars_index):
    """Load DEEP-HISTORY Bybit funding rate for `symbol` from
    data/funding/<SYM>USDT_funding.csv (8h interval, years of history) and
    align (forward-fill) to the 5m/1d training index. This is the high-value
    funding signal the backfills produced -- preferred over the legacy
    funding_history.csv live feed. Returns a single-column DataFrame
    ['funding_rate'] indexed like dbars_index, or empty if unavailable."""
    sym = symbol.upper().replace("/", "").replace("USDT", "")
    # funding files are named <SYM>USDT_funding.csv
    cand = FUND_DIR / f"{sym}USDT_funding.csv"
    if not cand.exists():
        return pd.DataFrame(index=dbars_index)
    d = pd.read_csv(cand, parse_dates=["ts"])
    if "ts" not in d.columns or "funding_rate" not in d.columns:
        return pd.DataFrame(index=dbars_index)
    d = d.set_index("ts").sort_index()
    d.index = pd.to_datetime(d.index, utc=True)
    d = d[~d.index.duplicated(keep="last")]
    d = d[["funding_rate"]].reindex(dbars_index, method="ffill")
    return d.ffill()


def load_micro(dbars_index):
    features = pd.DataFrame(index=dbars_index)


def _read_ts_first(path):
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["ts"])
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], errors="coerce", utc=True)
        df = df.dropna(subset=["ts"])
    return df


def load_micro(dbars_index):
    features = pd.DataFrame(index=dbars_index)

    fr = _read_ts_first(FUND_FILE)
    oi = _read_ts_first(OI_FILE)
    liq = _read_ts_first(LIQ_FILE)
    trades = _read_ts_first(TRADE_AGG_FILE)
    ob = _read_ts_first(ORDERBOOK_FILE)

    fr_cols = ["funding_rate", "index_price", "mark_price"]
    oi_cols = ["open_interest"]
    liq_cols = ["insurance_balance"]
    trade_cols = ["taker_buy_vol", "taker_sell_vol", "trade_count"]
    ob_cols = ["spread", "mid_price", "imbalance"]

    def _merge(df_src, cols):
        return set(df_src.columns).intersection(cols)

    if not fr.empty and _merge(fr, fr_cols):
        fr = fr.set_index("ts").sort_index()
        fr = fr[~fr.index.duplicated(keep="last")].reindex(dbars_index, method="ffill")
        for c in _merge(fr, fr_cols):
            features[c] = pd.to_numeric(fr[c], errors="coerce")
        if {"mark_price", "index_price"} <= set(features.columns):
            basis = (features["mark_price"] - features["index_price"]) / (features["index_price"] + 1e-10)
            features["basis_pct"] = basis
            features["basis_pct_chg"] = basis.diff(12)

    if not oi.empty and _merge(oi, oi_cols):
        oi = oi.set_index("ts").sort_index()
        oi = oi[~oi.index.duplicated(keep="last")].reindex(dbars_index, method="ffill")
        for c in _merge(oi, oi_cols):
            features[c] = pd.to_numeric(oi[c], errors="coerce")
        if "open_interest" in features.columns:
            features["oi_chg_12"] = features["open_interest"].pct_change(12)
            features["oi_chg_24"] = features["open_interest"].pct_change(24)

    if not liq.empty and _merge(liq, liq_cols):
        liq = liq.set_index("ts").sort_index()
        liq = liq[~liq.index.duplicated(keep="last")].reindex(dbars_index, method="ffill")
        for c in _merge(liq, liq_cols):
            features[c] = pd.to_numeric(liq[c], errors="coerce")
        if "insurance_balance" in features.columns:
            features["insurance_chg_12"] = features["insurance_balance"].pct_change(12)

    if not trades.empty and _merge(trades, trade_cols):
        trades = trades.set_index("ts").sort_index()
        trades = trades[~trades.index.duplicated(keep="last")].reindex(dbars_index, method="ffill")
        for c in _merge(trades, trade_cols):
            features[c] = pd.to_numeric(trades[c], errors="coerce")
        if {"taker_buy_vol", "taker_sell_vol"} <= set(features.columns):
            denom = features["taker_buy_vol"] + features["taker_sell_vol"] + 1e-10
            features["taker_buy_sell_ratio"] = features["taker_buy_vol"] / denom

    if not ob.empty and _merge(ob, ob_cols):
        ob = ob.set_index("ts").sort_index()
        ob = ob[~ob.index.duplicated(keep="last")].reindex(dbars_index, method="ffill")
        for c in _merge(ob, ob_cols):
            features[c] = pd.to_numeric(ob[c], errors="coerce")

    drop_candidates = ["symbols"]
    features = features.drop(columns=[c for c in drop_candidates if c in features.columns])
    return features.ffill().bfill()
