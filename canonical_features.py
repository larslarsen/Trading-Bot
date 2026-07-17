#!/usr/bin/env python3
"""Single source of truth for the ML feature set.

WHY THIS EXISTS
---------------
The 5m screener's three models (BTC/ETH/DOGE) were trained at different times
with different code, so they carried 85/81/75 features and the serving bot fed
them a DIFFERENT column block -> silently misaligned / flat predictions.

Two real bugs caused the divergence:
  1. ETH cross-asset names are BOTH hardcoded in pipeline.MULTI_ASSET_FEATURES
     AND dynamically emitted by multi_asset_features.add_multi_asset_features
     -> 12 duplicate ETH columns (111 raw names, 98 unique).
  2. oc_*_chain on-chain features are symbol-SPECIFIC (oc_btc_chain_* for BTC,
     oc_eth_chain_* for ETH, none for DOGE) -> the name set differs per pair,
     so no shared list is possible.

This module freezes ONE canonical list (98 names) shared by trainer + serving:
  * symbol-agnostic base features (5m/1h/4h technicals, macro, micro, dex, regime)
  * cross-asset block LOCKED to BTC/ETH/DOGE (24 names), generated deterministically
  * symbol-specific oc_*_chain features EXCLUDED
  * duplicates removed

resolve(df) guarantees every pair presents exactly these 98 columns in the same
order: missing columns are zero-filled, extra columns dropped. So retraining all
three pairs on resolve() output yields byte-compatible 98-dim models.
"""
import re
from pathlib import Path

import pandas as pd

# Cross-asset universe is LOCKED. BTCUSDT is the benchmark base (not a cross
# column); ETHUSDT + DOGEUSDT are the cross assets. Changing this list changes
# the model input dimension -> must retrain all models.
CROSS_ASSETS = ["ETHUSDT", "DOGEUSDT"]

# Symbol-specific on-chain chain features are excluded (they rename per pair).
_ONCHAIN_RE = re.compile(r"^oc_(btc|eth|doge|sol)_chain")

# Features that must always be present even if optional feeds are missing
# (zero-filled by resolve so the dimension never shifts).
# + Order-flow block (Anastasopoulos 2024, EFMA "Order Flow and
#   Cryptocurrency Returns": world order flow dominates fundamentals for
#   predicting crypto returns). load_micro computes these; resolve() was
#   DROPPING them -> the edge was collected but never trained on.
#   Sparse per-pair -> zero-filled like funding_rate.
ALWAYS = ["regime_high_vol", "regime_trending", "funding_rate",
           "taker_buy_sell_ratio", "imbalance", "trade_count", "spread"]


def cross_feature_names():
    """Deterministic 24-name cross-asset block (matches make_cross_features
    output for the locked CROSS_ASSETS)."""
    names = []
    for sym in CROSS_ASSETS:
        names += [
            f"{sym}_returns", f"{sym}_btc_ratio", f"{sym}_btc_ratio_chg",
            f"{sym}_volume_z",
            f"{sym}_btc_ret_rel_6", f"{sym}_btc_ret_rel_12",
            f"{sym}_btc_ret_rel_24", f"{sym}_btc_ret_rel_48",
            f"btc_corr_{sym}_6", f"btc_corr_{sym}_12",
            f"btc_corr_{sym}_24", f"btc_corr_{sym}_48",
        ]
    return names


def _base_list():
    """Symbol-agnostic base feature names, derived from pipeline.ALL_FEATURES
    minus the symbol-specific on-chain chain block, deduped, in stable order."""
    from pipeline import ALL_FEATURES
    seen, out = set(), []
    for f in ALL_FEATURES:
        if _ONCHAIN_RE.match(f):
            continue
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out


# Frozen canonical list: base + cross + always-present.
CANONICAL = _base_list() + cross_feature_names() + ALWAYS
# Final dedupe guard (defensive; cross/base must not overlap).
_seen, _final = set(), []
for f in CANONICAL:
    if f not in _seen:
        _seen.add(f)
        _final.append(f)
CANONICAL = _final

N_FEATURES = len(CANONICAL)


def resolve(df, built_features=None):
    """Return (df_subset, feature_list) presenting EXACTLY CANONICAL.

    df              : feature frame from the trainer/serving build.
    built_features  : optional ordered list the caller assembled; ignored for
                      selection (we use CANONICAL) but kept for call compat.

    Any CANONICAL column missing from df is zero-filled (so a pair lacking an
    optional feed still presents 98 dims). Columns not in CANONICAL are dropped
    EXCEPT the raw OHLCV needed by downstream labeling (triple_barrier_labels)
    and the index, which are preserved so training can label before slicing to
    CANONICAL. Residual NaN rows (leading-window rolling features) are dropped.
    Returns (df_with_ohlc_and_CANONICAL, CANONICAL).
    """
    _RAW_KEEP = ["open", "high", "low", "close", "volume"]
    raw_keep = [c for c in _RAW_KEEP if c in df.columns]
    present = [c for c in CANONICAL if c in df.columns]
    missing = [c for c in CANONICAL if c not in df.columns]
    cols = raw_keep + present
    out = df[cols].copy()
    if missing:
        for c in missing:
            out[c] = 0.0
    # enforce canonical order after the raw-keep prefix
    out = out[raw_keep + CANONICAL]
    # Drop rows only where a column that actually carries data is NaN
    # (leading-window NaNs from rolling features). An entirely-NaN optional
    # column is left as zeros rather than nuking every row.
    has_data = [c for c in CANONICAL if out[c].notna().any()]
    out = out.dropna(subset=has_data, how="any")
    return out, CANONICAL
