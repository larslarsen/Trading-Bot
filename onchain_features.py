#!/usr/bin/env python3
"""
On-chain feature loader for the ML trainers.

Reads the free, no-key AWS Public Blockchain daily feature CSVs produced by
backfill_onchain.py:

  data/onchain/<CHAIN>_features_daily.csv   (indexed by date)

and aligns them to a training frame's index (5m or 1d). Daily on-chain
features are forward-filled to the higher frequency (a daily metric is
constant within the day). Symbol -> chain mapping lets CEX symbols pull the
matching chain's network stress / whale-flow features as exogenous signals.

Only exposes columns that exist and are non-empty, so partial backfills
(e.g. a chain still filling) degrade gracefully instead of nuking rows.
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).parent
ONCHAIN_DIR = REPO / "data" / "onchain"

# CEX symbol -> on-chain chain whose activity is most relevant.
# BTC/ETH map to their own chain; majors on ETH L2s (base/arbitrum) get those.
SYMBOL_TO_CHAIN = {
    "BTC": "btc", "BTCUSDT": "btc",
    "ETH": "eth", "ETHUSDT": "eth",
    "SOL": "solana", "SOLUSDT": "solana",
    # ETH-L2 natives + tokens that live on Base/Arbitrum
    "ARB": "arbitrum", "ARBUSDT": "arbitrum",
    "MATIC": "polygon", "MATICUSDT": "polygon", "POL": "polygon", "POLUSDT": "polygon",
    "AVAX": "avax", "AVAXUSDT": "avax",
    "XRP": "xrp", "XRPUSDT": "xrp",
    "APT": "aptos", "APTUSDT": "aptos",
    "BNB": "bnb", "BNBUSDT": "bnb",
}


def _load_chain(chain):
    p = ONCHAIN_DIR / f"{chain}_features_daily.csv"
    if not p.exists():
        return pd.DataFrame()
    d = pd.read_csv(p, parse_dates=["date"])
    d = d.set_index("date").sort_index()
    d.index = pd.to_datetime(d.index, utc=True)
    return d


def load_onchain(df_index, symbol=None):
    """Return a DataFrame of on-chain features aligned to `df_index`.

    `df_index` is a DatetimeIndex (5m or 1d). Daily on-chain metrics are
    resampled to the frame frequency via forward-fill. Symbol selects the
    chain via SYMBOL_TO_CHAIN; unknown symbols -> empty (graceful)."""
    feat = pd.DataFrame(index=df_index)
    if symbol is None:
        return feat
    key = symbol.upper().replace("/", "")
    chain = SYMBOL_TO_CHAIN.get(key)
    if not chain:
        return feat
    oc = _load_chain(chain)
    if oc.empty:
        return feat
    # keep only numeric feature cols (drop the redundant 'date' if present)
    num = oc.select_dtypes(include=[np.number])
    if num.empty:
        return feat
    # align to target frequency: daily series -> reindex to df_index (ffill)
    num = num.reindex(df_index, method="ffill")
    prefix = f"oc_{chain}_"
    out = num.add_prefix(prefix)
    return out


def onchain_feature_names(symbol=None):
    """Column names this loader would produce (for feature-list bookkeeping)."""
    if symbol is None:
        return []
    key = symbol.upper().replace("/", "")
    chain = SYMBOL_TO_CHAIN.get(key)
    if not chain:
        return []
    oc = _load_chain(chain)
    if oc.empty:
        return []
    cols = list(oc.select_dtypes(include=[np.number]).columns)
    return [f"oc_{chain}_{c}" for c in cols]
