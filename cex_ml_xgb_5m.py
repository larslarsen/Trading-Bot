#!/usr/bin/env python3
"""
Multi-pair 5m ML paper trader.

One daemon that trades ALL our single-pair XGBoost models, ranked by signal
strength -- NOT all pairs all the time. For each configured pair it loads the
pair's 5m history, builds features with the EXACT pipeline model_trainer uses
(resampled + macro + multi-asset + micro + DEX breadth + derive + regime),
loads that pair's model, predicts a signal + confidence, then trades the
top-N pairs by signal strength via a single multi-position book.

This is the "rank ML trades like a screener" bot: instead of holding every
pair, it holds the N strongest directional signals (N = CONFIG.max_positions).

Inference is inline (models loaded once at startup) so there is one process to
supervise, not one server + one trader per pair.

Usage:
  python cex_ml_xgb_5m.py            # loop forever, poll every 300s
  python cex_ml_xgb_5m.py --once    # one ranking + execution pass
"""
import argparse
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline import (
    fetch_data, add_resampled_features, load_macro_data, add_macro_signals,
    derive_features, detect_regime, ALL_FEATURES, MULTI_ASSET_FILE, USE_MULTI_ASSET,
)
from multi_asset_features import add_multi_asset_features
from order_manager_multi import MultiPositionState

REPO = Path(__file__).parent
MODELS_DIR = REPO / "models"

# pair -> model file. BTC + DOGE to start; add more as single-pair models land.
# The fetch symbol differs from the pair key for BTC (root btc_5m.csv vs the
# data/<SYM>USDT_5m_max.csv convention) -- see FETCH_SYMBOL.
PAIRS = {
    "BTCUSDT": "latest_xgb.json",
    "DOGEUSDT": "doge_xgb.json",
}
# pair key -> symbol string accepted by pipeline.fetch_data (BTC -> "BTC").
FETCH_SYMBOL = {"BTCUSDT": "BTC", "DOGEUSDT": "DOGEUSDT"}

POLL_SEC = 300
STATE_FILE = REPO / "execution_state_ml_multi.json"
JOURNAL_FILE = REPO / "trade_journal_ml_multi.json"

# How many pairs to hold at once (top-N by signal strength).
MAX_POSITIONS = 5
# Minimum confidence to take a directional (LONG/SHORT) signal.
CONFIDENCE_THRESHOLD = 0.60
# Fee profile (mirrors paper_trader.py)
FEE_BP = 0.60
SLIPPAGE_BP = 5
SIZE_USD = 2000.0


def load_models():
    """Load every pair's model once. Returns {pair: XGBClassifier}."""
    models = {}
    for pair, fname in PAIRS.items():
        p = MODELS_DIR / fname
        if not p.exists():
            print(f"[ml_multi] no model for {pair} ({fname}), skipping")
            continue
        m = xgb.XGBClassifier()
        m.load_model(str(p))
        models[pair] = m
        print(f"[ml_multi] loaded {pair} model: n_features={getattr(m, 'n_features_in_', None)}")
    return models


def build_features(symbol):
    """Reproduce model_trainer.train_and_save's feature build for `symbol`."""
    fsym = FETCH_SYMBOL.get(symbol, symbol)
    df = fetch_data(fsym)
    tf = add_resampled_features(df)
    df = df.join(tf, how="left")
    macro = load_macro_data(df.index)
    df = add_macro_signals(df, macro)
    multi_cols = []
    if USE_MULTI_ASSET and Path(MULTI_ASSET_FILE).exists():
        df, multi_cols = add_multi_asset_features(df, MULTI_ASSET_FILE)
    # micro (Bybit CEX) -- optional, skip if unavailable
    try:
        from micro_features import load_micro
        micro = load_micro(df.index)
        if not micro.empty and micro.notna().any().any():
            df = df.join(micro, how="left")
    except Exception:
        pass
    # NEW: deep-history Bybit funding rate (matches trainer wiring)
    try:
        from micro_features import load_funding
        if "funding_rate" in df.columns:
            df = df.drop(columns=["funding_rate"])
        fund = load_funding(symbol, df.index)
        if fund is not None and not fund.empty:
            df = df.join(fund, how="left")
    except Exception:
        pass
    # NEW: on-chain network metrics (matches trainer wiring)
    try:
        from onchain_features import load_onchain
        oc = load_onchain(df.index, symbol)
        if oc is not None and not oc.empty:
            df = df.join(oc, how="left")
    except Exception:
        pass
    # DEX-wide breadth -- optional; capture its columns (trainer includes them)
    dex_cols = []
    try:
        from dex_features import add_dex_features
        df, dex_cols = add_dex_features(df)
    except Exception:
        pass
    df = derive_features(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.sort_index()
    df = detect_regime(df)
    features = [f for f in ALL_FEATURES if f in df.columns] + multi_cols + dex_cols + ["regime_high_vol", "regime_trending"]
    # NEW high-value exogenous features (funding + on-chain) if present
    extra = ["funding_rate"] + [c for c in df.columns if c.startswith("oc_")]
    for cand in extra:
        if cand in df.columns:
            features.append(cand)
    df = df.dropna(subset=[c for c in features if df[c].notna().any()])
    if df.empty:
        return None, None
    fvec = df[features].tail(1).copy()
    last_bar = df.tail(1).iloc[0]
    return fvec, last_bar


def predict_pair(model, fvec):
    """Predict a signal for one pair. Returns (signal, confidence, strength)."""
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and len(fvec.columns) != int(expected):
        return "FLAT", 0.0, 0.0
    exp_names = getattr(model, "feature_names_in_", None)
    if exp_names is not None:
        missing = [c for c in exp_names if c not in fvec.columns]
        if missing:
            return "FLAT", 0.0, 0.0
        fvec = fvec[list(exp_names)]
    X = np.nan_to_num(fvec.values, nan=0.0, posinf=0.0, neginf=0.0)
    probs = model.predict_proba(X)[0]
    cls = int(model.predict(X)[0])
    signal_map = {0: "SHORT", 1: "LONG", 2: "FLAT"}
    signal = signal_map.get(cls, "FLAT")
    conf = float(max(probs))
    # strength: directional edge scaled by confidence (FLAT = 0)
    if signal == "LONG":
        strength = conf
    elif signal == "SHORT":
        strength = -conf
    else:
        strength = 0.0
    return signal, conf, strength


def price_for(symbol):
    """Last close for a pair from its 5m history."""
    df = fetch_data(FETCH_SYMBOL.get(symbol, symbol))
    return float(df["close"].iloc[-1]) if not df.empty else None


def run_once(models, state):
    # 1) gather ranked signals
    ranked = []
    for pair, model in models.items():
        try:
            fvec, _ = build_features(pair)
            if fvec is None:
                continue
            signal, conf, strength = predict_pair(model, fvec)
            ranked.append({"pair": pair, "signal": signal, "conf": conf, "strength": strength})
        except Exception as e:
            print(f"[{datetime.now(timezone.utc).isoformat()}] {pair} predict error: {e}")
    ranked.sort(key=lambda r: -abs(r["strength"]))
    print(f"[{datetime.now(timezone.utc).isoformat()}] ranked: " +
          ", ".join(f"{r['pair']}={r['signal']}({r['conf']:.2f})" for r in ranked))

    # 2) decide desired book = top-N directional signals above threshold
    desired = [r for r in ranked if r["signal"] != "FLAT" and r["conf"] >= CONFIDENCE_THRESHOLD][:MAX_POSITIONS]
    desired_pairs = {r["pair"]: r for r in desired}

    # 3) close positions no longer desired / no longer signaled
    for sym in list(state.positions.keys()):
        if sym not in desired_pairs:
            px = price_for(sym)
            if px:
                state.close_position(sym, px)
                print(f"CLOSE {sym} @ {px:.6f} (not in top-{MAX_POSITIONS})")

    # 4) open top-N desired (respect max positions + circuit breakers)
    for r in desired:
        sym = r["pair"]
        if sym in state.positions:
            continue
        if len(state.positions) >= MAX_POSITIONS:
            break
        px = price_for(sym)
        if not px or px <= 0:
            continue
        ok, reason = state.check_circuit_breakers()
        if not ok:
            print(f"CIRCUIT BREAKER: {reason}, staying flat")
            break
        size = SIZE_USD
        pos = state.open_position(sym, px, size)
        if pos:
            print(f"OPEN {sym} @ {px:.6f}, size=${size:.2f} ({r['signal']} conf={r['conf']:.2f})")

    # 5) mark to market + save
    try:
        mtm = {r["pair"]: price_for(r["pair"]) for r in ranked}
        eq = state.mark_to_market({k: v for k, v in mtm.items() if v})
        dd = (state.peak_equity - eq) / state.peak_equity if state.peak_equity > 0 else 0
        print(f"MTM equity: ${eq:.2f}  Peak: ${state.peak_equity:.2f}  DD: {dd:.2%}  Positions: {len(state.positions)}")
        if dd > 0.20:
            print(f"Flattening: DD {dd:.2%} > 20%")
            state.flatten_all({k: v for k, v in mtm.items() if v})
            state.halt(f"live_drawdown_flatten_{pd.Timestamp.now().isoformat()}")
    except Exception as e:
        print(f"MTM error: {e}")
    state.save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    models = load_models()
    if not models:
        print("[ml_multi] no models loaded, exiting")
        return
    state = MultiPositionState(initial_capital=10000.0, state_file=STATE_FILE, journal_file=JOURNAL_FILE)
    print(f"[ml_multi] starting. pairs={list(models.keys())} max_positions={MAX_POSITIONS}")
    if args.once:
        run_once(models, state)
        return
    while True:
        try:
            run_once(models, state)
        except Exception as e:
            print(f"[{datetime.now(timezone.utc).isoformat()}] error: {e}")
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
