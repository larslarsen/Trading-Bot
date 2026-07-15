#!/usr/bin/env python3
import os
import argparse
"""
Weekly retrain + model serializer.
Trains XGBoost on the latest expanding window and saves the model
for the inference server to serve.
"""
import json
import time
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from datetime import datetime, timezone
from pipeline import (
    fetch_data, derive_features, triple_barrier_labels,
    walk_forward_splits, LAMBDA, COST,
    add_resampled_features, add_macro_signals,
    load_macro_data, ALL_FEATURES, detect_regime,
    USE_MULTI_ASSET, MULTI_ASSET_FILE,
)
from multi_asset_features import add_multi_asset_features
from data_feed import HISTORY_CSV
import warnings
warnings.filterwarnings("ignore")

# Parallelism: prefer the project's hardware setting (config.N_JOBS), which
# honors the TRADING_BOT_CORES env var. Fall back to leaving one core free if
# config isn't importable (e.g. standalone run outside the package).
try:
    from config import N_JOBS as _N_JOBS
except Exception:
    _N_JOBS = max(1, (os.cpu_count() or 8) // 2 - 1)

MODEL_DIR = Path(__file__).parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
METRICS_FILE = MODEL_DIR / "latest_metrics.json"

N_TREES = 200
MAX_DEPTH = 4


def model_out_path(sym_tag: str) -> Path:
    """Where to serialize a trained model.

    BTC keeps models/latest_xgb.json (the path the serving bot consumes).
    Any other symbol writes models/<sym>_xgb.json so it never clobbers BTC's
    production model.
    """
    if sym_tag == "BTC":
        return MODEL_DIR / "latest_xgb.json"
    return MODEL_DIR / f"{sym_tag.lower()}_xgb.json"


def train_and_save(symbol=None) -> bool | None:
    """Train on latest expanding window, save model JSON.
    `symbol` (default BTC) selects the 5m history file; non-BTC models are
    saved to models/<sym>_xgb.json so they never clobber BTC's latest_xgb.json
    (which the serving bot consumes)."""
    t0 = time.time()
    sym_tag = (symbol or "BTC").upper().replace("/", "")
    print(f"[{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}] Train starting for {sym_tag}...")

    # Load data
    df = fetch_data(symbol)
    tf = add_resampled_features(df)
    df = df.join(tf, how="left")
    macro = load_macro_data(df.index)
    df = add_macro_signals(df, macro)
    if USE_MULTI_ASSET and Path(MULTI_ASSET_FILE).exists():
        print(f'  Adding multi-asset features from {MULTI_ASSET_FILE}...')
        df = add_multi_asset_features(df, MULTI_ASSET_FILE)
    from micro_features import load_micro
    micro = load_micro(df.index)
    if not micro.empty and micro.notna().any().any():
        df = df.join(micro, how='left')
        print(f'  Loaded {len(micro.columns)} micro columns')
    else:
        print('  No usable micro data, continuing')
    df = derive_features(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df = triple_barrier_labels(df)

    # Use feature subset drop to avoid blanket dropna wiping valid rows when
    # optional micro/macro data is partially missing.
    feature_cols = [f for f in ALL_FEATURES if f in df.columns] + ["label"]
    available = [c for c in feature_cols if c in df.columns]
    # Exclude columns that are entirely NaN so partial micro data doesn't
    # nuke every row.
    available = [c for c in available if df[c].notna().any()]
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        for col in missing:
            df[col] = np.nan
    df.dropna(subset=available, inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.sort_index()
    df = detect_regime(df)

    features = [f for f in ALL_FEATURES if f in df.columns] + ["regime_high_vol", "regime_trending"]
    X = df[features].values
    y = df["label"].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    splits = walk_forward_splits(df)
    if not splits:
        print("ERROR: no valid walks produced")
        return

    # Use last fold as the "production" model
    sp = splits[-1]
    train_idx = np.array(sp["train_idx"], dtype=int)
    # Rolling 4-year window aligned to BTC halving cycle
    if len(train_idx):
        train_dates = df.index[train_idx]
        cutoff = train_dates[-1] - pd.Timedelta(days=int(365.25 * 4))
        train_idx = train_idx[train_dates >= cutoff]
    val_idx = np.array(sp["val_idx"], dtype=int)
    test_idx = np.array(sp["test_idx"], dtype=int)
    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    print(f"  Train={len(X_tr)}, Val={len(X_val)}, Test={len(X_te)}, Features={len(features)}")

    model = xgb.XGBClassifier(
        objective="multi:softmax", num_class=3,
        max_depth=MAX_DEPTH, learning_rate=0.05, n_estimators=N_TREES,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, n_jobs=_N_JOBS,
        random_state=42, early_stopping_rounds=30,
        eval_metric="mlogloss", class_weight="balanced",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    best = getattr(model, "best_ntree_limit", None) or model.n_estimators
    print(f"  Trained with {best} trees")

    # Test accuracy
    preds = model.predict(X_te)
    acc = float((preds == y_te).mean())
    print(f"  Test accuracy: {acc:.3f}")

    meta = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'features': features,
        'feature_count': len(features),
        'train_bars': int(len(X_tr)),
        'val_bars': int(len(X_val)),
        'test_bars': int(len(X_te)),
        'trees': int(best),
        'accuracy': acc,
        'use_multi_asset': bool(USE_MULTI_ASSET),
        'elapsed_sec': round(time.time() - t0, 1),
    }
    meta_path = MODEL_DIR / 'latest_meta.json'
    meta_path.write_text(json.dumps(meta, indent=2))

    # Save model — BTC default keeps latest_xgb.json (serving bot path);
    # any other symbol gets its own models/<sym>_xgb.json to avoid clobber.
    out_path = model_out_path(sym_tag)
    model.save_model(str(out_path))
    print(f"  Model saved: {out_path} ({out_path.stat().st_size/1024:.1f} KB)")
    print(f"  Meta saved: {meta_path}")

    # Save metrics
    metrics = {
        "timestamp": meta['timestamp'],
        "accuracy": acc,
        "features": len(features),
        "train_bars": int(len(X_tr)),
        "test_bars": int(len(X_te)),
        "trees": int(best),
        "elapsed_sec": meta['elapsed_sec'],
    }
    METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    print(f"  Metrics: {METRICS_FILE}")
    print(f"  Done in {metrics['elapsed_sec']:.1f}s")
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTC", help="pair to train (default BTC). e.g. DOGE")
    args = ap.parse_args()
    train_and_save(symbol=args.symbol)