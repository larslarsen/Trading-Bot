#!/usr/bin/env python3
"""
CEX 1d ML trainer -- ONE pooled cross-symbol XGBoost direction predictor.

CEX 1d history per symbol is thin (~300 daily bars), like DEX. So we POOL every
symbol's daily bars into one panel, build features PER-SYMBOL (rolling windows
never bleed across symbols), triple-barrier label, and train ONE model that
predicts direction for any CEX symbol from its own daily features. Consumed by
cex_ml_xgb_1d.py.

This is the 1d analogue of model_trainer.py (which is deep per-pair 5m) and the
CEX twin of train_dex_ml.py.

Output: models/cex_1d_xgb.json (+ cex_1d_ml_meta.json, cex_1d_ml_metrics.json)

Usage: python train_cex_1d_ml.py
"""
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline import derive_features, triple_barrier_labels, ALL_FEATURES

REPO = Path(__file__).parent
DATA_DIR = REPO / "data"
MODELS_DIR = REPO / "models"
MIN_BARS = 60
N_TREES = 200
MAX_DEPTH = 4
N_JOBS = 5


def load_symbol(path):
    """Load one CEX symbol's daily OHLCV, build features + labels in isolation."""
    df = pd.read_csv(path).dropna(subset=["close", "high", "low", "volume"])
    if len(df) < MIN_BARS or "ts" not in df.columns:
        return None
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts").set_index("ts")
    df = df[~df.index.duplicated(keep="first")]
    if len(df) < MIN_BARS:
        return None
    df = derive_features(df)
    # NEW high-value exogenous: deep-history funding + on-chain network features
    sym = path.stem.replace("_1d_max", "").replace("_1d", "")
    from micro_features import load_funding
    fund = load_funding(sym, df.index)
    if not fund.empty:
        df = df.join(fund, how="left")
    from onchain_features import load_onchain
    oc = load_onchain(df.index, sym)
    if not oc.empty:
        df = df.join(oc, how="left")
    df = triple_barrier_labels(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df["symbol"] = path.stem.replace("_1d_max", "")
    return df


def build_panel():
    frames = []
    for p in sorted(DATA_DIR.glob("*_1d_max.csv")):
        f = load_symbol(p)
        if f is not None and not f.empty:
            frames.append(f)
    if not frames:
        raise SystemExit("no usable CEX 1d symbols")
    return pd.concat(frames).sort_index()


def main():
    t0 = time.time()
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}] CEX 1d ML train starting...")
    panel = build_panel()
    n_syms = panel["symbol"].nunique()
    features = [f for f in ALL_FEATURES if f in panel.columns]
    features = [c for c in features if panel[c].notna().any()]
    # NEW high-value exogenous (funding + on-chain) if present in the panel
    extra = [c for c in panel.columns if c == "funding_rate" or c.startswith("oc_")]
    extra = [c for c in extra if panel[c].notna().any()]
    features = features + extra
    # Only dropna on core price features + label, NOT the optional exogenous
    # (funding/on-chain are sparse early; forward-filled, missing -> 0 not drop)
    core = [f for f in ALL_FEATURES if f in panel.columns and panel[f].notna().any()]
    panel = panel.dropna(subset=core + ["label"])
    print(f"  Pooled {len(panel)} rows / {n_syms} symbols / {len(features)} features")
    if len(panel) < 2000:
        raise SystemExit(f"too few pooled rows ({len(panel)}) to train")

    X = np.nan_to_num(panel[features].values, nan=0.0, posinf=0.0, neginf=0.0)
    y = panel["label"].values.astype(int)

    n = len(panel)
    i_tr, i_val = int(n * 0.80), int(n * 0.85)
    Xtr, ytr = X[:i_tr], y[:i_tr]
    Xval, yval = X[i_tr:i_val], y[i_tr:i_val]
    Xte, yte = X[i_val:], y[i_val:]

    model = xgb.XGBClassifier(
        n_estimators=N_TREES, max_depth=MAX_DEPTH, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softprob",
        num_class=3, n_jobs=N_JOBS, eval_metric="mlogloss",
        class_weight="balanced",
    )
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)

    acc = float((model.predict(Xte) == yte).mean()) if len(yte) else float("nan")
    MODELS_DIR.mkdir(exist_ok=True)
    out = MODELS_DIR / "cex_1d_xgb.json"
    model.save_model(str(out))
    (MODELS_DIR / "cex_1d_ml_meta.json").write_text(json.dumps(
        {"features": features, "n_symbols": int(n_syms), "rows": int(n),
         "trained_at": datetime.now(timezone.utc).isoformat()}, indent=2))
    (MODELS_DIR / "cex_1d_ml_metrics.json").write_text(json.dumps(
        {"test_acc": acc, "train": int(i_tr), "val": int(i_val - i_tr),
         "test": int(n - i_val), "n_features": len(features)}, indent=2))
    print(f"  Test accuracy: {acc:.3f}")
    print(f"  Model saved: {out} ({out.stat().st_size/1024:.1f} KB)")
    print(f"  Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
