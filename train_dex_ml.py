#!/usr/bin/env python3
"""
DEX ML trainer -- ONE pooled cross-token XGBoost direction predictor.

Unlike the CEX side (deep per-pair 5m history -> one model per pair), DEX
tokens are thin (median <200 daily bars, dex_data/<TOK>_1d_max.csv). Per-token
models are infeasible. So we POOL every token into a single panel (~50k rows,
540 tokens), build features PER-TOKEN (rolling windows never bleed across
tokens), triple-barrier label, and train ONE model that predicts direction for
any DEX token from its own features. Consumed by dex_ml_xgb_1d.py.

Output: models/dex_xgb.json (+ dex_ml_meta.json, dex_ml_metrics.json)

Usage: python train_dex_ml.py
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
DEX_DIR = REPO / "dex_data"
MODELS_DIR = REPO / "models"
MIN_BARS = 60          # per-token minimum after feature warmup
N_TREES = 200
MAX_DEPTH = 4
N_JOBS = 5


def load_token(path):
    """Load one DEX token's OHLCV, build features + labels in isolation."""
    df = pd.read_csv(path, parse_dates=["ts"]).dropna(subset=["close", "high", "low", "volume"])
    if len(df) < MIN_BARS:
        return None
    df = df.sort_values("ts").set_index("ts")
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="first")]
    df = derive_features(df)
    # NEW high-value exogenous (graceful: DEX tokens mostly have no matching
    # chain/funding -> empty -> skipped). BTC/ETH-denominated tokens pick up
    # the matching on-chain network features as cross-asset macro signal.
    sym = path.stem.replace("_1d_max", "")
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
    df["token"] = path.stem.replace("_1d_max", "")
    return df


def build_panel():
    """Pool every token into one labelled panel."""
    frames = []
    for p in sorted(DEX_DIR.glob("*_1d_max.csv")):
        f = load_token(p)
        if f is not None and not f.empty:
            frames.append(f)
    if not frames:
        raise SystemExit("no usable DEX tokens")
    panel = pd.concat(frames)
    panel = panel.sort_index()  # chronological across all tokens for causal split
    return panel


def main():
    t0 = time.time()
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}] DEX ML train starting...")
    panel = build_panel()
    n_tokens = panel["token"].nunique()
    features = [f for f in ALL_FEATURES if f in panel.columns]
    features = [c for c in features if panel[c].notna().any()]
    # NEW high-value exogenous (funding + on-chain) if present
    extra = [c for c in panel.columns if c == "funding_rate" or c.startswith("oc_")]
    extra = [c for c in extra if panel[c].notna().any()]
    features = features + extra
    # DEX tokens are THIN: dropping on all 38 core price features nukes ~99% of
    # rows (most tokens lack full feature history). Require only label + a
    # minimal essential set so the pooled panel stays trainable. Optional
    # exogenous (funding/oc_) stays as 0-fill, not a drop criterion.
    essential = [c for c in ("close", "log_ret", "ret_1", "vol_z", "rsi_14")
                 if c in panel.columns and panel[c].notna().any()]
    panel = panel.dropna(subset=essential + ["label"])
    print(f"  Pooled {len(panel)} rows / {n_tokens} tokens / {len(features)} features")
    if len(panel) < 2000:
        raise SystemExit(f"too few pooled rows ({len(panel)}) to train")

    X = np.nan_to_num(panel[features].values, nan=0.0, posinf=0.0, neginf=0.0)
    y = panel["label"].values.astype(int)

    # causal split: oldest 80% train, next 5% val, last 15% test (chronological)
    n = len(panel)
    i_tr, i_val = int(n * 0.80), int(n * 0.85)
    Xtr, ytr = X[:i_tr], y[:i_tr]
    Xval, yval = X[i_tr:i_val], y[i_tr:i_val]
    Xte, yte = X[i_val:], y[i_val:]

    model = xgb.XGBClassifier(
        n_estimators=N_TREES, max_depth=MAX_DEPTH, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, objective="multi:softmax",
        num_class=3, n_jobs=N_JOBS, eval_metric="mlogloss",
        class_weight="balanced",
    )
    model.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)

    acc = float((model.predict(Xte) == yte).mean()) if len(yte) else float("nan")
    MODELS_DIR.mkdir(exist_ok=True)
    out = MODELS_DIR / "dex_xgb.json"
    model.save_model(str(out))
    meta = {"features": features, "n_tokens": int(n_tokens), "rows": int(n),
            "trained_at": datetime.now(timezone.utc).isoformat()}
    (MODELS_DIR / "dex_ml_meta.json").write_text(json.dumps(meta, indent=2))
    metrics = {"test_acc": acc, "train": int(i_tr), "val": int(i_val - i_tr),
               "test": int(n - i_val), "n_features": len(features)}
    (MODELS_DIR / "dex_ml_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  Test accuracy: {acc:.3f}")
    print(f"  Model saved: {out} ({out.stat().st_size/1024:.1f} KB)")
    print(f"  Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
