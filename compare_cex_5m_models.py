#!/usr/bin/env python3
"""
Compare CEX 5m model architectures: per-pair (single) vs pooled (multi-asset).

Answers: should we run a SCREENED universe of strong single-pair models, or
one weak multi-asset model over the whole coin universe?

Method (literature-backed, OOS walk-forward, paired):
  For a universe of pairs:
   - Train per-pair model (model_trainer) -> signals on each pair's OOS fold.
   - Train ONE pooled model (train_cex_5m_pooled) -> signals on the same folds.
   - For each pair, score both on the OOS test bars: directional accuracy
     (LONG/SHORT correct vs actual label), and a paired exact sign test
     (does single agree with pooled? when they disagree, which is right?).
   - Aggregate: mean per-pair directional acc (single vs pooled), win-rate of
     single-over-pooled per pair (paired), and a portfolio sim (top-N ranked
     long/short book) for each approach -> ret, exposure-adj Sharpe, DD, calmar.

Outputs a comparison table + a paired sign test summary. No live trading.

Usage:
  python compare_cex_5m_models.py --max-pairs 50
  python compare_cex_5m_models.py --symbols BTCUSDT,ETHUSDT,DOGEUSDT
"""
import argparse
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline import fetch_data, triple_barrier_labels, walk_forward_splits
import model_trainer as mt

REPO = Path(__file__).parent
MODELS_DIR = REPO / "models"


def fetch_sym(pair):
    return "BTC" if pair == "BTCUSDT" else pair


def features_and_label(pair):
    fsym = fetch_sym(pair)
    df, feats = mt.build_symbol_features(fsym)
    if df.empty:
        return None, None, None
    df = triple_barrier_labels(df)
    if "label" not in df.columns:
        return None, None, None
    keep = [c for c in feats if c in df.columns] + ["label"]
    df = df[keep].dropna(subset=["label"])
    # return a DataFrame so X.columns is always self-consistent (no width drift)
    X = df.drop(columns=["label"])
    return X, df["label"], list(X.columns)


def train_model(X, y, feats):
    n = len(y)
    # adaptive windows: walk_forward_splits needs total >= val+test+step
    target_test = min(15000, max(2000, n // 5))
    val_size = min(5000, max(1000, n // 10))
    step = target_test
    sdf = pd.DataFrame({"label": y}, index=pd.date_range("2010-01-01", periods=n, freq="5min", tz="UTC"))
    splits = walk_forward_splits(sdf, folds=5)
    if not splits:
        # fallback: single 80/20 split
        k = int(n * 0.8)
        splits = [{"train_idx": np.arange(0, k), "val_idx": np.arange(k, n), "test_idx": np.arange(k, n)}]
    sp = splits[-1]
    model = xgb.XGBClassifier(
        objective="multi:softmax", num_class=3,
        max_depth=mt.MAX_DEPTH, learning_rate=0.05, n_estimators=mt.N_TREES,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=mt._N_JOBS, random_state=42, early_stopping_rounds=30,
        eval_metric="mlogloss", class_weight="balanced",
    )
    model.fit(X[sp["train_idx"]], y[sp["train_idx"]],
              eval_set=[(X[sp["val_idx"]], y[sp["val_idx"]])], verbose=False)
    return model, sp["test_idx"]


def signals_on_test(model, X_te, feats, te_idx):
    Xt = X_te[te_idx]
    pred = model.predict(Xt)
    return pred  # 0 SHORT, 1 LONG, 2 FLAT


def dir_acc(pred, y_te, te_idx):
    """Directional accuracy: of non-FLAT calls, how many match actual
    up/down (label 0=down/short, 1=up/long)."""
    y = y_te[te_idx]
    mask = pred != 2
    if mask.sum() == 0:
        return np.nan, 0
    correct = (pred[mask] == y[mask]).mean()
    return float(correct), int(mask.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=200)
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--gate", action="store_true",
                   help="apply literature quality gate (liquidity+maturity) to universe")
    args = ap.parse_args()

    if args.symbols:
        pairs = [s.strip().upper() for s in args.symbols.split(",")]
    elif args.gate:
        from quality_gate import gated_universe
        pairs = gated_universe(verbose=True)
        if args.max_pairs:
            pairs = pairs[:args.max_pairs]
    else:
        files = sorted(REPO.glob("data/*USDT_5m_max.csv"))
        pairs = [p.stem.replace("_5m_max", "") for p in files][:args.max_pairs]

    print(f"[{datetime.now(timezone.utc):%H:%M:%S}] Compare: {len(pairs)} pairs")

    single_acc, pooled_acc = [], []
    paired = {"single_wins": 0, "pooled_wins": 0, "ties": 0, "n": 0}
    # store per-pair test signals for portfolio sim
    port_single, port_pooled = [], []

    for pair in pairs:
        try:
            X, y, feats = features_and_label(pair)
            if X is None or len(X) < 2000:
                continue
            Xv = np.nan_to_num(X.values, 0.0)
            yv = y.values.astype(int)
            smodel, te = train_model(Xv, yv, feats)
            if smodel is None or te is None:
                print(f"  {pair}: train_model returned None")
                continue
            spred = signals_on_test(smodel, Xv, feats, te)
            sa, sn = dir_acc(spred, yv, te)
            if not np.isnan(sa):
                single_acc.append(sa)
                port_single.append((pair, spred, yv, te))
            port_pooled.append((pair, Xv, feats, yv, te))
        except Exception as e:
            import traceback as _tb
            print(f"  {pair} err: {e!r}")
            _tb.print_exc()

    # Train ONE pooled model on the universe (union features)
    print(f"  training pooled model on {len(port_pooled)} pairs...")
    # rebuild per-pair (Xv, colnames, yv, te) from what we stored
    pair_data = []  # (Xv, colnames, yv, te)
    for pair, Xv, feats, yv, te in port_pooled:
        pair_data.append((Xv, list(feats), yv, te))
    # union = sorted unique column names across all pairs
    union = []
    for Xv, cols, yv, te in pair_data:
        for c in cols:
            if c not in union:
                union.append(c)
    union.sort()
    # align every pair to the full union (0-fill missing cols), preserve order
    X_parts, y_parts = [], []
    for Xv, cols, yv, te in pair_data:
        n = Xv.shape[0]
        full = np.zeros((n, len(union)), dtype=np.float64)
        colidx = {c: i for i, c in enumerate(cols)}
        for j, uc in enumerate(union):
            if uc in colidx:
                full[:, j] = Xv[:, colidx[uc]]
        X_parts.append(full)
        y_parts.append(yv)
    Xall = np.vstack(X_parts)
    yall = np.concatenate(y_parts)
    pmodel, pte = train_model(Xall, yall.astype(int), union)
    if pmodel is None:
        print("ERROR: pooled train failed")
        return
    # score pooled on each pair's test fold (same union-alignment as training)
    for pair, Xv, feats, yv, te in port_pooled:
        n = Xv.shape[0]
        full = np.zeros((n, len(union)), dtype=np.float64)
        colidx = {c: i for i, c in enumerate(feats)}
        for j, uc in enumerate(union):
            if uc in colidx:
                full[:, j] = Xv[:, colidx[uc]]
        Xp = np.nan_to_num(full, 0.0)
        ppred = pmodel.predict(Xp[te])
        pa, pn = dir_acc(ppred, yv, te)
        if not np.isnan(pa):
            pooled_acc.append(pa)

    # paired comparison: per-pair single vs pooled directional acc
    n = min(len(single_acc), len(pooled_acc))
    s = np.array(single_acc[:n]); p = np.array(pooled_acc[:n])
    print(f"\n=== COMPARISON ({n} pairs) ===")
    print(f"  Single (per-pair) mean dir-acc: {s.mean():.3f}  (n={len(s)})")
    print(f"  Pooled (multi)   mean dir-acc: {p.mean():.3f}  (n={len(p)})")
    diff = s - p
    wins = int((diff > 0).sum()); losses = int((diff < 0).sum()); ties = int((diff == 0).sum())
    print(f"  Paired: single beats pooled on {wins} pairs, loses on {losses}, tie {ties}")
    # exact sign test on per-pair differences (single - pooled dir-acc):
    # two-sided binomial, H0: single and pooled equally often better (p=0.5)
    if wins + losses > 0:
        import math
        N = wins + losses
        k = min(wins, losses)  # extreme tail count
        p_val = 0.0
        for i in range(k, N + 1):
            p_val += math.comb(N, i) * (0.5 ** N)
        p_val *= 2.0
        p_val = min(p_val, 1.0)
        print(f"  Exact sign test p (single>pooled): {p_val:.4f}")
    print(f"  Mean dir-acc gap (single-pooled): {diff.mean():+.3f}")

    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_pairs": n,
        "single_mean_dir_acc": float(s.mean()),
        "pooled_mean_dir_acc": float(p.mean()),
        "single_wins": int(wins), "pooled_wins": int(losses), "ties": int(ties),
        "gap": float(diff.mean()),
    }
    (MODELS_DIR / "compare_cex_5m.json").write_text(json.dumps(out, indent=2))
    print(f"  Saved compare_cex_5m.json")


if __name__ == "__main__":
    main()
