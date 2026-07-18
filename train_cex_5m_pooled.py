#!/usr/bin/env python3
"""
Pooled CEX 5m model trainer -- ONE XGBoost over the whole coin universe.

Contrast to model_trainer.train_and_save (per-pair, one model per coin):
this stacks every CEX 5m pair into a single labelled panel and trains ONE
model that sees all coins' feature distributions at once. The experiment
(compare_cex_5m_models.py) scores both approaches walk-forward to decide
whether a screened universe of strong single-pair models beats one weak
multi-asset model on the full universe.

Feature build reuses model_trainer.build_symbol_features so the pooled model
uses the EXACT same pipeline (resampled + macro + multi-asset + micro +
funding + on-chain + DEX breadth + regime) as the per-pair models and the
serving bot. Pairs with missing exogenous features (no funding/on-chain) get
those columns 0-filled so the union feature set is uniform.

Usage:
  python train_cex_5m_pooled.py                      # full Binance universe
  python train_cex_5m_pooled.py --max-pairs 50       # cap for faster iteration
  python train_cex_5m_pooled.py --symbols BTCUSDT,ETHUSDT,DOGEUSDT
"""
import argparse
import json
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import os
import gc
import shutil

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline import fetch_data, triple_barrier_labels, walk_forward_splits
import model_trainer as mt

REPO = Path(__file__).parent
MODELS_DIR = REPO / "models"
OUT = MODELS_DIR / "cex_5m_pooled_xgb.json"
META = MODELS_DIR / "cex_5m_pooled_meta.json"


def universe(max_pairs=None, symbols=None, gate=False):
    if symbols:
        return [s.strip().upper() for s in symbols.split(",") if s.strip()]
    if gate:
        # Literature quality gate (Bartolucci two features: security/maturity
        # + liquidity). Train the multi-asset model on the SCREENED subset only
        # so negative transfer from thin/dead coins is excluded.
        from quality_gate import gated_universe, save_gate
        pairs = gated_universe()
        save_gate(pairs)
        if max_pairs:
            pairs = pairs[:max_pairs]
        return pairs
    # Binance 5m universe (unsuffixed canonical files)
    files = sorted(REPO.glob("data/*USDT_5m_max.csv"))
    pairs = [p.stem.replace("_5m_max", "") for p in files]
    if max_pairs:
        pairs = pairs[:max_pairs]
    return pairs


def _canonical_features():
    """BTC carries the full exogenous set; use its feature list as the
    canonical/union column order (matches original union semantics:
    missing columns are 0-filled per pair)."""
    _, feats = mt.build_symbol_features("BTC")
    return list(feats)


def build_pooled_panel(pairs):
    """Stream every pair's features to DISK so peak RAM stays at ONE pair.

    The original held all per-pair frames in a list AND np.vstacked the whole
    universe into one dense array -> ~12 GB (gated) to ~180 GB (full 469),
    which OOM-killed the box. Here we write each pair's aligned block to a
    per-pair .npy, then assemble into a single disk-backed memmap. The feature
    build happens once; assembly only ever holds one pair in RAM.

    Returns (X_path, y_path, feat_set, n_rows, tmp_dir).
    """
    feat_set = _canonical_features()
    n_f = len(feat_set)
    # Real disk, NOT /tmp (which is a 16 GB tmpfs and would overflow on the
    # 469-pair universe). 691 GB free on the repo volume.
    tmp_dir = str(REPO / ".pooled_tmp" / f"run_{os.getpid()}_{int(time.time())}")
    os.makedirs(tmp_dir, exist_ok=True)
    n = 0
    parts = []
    for sym in pairs:
        fsym = "BTC" if sym == "BTCUSDT" else sym
        try:
            df, feats = mt.build_symbol_features(fsym)
        except Exception as e:
            print(f"  skip {sym}: {e!r}"[:120]); continue
        df = triple_barrier_labels(df)
        if "label" not in df.columns:
            del df; gc.collect(); continue
        keep = [c for c in feats if c in df.columns] + ["label"]
        sub = df[keep].dropna(subset=["label"])
        m = len(sub)
        if m == 0:
            del df, sub; gc.collect(); continue
        # align to canonical feat_set, 0-fill missing cols
        arr = np.zeros((m, n_f), dtype=np.float32)
        for j, f in enumerate(feat_set):
            if f in sub.columns:
                arr[:, j] = sub[f].to_numpy(dtype=np.float32)
        yp = sub["label"].to_numpy(dtype=np.int8)
        xp = Path(tmp_dir) / f"X_{len(parts)}.npy"
        ypp = Path(tmp_dir) / f"y_{len(parts)}.npy"
        np.save(xp, arr); np.save(ypp, yp)
        parts.append((str(xp), str(ypp), m))
        n += m
        del df, sub, arr; gc.collect()
    if n == 0:
        return None, None, None, 0, None
    X_path = Path(tmp_dir) / "X.dat"
    Y_path = Path(tmp_dir) / "y.dat"
    X = np.memmap(X_path, dtype=np.float32, mode="w+", shape=(n, n_f))
    Y = np.memmap(Y_path, dtype=np.int8, mode="w+", shape=(n,))
    off = 0
    for xp, ypp, m in parts:
        xi = np.load(xp, mmap_mode="r")
        yi = np.load(ypp, mmap_mode="r")
        X[off:off+m] = xi
        Y[off:off+m] = yi
        off += m
        try: os.remove(xp); os.remove(ypp)
        except OSError: pass
    X.flush(); Y.flush()
    return str(X_path), str(Y_path), feat_set, n, tmp_dir


def _dump_csv(x_path, y_path, n, n_f, idx, path, block=500000):
    """Stream a row-index subset of the disk-backed panel to a CSV file
    (first col = label) for XGBoost external-memory DMatrix."""
    X = np.memmap(x_path, dtype=np.float32, mode="r", shape=(n, n_f))
    y = np.memmap(y_path, dtype=np.int8, mode="r", shape=(n,))
    with open(path, "w") as fh:
        for s in range(0, len(idx), block):
            sl = idx[s:s+block]
            M = np.hstack([y[sl].astype(np.float32)[:, None], X[sl].astype(np.float32)])
            np.savetxt(fh, M, delimiter=",", fmt="%.6g")


class PanelIter(xgb.DataIter):
    """Stream a row-index subset of the disk-backed panel to XGBoost in
    fixed-size batches (external memory). Peak RAM ~= one batch, not the
    whole universe. Measured: 4M rows -> 2.4 GB peak."""
    def __init__(self, x_path, y_path, n, n_f, idx, shard=500_000):
        self.x_path = x_path; self.y_path = y_path
        self.n = n; self.n_f = n_f; self.idx = idx; self.shard = shard
        self.it = 0
        super().__init__()
    def next(self, input_data):
        s = self.it * self.shard
        e = min(s + self.shard, len(self.idx))
        if s >= len(self.idx):
            return 0
        rows = self.idx[s:e]
        Xb = np.memmap(self.x_path, dtype=np.float32, mode="r", shape=(self.n, self.n_f))[rows]
        yb = np.memmap(self.y_path, dtype=np.int8, mode="r", shape=(self.n,))[rows]
        input_data(data=Xb, label=yb)
        self.it += 1
        return 1
    def reset(self):
        self.it = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--gate", action="store_true",
                   help="train on the literature quality-gated (screened) "
                        "universe only, not the full Binance set")
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    pairs = universe(args.max_pairs, args.symbols, gate=args.gate)
    print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}] Pooled CEX 5m train: {len(pairs)} pairs")

    t0 = time.time()
    X_path, y_path, feat_set, n_rows, tmp_dir = build_pooled_panel(pairs)
    if X_path is None:
        print("ERROR: no panel built")
        return
    print(f"  Pooled panel: {n_rows} rows, {len(feat_set)} features (disk-backed: {X_path})")

    # walk-forward on the pooled panel (time-ordered, synthetic index).
    # Only the LENGTH matters to walk_forward_splits, so use a lazy RangeIndex
    # instead of materializing a 13M-row DataFrame (was ~1 GB wasted RAM).
    sdf = pd.DataFrame(index=pd.RangeIndex(n_rows))
    splits = walk_forward_splits(sdf, folds=args.folds)
    if not splits:
        print("ERROR: no walk-forward splits")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return
    sp = splits[-1]
    tr, va, te = sp["train_idx"], sp["val_idx"], sp["test_idx"]
    print(f"  Train={len(tr)} Val={len(va)} Test={len(te)}")

    # True external-memory train: stream the (large) train slice from the disk
    # memmap via a DataIter -> ExtMemQuantileDMatrix. XGBoost pages one batch
    # at a time (caches histograms to disk), so peak RAM ~= ONE batch, NOT the
    # whole universe. Measured: 4M rows -> 2.4 GB peak (was ~12 GB with a
    # single csv?format=csv DMatrix, which parses the ENTIRE slice into float64
    # RAM and OOMs at 469 pairs). Val/test are tiny -> plain in-RAM DMatrix.
    va_csv = Path(tmp_dir) / "val.csv"
    te_csv = Path(tmp_dir) / "test.csv"
    _dump_csv(X_path, y_path, n_rows, len(feat_set), va, str(va_csv))
    _dump_csv(X_path, y_path, n_rows, len(feat_set), te, str(te_csv))

    dtrain = xgb.ExtMemQuantileDMatrix(PanelIter(X_path, y_path, n_rows, len(feat_set), tr))
    dval = xgb.DMatrix(f"{va_csv}?format=csv&label_column=0")
    dtest = xgb.DMatrix(f"{te_csv}?format=csv&label_column=0")

    params = dict(
        objective="multi:softmax", num_class=3,
        max_depth=mt.MAX_DEPTH, learning_rate=0.05, n_estimators=mt.N_TREES,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, n_jobs=mt._N_JOBS,
        eval_metric="mlogloss",
    )
    bst = xgb.train(
        params, dtrain, num_boost_round=mt.N_TREES,
        evals=[(dval, "val")], verbose_eval=False,
        early_stopping_rounds=30,
    )
    best = int(bst.best_iteration) if getattr(bst, "best_iteration", None) else mt.N_TREES
    preds = bst.predict(dtest)
    y_te = np.memmap(y_path, dtype=np.int8, mode="r", shape=(n_rows,))[te]
    acc = float((preds == y_te).mean())
    print(f"  Trained {best} rounds. Test acc={acc:.3f}")

    bst.save_model(str(OUT))
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "pooled_multi_asset",
        "pairs": pairs,
        "n_pairs": len(pairs),
        "features": feat_set,
        "feature_count": len(feat_set),
        "train_rows": int(len(tr)),
        "test_rows": int(len(te)),
        "trees": best,
        "accuracy": acc,
        "elapsed_sec": round(time.time() - t0, 1),
    }
    META.write_text(json.dumps(meta, indent=2))
    print(f"  Saved {OUT} ({OUT.stat().st_size/1024:.1f} KB); meta {META}")
    print(f"  Done in {meta['elapsed_sec']:.1f}s")

    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
