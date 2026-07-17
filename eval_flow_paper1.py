#!/usr/bin/env python3
"""
Paper #1 OOS verdict: order-flow features (Anastasopoulos 2024) vs no-flow.
Paired exact sign test on per-pair OOS directional accuracy (the 1d gold
standard: walk_forward OOS + paired sign test, NOT in-sample acc).

For each gated pair:
  - FLOW model  = on-disk models/<sym>_xgb.json (trained WITH flow cols)
  - NO-FLOW     = retrained here with flow cols STRIPPED from ALWAYS
                  (monkeypatch canonical_features) -> .pooled_tmp/noflow/<sym>.json
Both evaluated via walk_forward_splits OOS -> per-fold dir-acc.
Paired exact sign test: flow dir-acc vs no-flow dir-acc across pairs.
"""
import sys, json, time, shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))
from quality_gate import gated_universe
from pipeline import triple_barrier_labels, walk_forward_splits
import model_trainer as mt
import canonical_features as cf

MODELS = REPO / "models"
NOFLOW = REPO / ".pooled_tmp/noflow"
NOFLOW.mkdir(parents=True, exist_ok=True)


def strip_flow_train(sym, df=None, feats=None):
    """Retrain with flow cols removed from ALWAYS -> no-flow model in NOFLOW.
    Reuses a prebuilt (df, feats) if given (avoids double fetch)."""
    cf.ALWAYS = [c for c in cf.ALWAYS if c not in
                 ("taker_buy_sell_ratio", "imbalance", "trade_count", "spread")]
    if df is None:
        df, feats = mt.build_symbol_features(sym)
    if df is None or df.empty:
        return None
    df = triple_barrier_labels(df)
    if "label" not in df.columns:
        return None
    keep = [c for c in feats if c in df.columns] + ["label"] + ["open","high","low","close","volume"]
    df = df[keep].dropna(subset=["label"])
    if len(df) < 2000:
        return None
    X = np.nan_to_num(df.drop(columns=["label","open","high","low","close","volume"]).values, 0.0)
    y = df["label"].values.astype(int)
    n = len(y); i = int(n * 0.8)
    m = xgb.XGBClassifier(objective="multi:softmax", num_class=3,
        max_depth=mt.MAX_DEPTH, learning_rate=0.05, n_estimators=mt.N_TREES,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=mt._N_JOBS, random_state=42, early_stopping_rounds=30,
        eval_metric="mlogloss", class_weight="balanced")
    m.fit(X[:i], y[:i], eval_set=[(X[i:], y[i:])], verbose=False)
    out = NOFLOW / f"{sym.replace('USDT','')}_xgb.json"
    m.save_model(str(out))
    return out


def oos_dir_acc(model, X, y):
    n = len(y)
    sdf = pd.DataFrame({"label": y}, index=pd.date_range("2010", periods=n, freq="5min", tz="UTC"))
    splits = walk_forward_splits(sdf, folds=5)
    if not splits:
        return []
    accs = []
    for sp in splits:
        te = sp["test_idx"]
        pred = model.predict(X[te])
        mask = pred != 2
        if mask.sum() > 0:
            accs.append((pred[mask] == y[te][mask]).mean())
    return accs


def load_flow(sym):
    p = MODELS / f"{sym.replace('USDT','').lower()}_xgb.json"
    if not p.exists():
        return None
    m = xgb.XGBClassifier(); m.load_model(str(p)); return m


def main():
    t0 = time.time()
    cf.ALWAYS = ["regime_high_vol", "regime_trending", "funding_rate",
                  "taker_buy_sell_ratio", "imbalance", "trade_count", "spread"]
    uni = gated_universe()
    flow_acc, noflow_acc = [], []
    for sym in uni:
        try:
            df, feats = mt.build_symbol_features(sym)
            if df.empty:
                continue
            df = triple_barrier_labels(df)
            keep = [c for c in feats if c in df.columns] + ["label"] + ["open","high","low","close","volume"]
            df = df[keep].dropna(subset=["label"])
            if len(df) < 2000:
                continue
            X = np.nan_to_num(df.drop(columns=["label","open","high","low","close","volume"]).values, 0.0)
            y = df["label"].values.astype(int)
            fm = load_flow(sym)
            nm_path = strip_flow_train(sym, df=df, feats=feats)
            if fm is None or nm_path is None:
                continue
            nm = xgb.XGBClassifier(); nm.load_model(str(nm_path))
            fa = oos_dir_acc(fm, X, y)
            na = oos_dir_acc(nm, X, y)
            if fa and na:
                flow_acc.append(np.mean(fa)); noflow_acc.append(np.mean(na))
                print(f"  {sym}: flow={np.mean(fa):.3f} noflow={np.mean(na):.3f}")
        except Exception as e:
            print(f"  {sym} err {e!r}")
    flow_acc = np.array(flow_acc); noflow_acc = np.array(noflow_acc)
    diff = flow_acc - noflow_acc
    wins = int((diff > 0).sum()); losses = int((diff < 0).sum()); ties = int((diff == 0).sum())
    N = wins + losses
    p = 1.0
    if N > 0:
        import math
        k = min(wins, losses)
        pv = 0.0
        for i in range(k, N + 1):
            pv += math.comb(N, i) * (0.5 ** N)
        p = min(pv * 2.0, 1.0)
    print(f"\n=== PAPER #1: flow vs no-flow OOS dir-acc (n={len(flow_acc)}) ===")
    print(f"  flow   mean={flow_acc.mean():.3f}")
    print(f"  noflow mean={noflow_acc.mean():.3f}")
    print(f"  flow beats noflow on {wins} pairs, loses {losses}, tie {ties}")
    print(f"  Exact sign test p(flow>noflow) = {p:.4f}")
    print(f"  VERDICT: {'FLOW HELPS (p<0.05)' if p<0.05 and wins>losses else 'flow does NOT beat no-flow'}")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
