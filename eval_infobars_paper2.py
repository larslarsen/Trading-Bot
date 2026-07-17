#!/usr/bin/env python3
"""
Paper #2 OOS verdict: information-driven bars (DNB 2024 / Springer 2025)
vs fixed 5m time bars. Literature: triple-barrier on volume/order-flow
bars outperforms time bars + next-bar, positive after costs.

Method: per gated pair, build features (flow cols kept from paper #1).
  baseline: triple_barrier_labels on 5m TIME bars -> time_acc
  info:     resample to volume-targeted bars -> triple_barrier_labels -> info_acc
Paired exact sign test (info vs time) across pairs. OOS-gated.
"""
import sys, time, math
import numpy as np
import pandas as pd
import pathlib
sys.path.insert(0, ".")
import xgboost as xgb
import canonical_features as cf
import model_trainer as mt
import quality_gate as qg
from pipeline import triple_barrier_labels, walk_forward_splits

MODELS = pathlib.Path("models")


def info_bars(df, vol_target):
    """Resample df to volume-targeted bars. Each bar = vol_target cumulative
    volume. Aggregates OHLCV; carries forward last feature values per group."""
    df = df.copy()
    if "volume" not in df.columns:
        return df
    df["_cumvol"] = df["volume"].cumsum()
    # group boundaries where cumulative volume crosses k*vol_target
    edges = np.arange(vol_target, df["_cumvol"].iloc[-1], vol_target)
    df["_grp"] = np.searchsorted(edges, df["_cumvol"].values)
    feat_cols = [c for c in df.columns if c not in
                 ("open", "high", "low", "close", "volume", "_cumvol", "_grp", "label")]
    agg = {c: "last" for c in feat_cols}
    agg.update({"open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum"})
    res = df.groupby("_grp").agg(agg).dropna(subset=["close"])
    res = res.sort_index()
    return res


def oos_dir_acc(model, X, y):
    n = len(y)
    if n < 5000:
        return []
    sdf = pd.DataFrame({"label": y},
                       index=pd.date_range("2010-01-01", periods=n, freq="5min"))
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


def load_time_model(sym):
    p = MODELS / f"{sym.replace('USDT','').lower()}_xgb.json"
    if not p.exists():
        return None
    m = xgb.XGBClassifier(); m.load_model(str(p)); return m


def train_on_labels(sym, df, feats, label_df):
    keep = [c for c in feats if c in label_df.columns] + ["label"] + \
           ["open", "high", "low", "close", "volume"]
    sub = label_df[[c for c in keep if c in label_df.columns]].dropna(subset=["label"])
    if len(sub) < 2000:
        return None
    X = np.nan_to_num(sub.drop(columns=["label", "open", "high", "low", "close", "volume"]).values, 0.0)
    y = sub["label"].values.astype(int)
    n = len(y); i = int(n * 0.8)
    m = xgb.XGBClassifier(objective="multi:softmax", num_class=3,
        max_depth=mt.MAX_DEPTH, learning_rate=0.05, n_estimators=mt.N_TREES,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=mt._N_JOBS, random_state=42, early_stopping_rounds=30,
        eval_metric="mlogloss", class_weight="balanced")
    m.fit(X[:i], y[:i], eval_set=[(X[i:], y[i:])], verbose=False)
    return m, X, y


def main():
    t0 = time.time()
    uni = qg.gated_universe()
    time_acc, info_acc = [], []
    for sym in uni:
        try:
            df, feats = mt.build_symbol_features(sym)
            if df is None or df.empty:
                continue
            # baseline: time bars
            tb = triple_barrier_labels(df.copy())
            tm = train_on_labels(sym, df, feats, tb)
            if tm is None:
                continue
            tmodel, tX, ty = tm
            # info bars: volume-targeted. Target ~60000 bars so walk_forward_splits
            # (needs >=35000 for a fold) and oos_dir_acc (n>=5000) can evaluate.
            vol_target = max(1, int(df["volume"].sum() / 60000))  # ~60000 bars
            ib = info_bars(df, vol_target)
            ib = triple_barrier_labels(ib)
            im = train_on_labels(sym, df, feats, ib)
            if im is None:
                continue
            imodel, iX, iy = im
            ta = oos_dir_acc(tmodel, tX, ty)
            ia = oos_dir_acc(imodel, iX, iy)
            if ta and ia:
                time_acc.append(np.mean(ta)); info_acc.append(np.mean(ia))
                print(f"  {sym}: time={np.mean(ta):.3f} info={np.mean(ia):.3f}")
        except Exception as e:
            print(f"  {sym} err {e!r}")
    time_acc = np.array(time_acc); info_acc = np.array(info_acc)
    if len(time_acc) == 0:
        print("n=0 — no pairs evaluated"); return
    diff = info_acc - time_acc
    wins = int((diff > 0).sum()); losses = int((diff < 0).sum()); ties = int((diff == 0).sum())
    N = wins + losses; p = 1.0
    if N > 0:
        k = max(wins, losses); pv = 0.0
        for i in range(k, N + 1):
            pv += math.comb(N, i) * (0.5 ** N)
        p = min(pv * 2.0, 1.0)
    print(f"\n=== PAPER #2: info-bar vs time-bar OOS dir-acc (n={len(time_acc)}) ===")
    print(f"  time   mean={time_acc.mean():.3f}")
    print(f"  info   mean={info_acc.mean():.3f}")
    print(f"  info beats time on {wins} pairs, loses {losses}, tie {ties}")
    print(f"  Exact sign test p(info>time) = {p:.4f}")
    print(f"  VERDICT: {'INFO BARS HELP (p<0.05)' if p<0.05 and wins>losses else 'info bars do NOT beat time bars'}")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
