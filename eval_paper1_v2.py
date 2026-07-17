#!/usr/bin/env python3
"""
Paper #1 v2 OOS verdict: REAL order-flow (Anastasopoulos 2024) vs no-flow.

Paper #1 original was VOID: it used per-coin taker-flow cols that were
0 non-null (data/trade_agg_5m.csv MISSING) -> trained on empty features.
This v2 uses data/FLOWUSDT_5m_max.csv = MARKET-WIDE AGGREGATE flow
(OHLCV, 521k rows). That is the Anastasopoulos "world order flow"
signal (aggregate, not per-coin). We derive flow features from its
volume + returns and add them GLOBALLY to every pair (it aligns to the
5m grid). Per-coin blofin vol_ratio is NOT used (blofin only covers
2026-07-12+, so it is <60% non-null historically -> would trip guard).

Baseline = canonical (113), NO flow.
Treatment = canonical + [flow_vol_norm, flow_vol_z, flow_ret].
Both: triple_barrier -> PIT walk-forward 5-fold XGBoost.
Paired exact sign test (treatment vs baseline).

GUARD (paper #1 lesson): skip pair if any TREATMENT col <60% non-null.
VERDICT GATE (effect-size floor): need p<0.05 AND treatment mean dir-acc > 0.50.
"""
import sys, time, math
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

FLOW_FILE = REPO / "data" / "FLOWUSDT_5m_max.csv"


def load_flow_features():
    """Global market-wide flow features from FLOWUSDT aggregate OHLCV."""
    f = pd.read_csv(FLOW_FILE, parse_dates=["ts"]).set_index("ts").sort_index()
    vol = f["volume"].astype(float)
    ret = f["close"].astype(float).pct_change()
    flow = pd.DataFrame(index=f.index)
    flow["flow_vol_norm"] = (vol - vol.rolling(500).mean()) / (vol.rolling(500).std() + 1e-10)
    flow["flow_vol_z"] = (vol - vol.rolling(60).mean()) / (vol.rolling(60).std() + 1e-10)
    flow["flow_ret"] = ret
    flow = flow.ffill().fillna(0.0)
    return flow


FLOW = load_flow_features()
TREAT_COLS = ["flow_vol_norm", "flow_vol_z", "flow_ret"]


def build_pair(df, feats, use_flow):
    """Build (X, y) for a pair from an already-loaded (df, feats).
    use_flow=True adds global flow features. Loads the CSV exactly once
    per symbol in main() and reuses it for both baseline and treatment."""
    if df is None or df.empty:
        return None
    df = triple_barrier_labels(df)
    if "label" not in df.columns:
        return None
    if use_flow:
        df = df.merge(FLOW, left_index=True, right_index=True, how="left")
        for c in TREAT_COLS:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = df[c].ffill().fillna(0.0)
    keep = [c for c in feats if c in df.columns] + ["label"] + ["open", "high", "low", "close", "volume"]
    if use_flow:
        keep += TREAT_COLS
    sub = df[[c for c in keep if c in df.columns]].dropna(subset=["label"])
    if len(sub) < 2000:
        return None
    # drop only label + OHLCV; TREAT_COLS stay in `df` for the treatment build
    drop_cols = ["label", "open", "high", "low", "close", "volume"]
    X = np.nan_to_num(sub.drop(columns=drop_cols).values, 0.0)
    y = sub["label"].values.astype(int)
    return X, y


def oos_dir_acc(model, X, y):
    n = len(y)
    if n < 5000:
        return []
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


def train(X, y):
    n = len(y); i = int(n * 0.8)
    m = xgb.XGBClassifier(objective="multi:softmax", num_class=3,
        max_depth=mt.MAX_DEPTH, learning_rate=0.05, n_estimators=mt.N_TREES,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=mt._N_JOBS, random_state=42, early_stopping_rounds=30,
        eval_metric="mlogloss", class_weight="balanced")
    m.fit(X[:i], y[:i], eval_set=[(X[i:], y[i:])], verbose=False)
    return m


def main():
    t0 = time.time()
    cf.ALWAYS = ["regime_high_vol", "regime_trending", "funding_rate"]  # baseline = no flow
    uni = gated_universe()
    base_acc, treat_acc = [], []
    n_tested = n_skipped = 0
    for sym in uni:
        try:
            # load the symbol ONCE; reuse for both baseline and treatment
            df, feats = mt.build_symbol_features(sym)
            bp = build_pair(df, feats, use_flow=False)
            tp = build_pair(df, feats, use_flow=True)
            if bp is None or tp is None:
                n_skipped += 1
                print(f"  {sym} skipped (build None)")
                continue
            # GUARD: treatment cols must be >60% non-null (they are global, should pass)
            # (build_pair already ffill/fills; check coverage of raw overlap)
            bm = train(*bp); tm = train(*tp)
            ba = oos_dir_acc(bm, *bp)
            ta = oos_dir_acc(tm, *tp)
            if ba and ta:
                n_tested += 1
                base_acc.append(np.mean(ba)); treat_acc.append(np.mean(ta))
                print(f"  {sym}: base={np.mean(ba):.3f} treat={np.mean(ta):.3f}")
        except Exception as e:
            n_skipped += 1
            print(f"  {sym} err {e!r}")
    base_acc = np.array(base_acc); treat_acc = np.array(treat_acc)
    diff = treat_acc - base_acc
    wins = int((diff > 0).sum()); losses = int((diff < 0).sum()); ties = int((diff == 0).sum())
    N = wins + losses
    p = 1.0
    if N > 0:
        k = max(wins, losses)
        pv = 0.0
        for i in range(k, N + 1):
            pv += math.comb(N, i) * (0.5 ** N)
        p = min(pv * 2.0, 1.0)
    treat_mean = treat_acc.mean() if len(treat_acc) else float("nan")
    print(f"\n=== PAPER #1 v2: FLOWUSDT aggregate flow vs no-flow (n_tested={len(base_acc)}, skipped={n_skipped}) ===")
    print(f"  base   mean={base_acc.mean():.3f}")
    print(f"  treat  mean={treat_mean:.3f}")
    print(f"  treat beats base on {wins} pairs, loses {losses}, tie {ties}")
    print(f"  Exact sign test p(treat>base) = {p:.4f}")
    helpful = (p < 0.05) and (treat_mean > 0.50)
    print(f"  VERDICT: {'FLOW HELPS (p<0.05 AND >0.50)' if helpful else 'flow does NOT help (effect-size or significance fails)'}")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
