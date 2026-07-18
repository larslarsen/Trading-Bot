#!/usr/bin/env python3
"""
Paper #2 v2 OOS verdict: information-driven (volume) bars (DNB 2024 /
Springer 2025) vs fixed 5m time bars. Re-confirm + venue robustness.

Original (eval_infobars_paper2.py) gave info 0.601 vs time 0.449 on
Binance volume -> STANDS. v2:
  Treatment A: info bars on Binance volume (~60000 bars) -> confirm 0.601.
  Treatment B: info bars on BLOFIN volume, but blofin only covers the
  recent window (2026-07-12+). So build B on the overlapping recent slice
  -> venue-robustness check (does the edge survive on a 2nd exchange?).
Baseline: triple_barrier on 5m time bars (Binance).
Both: PIT walk-forward 5-fold XGBoost. Paired exact sign test.

GUARD: info-bar row count >= 35000 (walk_forward_splits min) else skip.
VERDICT GATE: p<0.05 AND mean dir-acc > 0.50.
"""
import sys, time, math
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))
import model_trainer as mt
import quality_gate as qg
from pipeline import triple_barrier_labels, walk_forward_splits

BLOFIN_DIR = REPO / "data"


def info_bars(df, vol_target):
    df = df.copy()
    if "volume" not in df.columns:
        return df
    df["_cumvol"] = df["volume"].cumsum()
    edges = np.arange(vol_target, df["_cumvol"].iloc[-1], vol_target)
    df["_grp"] = np.searchsorted(edges, df["_cumvol"].values)
    feat_cols = [c for c in df.columns if c not in
                 ("open", "high", "low", "close", "volume", "_cumvol", "_grp", "label")]
    agg = {c: "last" for c in feat_cols}
    agg.update({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    res = df.groupby("_grp").agg(agg).dropna(subset=["close"]).sort_index()
    return res


def oos_dir_acc(model, X, y):
    n = len(y)
    if n < 5000:
        return []
    sdf = pd.DataFrame({"label": y}, index=pd.date_range("2010", periods=n, freq="5min"))
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


def build_labels(df, feats, vol_source):
    """vol_source: 'binance' uses df.volume; 'blofin' uses blofin volume col."""
    if vol_source == "blofin":
        # need blofin volume merged; caller passes df with 'blofin_volume'
        if "blofin_volume" not in df.columns:
            return None
        df = df.rename(columns={"blofin_volume": "volume"})
    df = triple_barrier_labels(df)
    if "label" not in df.columns:
        return None
    keep = [c for c in feats if c in df.columns] + ["label"] + ["open", "high", "low", "close", "volume"]
    sub = df[[c for c in keep if c in df.columns]].dropna(subset=["label"])
    if len(sub) < 2000:
        return None
    X = np.nan_to_num(sub.drop(columns=["label", "open", "high", "low", "close", "volume"]).values, 0.0)
    y = sub["label"].values.astype(int)
    return X, y


def load_blofin(sym):
    p = BLOFIN_DIR / f"{sym}_5m_blofin_max.csv"
    if not p.exists():
        return None
    b = pd.read_csv(p, parse_dates=["ts"]).set_index("ts").sort_index()
    b = b[~b.index.duplicated()]
    return b


def main():
    t0 = time.time()
    uni = qg.gated_universe()
    time_acc, infoA_acc, infoB_acc = [], [], []
    n_tested = n_skipped = 0
    for sym in uni:
        try:
            df, feats = mt.build_symbol_features(sym)
            if df is None or df.empty:
                continue
            df = df[~df.index.duplicated()]
            # baseline: time bars (Binance)
            bp = build_labels(df.copy(), feats, "binance")
            if bp is None:
                n_skipped += 1; print(f"  {sym} skipped (time build None)"); continue
            # treatment A: Binance-volume info bars
            vol_target = max(1, int(df["volume"].sum() / 60000))
            ibA = info_bars(df.copy(), vol_target)
            if len(ibA) < 35000:
                n_skipped += 1; print(f"  {sym} skipped (infoA<35000)"); continue
            ap = build_labels(ibA, feats, "binance")
            if ap is None:
                n_skipped += 1; print(f"  {sym} skipped (infoA build None)"); continue
            bm = train(*bp); am = train(*ap)
            ta = oos_dir_acc(bm, *bp); aa = oos_dir_acc(am, *ap)
            # treatment B: blofin-volume info bars on overlap window
            bb = load_blofin(sym)
            ib_val = None
            if bb is not None and "volume" in bb.columns:
                merged = df.copy()
                merged["blofin_volume"] = bb["volume"]
                merged["blofin_volume"] = merged["blofin_volume"].ffill().fillna(0.0)
                if merged["blofin_volume"].sum() > 0:
                    vt2 = max(1, int(merged["blofin_volume"].sum() / 60000))
                    ibB = info_bars(merged, vt2)
                    if len(ibB) >= 35000:
                        bpB = build_labels(ibB, feats, "blofin")
                        if bpB is not None:
                            bBm = train(*bpB)
                            ib_val = np.mean(oos_dir_acc(bBm, *bpB)) if oos_dir_acc(bBm, *bpB) else None
            if ta and aa:
                n_tested += 1
                time_acc.append(np.mean(ta)); infoA_acc.append(np.mean(aa))
                infoB_acc.append(ib_val if ib_val is not None else float("nan"))
                print(f"  {sym}: time={np.mean(ta):.3f} infoA={np.mean(aa):.3f} infoB={infoB_acc[-1]:.3f}")
        except Exception as e:
            n_skipped += 1
            print(f"  {sym} err {e!r}")
    time_acc = np.array(time_acc); infoA_acc = np.array(infoA_acc)
    print(f"\n=== PAPER #2 v2: info-bar vs time-bar (n_tested={len(time_acc)}, skipped={n_skipped}) ===")
    print(f"  time    mean={time_acc.mean():.3f}")
    print(f"  infoA(Binance vol) mean={infoA_acc.mean():.3f}")
    # sign test A vs time
    diff = infoA_acc - time_acc
    wins = int((diff > 0).sum()); losses = int((diff < 0).sum()); ties = int((diff == 0).sum())
    N = wins + losses; p = 1.0
    if N > 0:
        k = max(wins, losses); pv = 0.0
        for i in range(k, N + 1):
            pv += math.comb(N, i) * (0.5 ** N)
        p = min(pv * 2.0, 1.0)
    print(f"  infoA beats time on {wins} pairs, loses {losses}, tie {ties}; p={p:.4f}")
    helpfulA = (p < 0.05) and (infoA_acc.mean() > 0.50)
    print(f"  VERDICT A (Binance info bars): {'HELPS (p<0.05 AND >0.50)' if helpfulA else 'no help'}")
    # infoB robustness (pairs with blofin data)
    barr = np.array([x for x in infoB_acc if not np.isnan(x)])
    if len(barr) > 0:
        bdiff = barr - time_acc[:len(barr)]
        bw = int((bdiff > 0).sum()); bl = int((bdiff < 0).sum())
        bp = 1.0; bN = bw + bl
        if bN > 0:
            bk = max(bw, bl); bpv = 0.0
            for i in range(bk, bN + 1):
                bpv += math.comb(bN, i) * (0.5 ** bN)
            bp = min(bpv * 2.0, 1.0)
        print(f"  infoB (blofin vol, n={len(barr)}) mean={barr.mean():.3f}; beats time {bw}/{bl}; p={bp:.4f}")
        print(f"  VERDICT B (blofin robustness): {'ROBUST (p<0.05 AND >0.50)' if (bp<0.05 and barr.mean()>0.50) else 'not robust / insufficient'}")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
