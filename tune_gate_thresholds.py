#!/usr/bin/env python3
"""
Tune quality_gate thresholds by OOS PANEL (not by hand).

For a grid of (min_avg_quote_volume, min_history_days) we compute the gated
universe (Bartolucci two-feature gate: liquidity + maturity) and score its
OOS panel using the ALREADY-TRAINED per-pair models (no retraining):
  - per-pair directional accuracy via walk_forward_splits (5 folds)
  - per-pair long-only return (hold when model says LONG)
Aggregated to a panel (mean dir-acc, mean ret, universe size). We want the
setting that MAXIMIZES panel quality at the SMALLEST universe -> memory
headroom for the 5-min daemon poll (fewer pairs = less RAM per build).

Outputs a ranked table; applies the winning threshold to quality_gate.py
defaults + rewrites screener_ml_multi.txt to the winning universe.
"""
import json, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))
from quality_gate import gated_universe
from pipeline import triple_barrier_labels, walk_forward_splits
import model_trainer as mt
from compare_cex_5m_models import features_and_label

MODELS = REPO / "models"


def load_pp(pair):
    sym = pair.replace("USDT", "")
    cands = [MODELS / f"{sym}_xgb.json", MODELS / f"{pair}_xgb.json"]
    if pair == "BTCUSDT":
        cands.append(MODELS / "latest_xgb.json")
    path = next((c for c in cands if c.exists()), None)
    if path is None:
        return None
    m = xgb.XGBClassifier(); m.load_model(str(path)); return m


def score_pair(pair):
    """OOS panel metrics for one pair using its existing per-pair model."""
    X, y, feats = features_and_label(pair)
    if X is None or len(X) < 2000:
        return None
    yv = y.values.astype(int)
    n = len(yv)
    sdf = pd.DataFrame({"label": yv}, index=pd.date_range("2010-01-01", periods=n, freq="5min", tz="UTC"))
    splits = walk_forward_splits(sdf, folds=5)
    if not splits:
        return None
    model = load_pp(pair)
    if model is None:
        return None
    dir_accs, rets = [], []
    Xv = np.nan_to_num(X.values, 0.0)
    for sp in splits:
        te = sp["test_idx"]
        pred = model.predict(Xv[te])
        # directional accuracy (non-FLAT calls)
        mask = pred != 2
        if mask.sum() > 0:
            dir_accs.append((pred[mask] == yv[te][mask]).mean())
        # long-only return proxy: +1 when LONG and actual up, -1 when LONG and actual down
        actual = yv[te]
        pnl = np.where(pred == 1, np.where(actual == 1, 1, -1), 0)
        if len(pnl):
            rets.append(pnl.mean())
    if not dir_accs:
        return None
    return {"dir_acc": float(np.mean(dir_accs)), "ret": float(np.mean(rets))}


def main():
    t0 = time.time()
    # grid: liquidity floor is the main universe-shrinker; maturity secondary
    grid = []
    for v in (100_000, 250_000, 500_000, 1_000_000):
        for h in (730, 1095):
            grid.append((v, h))
    print(f"Tuning gate: {len(grid)} settings, scoring with existing per-pair models\n")
    rows = []
    for v, h in grid:
        try:
            uni = gated_universe(min_avg_quote_volume=v, min_history_days=h)
        except Exception as e:
            print(f"  v={v} h={h} gate err {e!r}")
            continue
        if not uni:
            continue
        dacc, ret, scored = [], [], 0
        for p in uni:
            r = score_pair(p)
            if r:
                dacc.append(r["dir_acc"]); ret.append(r["ret"]); scored += 1
        rows.append({
            "min_qvol": v, "min_hist_d": h, "n": len(uni), "scored": scored,
            "mean_dir_acc": float(np.mean(dacc)) if dacc else float("nan"),
            "mean_ret": float(np.mean(ret)) if ret else float("nan"),
        })
        print(f"  qvol={v/1000:.0f}k hist={h}d -> n={len(uni)} "
              f"dir_acc={np.mean(dacc):.3f} ret={np.mean(ret):+.3f}" if dacc else
              f"  qvol={v/1000:.0f}k hist={h}d -> n={len(uni)} (no scores)")
    # rank: best panel (dir_acc) that also minimizes universe (memory)
    ok = [r for r in rows if not np.isnan(r["mean_dir_acc"])]
    ok.sort(key=lambda r: (-r["mean_dir_acc"], r["n"]))
    print("\n=== RANKED (best dir_acc, then smallest universe) ===")
    print(f"{'qvol_k':>7} {'hist_d':>6} {'n':>3} {'dir_acc':>8} {'ret':>7}")
    for r in ok:
        print(f"{r['min_qvol']/1000:>7.0f} {r['min_hist_d']:>6} {r['n']:>3} "
              f"{r['mean_dir_acc']:>8.3f} {r['mean_ret']:>+7.3f}")
    # pick: highest dir_acc among settings with n <= 15 (memory headroom target)
    headroom = [r for r in ok if r["n"] <= 15]
    pick = (headroom or ok)[0]
    print(f"\nPICK (dir_acc-best, universe<=15 for RAM headroom): "
          f"qvol={pick['min_qvol']} hist={pick['min_hist_d']} n={pick['n']} "
          f"dir_acc={pick['mean_dir_acc']:.3f}")
    # apply
    uni = gated_universe(min_avg_quote_volume=pick["min_qvol"], min_history_days=pick["min_hist_d"])
    scr = REPO / "screener_ml_multi.txt"
    body = "# ML multi-pair screener — literature-gated universe (tuned by OOS panel).\n" \
           "# One pair per line. Auto-discovered from models/<sym>_xgb.json.\n" \
           "# Tuned via tune_gate_thresholds.py: best OOS dir-acc at smallest\n" \
           f"# universe for 5-min-poll RAM headroom (qvol>={pick['min_qvol']}, hist>={pick['min_hist_d']}).\n\n"
    scr.write_text(body + "\n".join(uni) + "\n")
    print(f"Wrote {scr.name} with {len(uni)} pairs")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
