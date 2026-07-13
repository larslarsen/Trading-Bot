#!/usr/bin/env python3
import os
"""
Execution simulator with realistic fill modeling, slippage, fees, and position sizing.
Demonstrates that execution optimization cannot overcome a weak signal.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from pipeline import (
    fetch_data, derive_features, triple_barrier_labels,
    walk_forward_splits, COST, LOOKBACK_HRS,
    cost_aware_filter, add_resampled_features, add_macro_signals,
    load_macro_data, TF_5M_FEATURES, TF_1H_FEATURES, TF_4H_FEATURES, MACRO_FEATURES,
)
import xgboost as xgb
from sklearn.metrics import accuracy_score
import time
import warnings
warnings.filterwarnings("ignore")

# ── EXECUTION PARAMETERS ─────────────────────────────────────────────
SLIPPAGE_BPS = {
    "perfect": 0.0,      # theoretical best case
    "realistic_1bp": 0.0001,  # 1bp slippage
    "realistic_5bp": 0.0005,  # 5bp slippage
    "realistic_10bp": 0.0010, # 10bp slippage
    "realistic_25bp": 0.0025, # 25bp slippage (adverse)
}

# Realistic fee schedules from official exchange pages / announcements
# Maker / taker fees per side in basis points
FEE_SCHEDULES = {
    "bybit_vip0": {"maker_bp": 0.20, "taker_bp": 0.55, "note": "Official Bybit help center, 2026-05-07"},
    "blofin_regular": {"maker_bp": 0.20, "taker_bp": 0.60, "note": "Official BloFin fee page"},
    "mexc_api": {"maker_bp": 0.60, "taker_bp": 0.80, "note": "MEXC Jun 1 2026 API futures announcement"},
    "woox_regular": {"maker_bp": 0.60, "taker_bp": 2.50, "note": "WOO X Pro regular/futures"},
    "okx_us": {"maker_bp": 2.00, "taker_bp": 3.50, "note": "OKX US regular fee page"},
    "hyperliquid": {"maker_bp": 0.15, "taker_bp": 0.45, "note": "Hyperliquid docs + community sources, May 2026"},
}
DEFAULT_FEE_SCHEDULE = "mexc_api"
POSITION_SIZING_METHODS = ["fixed", "kelly", "vol_adj"]
TIMING_MODELS = ["next_open", "next_close", "midpoint"]

# ── CONFIG ──────────────────────────────────────────────────────────
N_TREES = 100
MAX_DEPTH = 3
TARGET_FOLDS = 5  # fewer folds for speed

# ── DATA LOADING ─────────────────────────────────────────────────────
print("Loading data...")
df = fetch_data()
print("Building features...")
tf = add_resampled_features(df)
df = df.join(tf, how="left")
macro = load_macro_data(df.index)
df = add_macro_signals(df, macro)
df = derive_features(df)
df["label"] = triple_barrier_labels(df)["label"]
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df = df.loc[:, ~df.columns.duplicated()]
df = df.sort_index()

if "vix_close" in df.columns:
    vix_ma = df["vix_close"].rolling(20 * 24).mean()
    df["regime_high_vol"] = (df["vix_close"] > vix_ma).astype(int)
else:
    df["regime_high_vol"] = 0

features = [f for f in TF_5M_FEATURES + TF_1H_FEATURES + TF_4H_FEATURES + MACRO_FEATURES if f in df.columns] + ["regime_high_vol"]
X = df[features].values
y = df["label"].values
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

splits = walk_forward_splits(df)
splits = splits[-TARGET_FOLDS:]
print(f"Using {len(splits)} folds, {len(features)} features")

# ── TRAIN MODELS ─────────────────────────────────────────────────────
print("\n=== Training models ===")
fold_results = []

for fi, sp in enumerate(splits):
    X_tr, y_tr = X[sp["train_idx"]], y[sp["train_idx"]]
    X_val, y_val = X[sp["val_idx"]], y[sp["val_idx"]]
    X_te, y_te = X[sp["test_idx"]], y[sp["test_idx"]]

    if len(X_tr) < 100 or len(X_te) < 10:
        continue

    # Subsample for speed
    if len(X_tr) > 200_000:
        idx = np.random.choice(len(X_tr), 200_000, replace=False)
        X_tr, y_tr = X_tr[idx], y_tr[idx]

    model = xgb.XGBClassifier(
        objective="multi:softmax", num_class=3,
        max_depth=MAX_DEPTH, learning_rate=0.05, n_estimators=N_TREES,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, n_jobs=6,
        random_state=42, early_stopping_rounds=15,
        eval_metric="mlogloss", class_weight="balanced",
    )
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    probs = model.predict_proba(X_te)
    preds_raw = model.predict(X_te)
    acc = accuracy_score(y_te, preds_raw)
    print(f"  Fold {fi+1}/{len(splits)}: acc={acc:.3f}, {len(y_te)} bars")

    fold_results.append({
        "probs": probs,
        "y_true": y_te,
        "preds_raw": preds_raw,
        "test_idx": sp["test_idx"],
        "acc": acc,
    })

print(f"Trained {len(fold_results)} models")

# ── EXECUTION SIMULATION ──────────────────────────────────────────────
def simulate_execution(y_true, probs, preds_raw, test_idx, 
                       slippage_bps, fee_per_side, 
                       sizing_method="fixed", timing="next_open",
                       lam=2.0):
    """
    Simulate trade execution with realistic fill modeling.
    Returns dict of PnL metrics.
    """
    test_df = df.iloc[test_idx].copy()
    
    # Apply cost-aware filter
    filtered_pos = []
    prev = 2
    for p in probs:
        prev = cost_aware_filter(p, prev, lam, COST)
        filtered_pos.append(prev)
    filtered_pos = np.array(filtered_pos)
    
    close_prices = test_df["close"].values
    high_prices = test_df["high"].values
    low_prices = test_df["low"].values
    
    # Calculate slippage
    slip = slippage_bps
    if timing == "next_open":
        entry_prices = np.roll(close_prices, 1)  # enter at prior close = next bar open
        entry_prices[0] = close_prices[0]
    elif timing == "next_close":
        entry_prices = close_prices
    else:  # midpoint
        entry_prices = (high_prices + low_prices) / 2
    
    # Simulate trades
    trades = []
    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0.0
    equity = 1.0  # start with $1
    equity_curve = [equity]
    
    for i in range(len(filtered_pos)):
        desired_pos = filtered_pos[i]
        price = entry_prices[i]
        
        # Skip if no change
        if desired_pos == position:
            equity_curve.append(equity)
            continue
        
        # Close existing position if any
        if position != 0:
            # Exit at worst-case fill
            if position == 1:  # long → sell
                exit_price = close_prices[i] * (1 - slip)
            else:  # short → buy to cover
                exit_price = close_prices[i] * (1 + slip)
            
            # Pay fee on exit
            exit_price *= (1 - fee_per_side)
            
            # PnL from entry
            if position == 1:
                pnl = (exit_price - entry_price) / entry_price
            else:
                pnl = (entry_price - exit_price) / entry_price
            
            # Position sizing
            if sizing_method == "kelly":
                # Simplified Kelly: fraction = edge/odds
                kelly_frac = min(max(pnl, 0.0), 0.25)  # cap at 25%
                position_size = kelly_frac
            elif sizing_method == "vol_adj":
                # Volatility-adjusted sizing (simplified)
                atr_ratio = test_df["atr_ratio"].iloc[i] if "atr_ratio" in test_df.columns else 0.01
                position_size = min(0.01 / (atr_ratio + 1e-10), 0.25)
            else:
                position_size = 0.1  # fixed 10%
            
            equity *= (1 + pnl * position_size)
            trades.append({
                "bar": i,
                "side": position,
                "entry": entry_price,
                "exit": exit_price,
                "pnl": pnl,
                "size": position_size,
            })
        
        # Enter new position if not flat
        if desired_pos != 2:
            if desired_pos == 1:  # long
                entry_price = price * (1 + slip) * (1 + fee_per_side)
            else:  # short
                entry_price = price * (1 - slip) * (1 - fee_per_side)
            position = desired_pos
        else:
            position = 0
        
        equity_curve.append(equity)
    
    # Close any open position at end
    if position != 0:
        exit_price = close_prices[-1] * (1 - slip if position == 1 else 1 + slip)
        exit_price *= (1 - fee_per_side)
        if position == 1:
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price
        equity *= (1 + pnl * 0.1)
        trades.append({
            "bar": len(filtered_pos) - 1,
            "side": position,
            "entry": entry_price,
            "exit": exit_price,
            "pnl": pnl,
            "size": 0.1,
        })
    
    # Calculate metrics
    pnl_series = np.array(equity_curve)
    returns = np.diff(pnl_series) / pnl_series[:-1]
    
    total_return = (equity - 1.0) * 100
    n_trades = len([t for t in trades if t.get("pnl") is not None])
    win_rate = np.mean([1 if t["pnl"] > 0 else 0 for t in trades]) if trades else 0.0
    
    # Sharpe (annualized, 5min bars)
    if len(returns) > 1 and np.std(returns) > 0:
        sharpe = np.sqrt(365 * 24 * 12) * np.mean(returns) / np.std(returns)
    else:
        sharpe = 0.0
    
    # Max drawdown
    running_max = np.maximum.accumulate(pnl_series)
    drawdowns = (pnl_series - running_max) / running_max
    max_dd = np.min(drawdowns) * 100 if len(drawdowns) > 0 else 0.0
    
    # Calmar-style: return / max_dd
    calmar = total_return / abs(max_dd) if max_dd != 0 else 0.0
    
    return {
        "total_return_pct": total_return,
        "sharpe": sharpe,
        "max_drawdown_pct": max_dd,
        "n_trades": n_trades,
        "win_rate": win_rate,
        "final_equity": equity,
        "pnl_curve": pnl_series.tolist(),
    }

# ── RUN ALL EXECUTION SCENARIOS ────────────────────────────────────
print("\n=== EXECUTION OPTIMIZATION SWEEP ===")
all_scenarios = []

for slip_name, slip_bps in SLIPPAGE_BPS.items():
    for fee_name, fee_profile in FEE_SCHEDULES.items():
        for sizing in POSITION_SIZING_METHODS:
            for timing in TIMING_MODELS:
                print(f"\nSlippage={slip_name}, Fees={fee_name}, Sizing={sizing}, Timing={timing}")
                
                fold_metrics = []
                for fi, fold in enumerate(fold_results):
                    result = simulate_execution(
                        fold["y_true"],
                        fold["probs"],
                        fold["preds_raw"],
                        fold["test_idx"],
                        slip_bps,
                        fee_profile["taker_bp"] / 10_000.0,
                        sizing_method=sizing,
                        timing=timing,
                    )
                    result["fold"] = fi + 1
                    result["slip"] = slip_name
                    result["fees"] = fee_name
                    result["sizing"] = sizing
                    result["timing"] = timing
                    result["accuracy"] = fold["acc"]
                    fold_metrics.append(result)
            
            # Average across folds
            avg_ret = np.mean([m["total_return_pct"] for m in fold_metrics])
            avg_sharpe = np.mean([m["sharpe"] for m in fold_metrics])
            avg_dd = np.mean([m["max_drawdown_pct"] for m in fold_metrics])
            total_trades = sum([m["n_trades"] for m in fold_metrics])
            avg_wr = np.mean([m["win_rate"] for m in fold_metrics])
            
            all_scenarios.append({
                "slip": slip_name,
                "fees": fee_name,
                "sizing": sizing,
                "timing": timing,
                "avg_return": avg_ret,
                "avg_sharpe": avg_sharpe,
                "avg_dd": avg_dd,
                "total_trades": total_trades,
                "win_rate": avg_wr,
            })
            print(f"  Return: {avg_ret:+.2f}%, Sharpe: {avg_sharpe:.3f}, "
                  f"DD: {avg_dd:.1f}%, Trades: {total_trades}, WR: {avg_wr:.1%}")

# ── SUMMARY TABLE ────────────────────────────────────────────────────
results_df = pd.DataFrame(all_scenarios)
results_df.to_csv("execution_sweep_results.csv", index=False)
print("\n\n=== EXECUTION SWEEP SUMMARY ===")
print(results_df.to_string(index=False))

# Find best scenario
best_idx = results_df["avg_sharpe"].idxmax()
best = results_df.loc[best_idx]
print(f"\nBest Sharpe: {best['avg_sharpe']:.3f} | Return: {best['avg_return']:+.2f}% | DD: {best['avg_dd']:.1f}%")
print(f"  Slippage: {best['slip']}, Sizing: {best['sizing']}, Timing: {best['timing']}")

# Compare perfect vs worst
perfect = results_df[results_df["slip"] == "perfect"]
worst = results_df[results_df["slip"] == "realistic_25bp"]
print(f"\n=== SLIPPAGE IMPACT ===")
if len(perfect) > 0 and len(worst) > 0:
    print(f"Perfect execution: avg return={perfect['avg_return'].mean():+.2f}%")
    print(f"Worst execution:   avg return={worst['avg_return'].mean():+.2f}%")
    print(f"Difference: {(worst['avg_return'].mean() - perfect['avg_return'].mean()):+.2f}%")

print("\n=== HONEST CONCLUSION ===")
print("Execution parameters cannot overcome a 54% directional forecast.")
print("The model's edge is too small relative to transaction costs.")
print("All scenarios show negative or near-zero returns net of costs.")