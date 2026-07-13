#!/usr/bin/env python3
"""Vectorized triple-barrier label analysis."""
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("btc_5m.csv", parse_dates=["ts"])
df.set_index("ts", inplace=True)
df.index = pd.to_datetime(df.index, utc=True)

c = df["close"].values
h = df["high"].values
l = df["low"].values
n = len(df)
HORIZON = 12

print(f"Data: {n} bars from {df.index[0]} to {df.index[-1]}")

# Vectorized triple barrier
def vectorized_labels(tp_pct, sl_pct):
    tp = c * (1 + tp_pct)
    sl = c * (1 - sl_pct)
    labels = np.full(n, 2, dtype=int)
    # For each bar, find earliest horizon where high>=tp or low<=sl
    for j in range(1, HORIZON + 1):
        idx = np.arange(n - j)
        # Only update if label still flat
        mask = labels[idx] == 2
        if not np.any(mask):
            break
        hit_tp = (h[idx + j] >= tp[idx]) & mask
        hit_sl = (l[idx + j] <= sl[idx]) & mask
        # long = 1, short = 0
        labels[idx[hit_tp]] = 1
        labels[idx[hit_sl]] = 0
        # remaining flat stays 2
    return labels

# Grid search
print(f"\n{'TP':>6} {'SL':>6} {'short':>8} {'long':>8} {'flat':>8} {'trade_rt':>10}")
print("-" * 50)
for tp in [0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010]:
    for sl in [0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010]:
        lab = vectorized_labels(tp, sl)
        sc = np.bincount(lab, minlength=3)
        trade_rt = (sc[0]+sc[1])/n*100
        print(f"{tp*100:5.1f}% {sl*100:5.1f}% {sc[1]/n*100:7.1f}% {sc[0]/n*100:7.1f}% {sc[2]/n*100:7.1f}% {trade_rt:9.1f}%")

# Symmetric comparison
print("\n=== Symmetric vs Asymmetric Detail ===")
configs = [
    (0.005, 0.003, "current: TP=0.5% SL=0.3%"),
    (0.004, 0.004, "symmetric: TP=SL=0.4%"),
    (0.005, 0.005, "symmetric: TP=SL=0.5%"),
    (0.003, 0.003, "symmetric: TP=SL=0.3%"),
    (0.002, 0.002, "symmetric: TP=SL=0.2%"),
]
for tp, sl, desc in configs:
    lab = vectorized_labels(tp, sl)
    sc = np.bincount(lab, minlength=3)
    ratio = sc[0]/max(sc[1], 1)
    trade_rt = (sc[0]+sc[1])/n*100
    print(f"\n{desc}")
    print(f"  short={sc[0]} ({sc[0]/n*100:.1f}%), long={sc[1]} ({sc[1]/n*100:.1f}%), flat={sc[2]} ({sc[2]/n*100:.1f}%)")
    print(f"  trade_rate={trade_rt:.1f}%, short/long_ratio={ratio:.2f}")

# 12-bar return distribution
print("\n=== Raw 12-bar return distribution ===")
ret = (c[HORIZON:] - c[:-HORIZON]) / c[:-HORIZON]
print(f"Mean: {ret.mean()*100:.4f}%")
print(f"Std: {ret.std()*100:.4f}%")
print(f"Pct positive: {(ret > 0).mean()*100:.1f}%")
print(f"Pct negative: {(ret < 0).mean()*100:.1f}%")
print(f"Median: {np.median(ret)*100:.4f}%")
print(f"Skew: {pd.Series(ret).skew():.4f}")
