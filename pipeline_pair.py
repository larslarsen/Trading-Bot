#!/usr/bin/env python3
"""
Per-PAIR ML trainer — generalizes pipeline.py to ANY symbol.

Trains ONE model for a single target pair using that pair's 5m bars + cross-asset
features (every OTHER coin's 5m returns/vol, so the model can learn "does X lead
Y"), plus leading on-chain-style signals (BTC funding rate, BTC taker order
flow), macro, and equities regime. Reuses pipeline.py's PROVEN engine
(derive_features, triple_barrier_labels, walk_forward_splits, _train_fold,
cost_aware_filter) so the methodology is identical — only the data feed is
generalized.

This is the PER-PAIR model (one model per symbol). The MULTI-PAIR pooled model
(see pipeline_multi.py) is a separate architecture: one model, all symbols, with
a symbol embedding. They answer different questions; build + try both.

Usage:
    python pipeline_pair.py                       # default: BTCUSDT on mexc
    python pipeline_pair.py --symbol SOLUSDT --src mexc
    python pipeline_pair.py --symbol AAVE --src dex --file data/AAVE_5m_dex_max.csv
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

import pipeline as P
import config as _cfg

ROOT = Path(__file__).parent
DATA = ROOT / "data"


def load_target(symbol, src, override=None):
    if override:
        p = Path(override)
    else:
        p = DATA / f"{symbol.upper()}_5m_{src}_max.csv"
    if not p.exists():
        raise SystemExit(f"Target file missing: {p}\nRun the collector first "
                         f"(collector_daemon.py for CEX, dex_forward_collector.py for DEX).")
    df = pd.read_csv(p, parse_dates=["ts"])
    df.set_index("ts", inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)
    print(f"Target {symbol}@{src}: {len(df)} bars {df.index[0]} -> {df.index[-1]}")
    return df


def cross_asset_features(target_idx, exclude_stem, min_bars=100):
    """Per-coin return/vol features from all OTHER data/*_5m_*.csv files,
    aligned (ffill) to the target index. Captures cross-coin lead/lag.
    Skips files too thin to yield a signal (avoids all-NaN junk columns)."""
    feats = {}
    files = sorted(glob.glob(str(DATA / "*_5m_*.csv")))
    for f in files:
        stem = Path(f).stem.split("_5m_")[0]
        if stem.upper() == exclude_stem.upper():
            continue
        try:
            d = pd.read_csv(f, parse_dates=["ts"]).set_index("ts")
            d.index = pd.to_datetime(d.index, utc=True)
            if "close" not in d or len(d) < min_bars:
                continue
            r = d["close"].astype(float).reindex(target_idx).ffill().pct_change()
            r1h = d["close"].astype(float).reindex(target_idx).ffill().pct_change(12)
            v = d["volume"].astype(float).reindex(target_idx).ffill().rolling(12).std()
            feats[f"{stem}_ret"] = r
            feats[f"{stem}_ret1h"] = r1h
            feats[f"{stem}_vol"] = v
        except Exception as e:
            print(f"  cross-asset skip {stem}: {e}")
    if feats:
        ca = pd.DataFrame(feats, index=target_idx)
        ca = ca.add_prefix("ca_")
        print(f"  cross-asset features: {ca.shape[1]} ({len(feats)//3} coins)")
        return ca
    return pd.DataFrame(index=target_idx)


def leading_signals(target_idx):
    """BTC funding rate + taker order flow as leading indicators (aligned)."""
    out = {}
    fh = ROOT / "funding_history.csv"
    if fh.exists():
        m = pd.read_csv(fh, parse_dates=["ts"]).set_index("ts")
        m.index = pd.to_datetime(m.index, utc=True)
        out["btc_funding"] = m["funding_rate"].astype(float).reindex(target_idx).ffill()
    ta = ROOT / "trade_agg_5m.csv"
    if ta.exists():
        m = pd.read_csv(ta, parse_dates=["ts"]).set_index("ts")
        m.index = pd.to_datetime(m.index, utc=True)
        flow = (m["taker_buy_vol"].astype(float) - m["taker_sell_vol"].astype(float))
        out["btc_taker_flow"] = flow.reindex(target_idx).ffill()
    if out:
        df = pd.DataFrame(out, index=target_idx)
        print(f"  leading signals: {df.shape[1]}")
        return df
    return pd.DataFrame(index=target_idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--src", default="mexc", help="cex exchange or 'dex'")
    ap.add_argument("--file", default=None, help="explicit target CSV override")
    args = ap.parse_args()

    # 1. Target
    df = load_target(args.symbol, args.src, args.file)
    tf_resampled = P.add_resampled_features(df)
    df = df.join(tf_resampled, how="left")
    df = df.join(P.build_equities_regime(df), how="left")
    df = P.derive_features(df)

    # 2. Cross-asset + leading signals
    df = df.join(cross_asset_features(df.index, args.symbol))
    df = df.join(leading_signals(df.index))

    # 3. Macro
    macro = P.load_macro_data(df.index)
    df = P.add_macro_signals(df, macro)

    # 4. Clean + features
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.loc[:, ~df.columns.duplicated()].sort_index()
    features = [f for f in P.ALL_FEATURES if f in df.columns]
    print(f"  Active features ({len(features)}): {features}")

    X = df[features].values
    y = df["label"].values if "label" in df.columns else None
    if y is None:
        df = P.triple_barrier_labels(df)
    counts = df["label"].value_counts().sort_index()
    print(f"Label dist: short={counts.get(0,0)} long={counts.get(1,0)} flat={counts.get(2,0)}")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    X = df[features].values
    y = df["label"].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 5. Parallel walk-forward (reuse pipeline's verified worker)
    splits = P.walk_forward_splits(df)
    print(f"Walk-forward folds: {len(splits)} (symbol={args.symbol}@{args.src})")
    from multiprocessing import Pool
    tasks = [(fi, sp, X, y) for fi, sp in enumerate(splits)]
    fold_metrics = []
    with Pool(_cfg.N_WORKERS_CPU) as pool:
        for fi, m in pool.map(P._train_fold, tasks):
            if m is None:
                print(f"  Fold {fi+1}: skipped")
                continue
            fold_metrics.append(m)
            print(f"  Fold {m['fold']}: n_test={m['n_test']} acc={m['accuracy']:.3f} "
                  f"trades={m['n_trades']}")

    metrics_df = pd.DataFrame(fold_metrics)
    if metrics_df.empty:
        print("No valid folds."); return
    print("\n=== Walk-Forward Results ===")
    print(metrics_df.to_string(index=False))
    print("\n=== Summary (mean across folds) ===")
    print(metrics_df[["accuracy", "f1_macro", "pct_trades", "mean_margin", "required"]].mean().to_string())
    print(f"\nTotal trades: {metrics_df['n_trades'].sum():,}")
    print("Done.")


if __name__ == "__main__":
    main()
