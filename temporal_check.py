#!/usr/bin/env python3
"""
Temporal degradation check: split VBT test into pre-2022 vs post-2024.
Outputs JSON to /models/temporal_report.json.
"""
import json, os, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline import (
    fetch_data, add_resampled_features, load_macro_data, add_macro_signals,
    derive_features, detect_regime, walk_forward_splits, ALL_FEATURES,
    USE_MULTI_ASSET, MULTI_ASSET_FILE, triple_barrier_labels,
)
from multi_asset_features import add_multi_asset_features

MODEL_DIR = Path(__file__).parent / 'models'


def build_frame():
    df = fetch_data()
    df = df.join(add_resampled_features(df), how='left')
    macro = load_macro_data(df.index)
    df = add_macro_signals(df, macro)
    if USE_MULTI_ASSET and Path(MULTI_ASSET_FILE).exists():
        df = add_multi_asset_features(df, MULTI_ASSET_FILE)
    df = derive_features(df)
    df['label'] = triple_barrier_labels(df)['label']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.sort_index()
    df = detect_regime(df)
    feature_cols = [f for f in ALL_FEATURES if f in df.columns]
    for col in ['regime_high_vol', 'regime_trending']:
        if col not in df.columns:
            df[col] = 0
            if col not in feature_cols:
                feature_cols.append(col)
    required = [c for c in feature_cols if df[c].notna().any()]
    if required:
        df = df.dropna(subset=required)
    return df, feature_cols


def summarize(name, preds, probs, y_true, thr=0.65, label_mask=None):
    raw = preds != 2
    conf = probs.max(axis=1)
    gated = raw & (conf >= thr)
    if label_mask is not None:
        raw = raw & label_mask
        gated = gated & label_mask
    acc = float((preds == y_true).mean())
    trades = int(gated.sum())
    g_acc = float((preds[gated] == y_true[gated]).mean()) if trades else None
    return {
        'name': name,
        'accuracy': acc,
        'raw_trades': int(raw.sum()),
        'gated_trades': trades,
        'gated_accuracy': g_acc,
    }


def main():
    t0 = time.time()
    OUT = MODEL_DIR / 'temporal_report.json'
    print('[TD] Building frame...')
    df, feature_cols = build_frame()
    splits = walk_forward_splits(df)
    if not splits:
        raise RuntimeError('no valid walk-forward splits')
    sp = splits[-1]

    train_mask = np.zeros(len(df), dtype=bool)
    test_mask = np.zeros(len(df), dtype=bool)
    train_mask[sp['train_idx']] = True
    test_mask[sp['test_idx']] = True

    X = np.nan_to_num(df[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    y = df['label'].values
    X_tr, y_tr = X[train_mask], y[train_mask]
    X_te, y_te = X[test_mask], y[test_mask]

    print('[TD] Training on latest split...')
    model = xgb.XGBClassifier(
        objective='multi:softmax', num_class=3,
        max_depth=4, learning_rate=0.05, n_estimators=200,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=6, random_state=42,
        eval_metric='mlogloss', class_weight='balanced',
    )
    model.fit(X_tr, y_tr, eval_set=[(X_tr[:100], y_tr[:100])], verbose=False)
    best = getattr(model, 'best_ntree_limit', None) or model.n_estimators

    preds = model.predict(X_te)
    probs = model.predict_proba(X_te)
    idx = df.index[test_mask]

    pre2022 = idx.year < 2022
    post2024 = idx.year >= 2024
    mid = ~pre2022 & ~post2024

    cases = [
        summarize('all', preds, probs, y_te),
        summarize('pre2022', preds, probs, y_te, label_mask=pre2022),
        summarize('2022_2023', preds, probs, y_te, label_mask=mid),
        summarize('post2024', preds, probs, y_te, label_mask=post2024),
    ]
    lift = lambda a, b: round(a - b, 4) if a is not None and b is not None else None
    report = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'split_test': int(test_mask.sum()),
        'best_trees': int(best),
        'cases': cases,
        'lift_pre2022_vs_post2024': {
            'accuracy': lift(cases[1]['accuracy'], cases[3]['accuracy']),
            'gated_accuracy': lift(cases[1]['gated_accuracy'], cases[3]['gated_accuracy']),
        },
        'elapsed_sec': round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print('[TD] saved', OUT)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    main()
