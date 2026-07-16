#!/usr/bin/env python3
"""Binary classifiers: separate long-vs-not and short-vs-not models."""
import warnings
warnings.filterwarnings("ignore")

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from pipeline import (
    fetch_data, add_resampled_features, derive_features,
    load_macro_data, add_macro_signals,
    triple_barrier_labels, ALL_FEATURES, LAMBDA, COST,
    USE_MULTI_ASSET, MULTI_ASSET_FILE,
    detect_regime, walk_forward_splits,
)
from multi_asset_features import add_multi_asset_features

MODEL_DIR = Path(__file__).parent / 'models'


def build_frame():
    df = fetch_data()
    df = df.join(add_resampled_features(df), how='left')
    macro = load_macro_data(df.index)
    df = add_macro_signals(df, macro)
    if USE_MULTI_ASSET and Path(MULTI_ASSET_FILE).exists():
        df, multi_cols = add_multi_asset_features(df, MULTI_ASSET_FILE)
    else:
        multi_cols = []
    df = derive_features(df)
    df['label'] = triple_barrier_labels(df)['label']
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.sort_index()
    df = detect_regime(df)
    feature_cols = [f for f in ALL_FEATURES if f in df.columns] + multi_cols
    for col in ['regime_high_vol', 'regime_trending']:
        if col not in df.columns:
            df[col] = 0
            if col not in feature_cols:
                feature_cols.append(col)
    required = [c for c in feature_cols if df[c].notna().any()]
    if required:
        df = df.dropna(subset=required)
    return df, feature_cols


def main():
    OUT = MODEL_DIR / 'binary_results.json'
    df, feature_cols = build_frame()
    splits = walk_forward_splits(df)
    if not splits:
        raise RuntimeError('no valid walk-forward splits')
    sp = splits[-1]

    X = np.nan_to_num(df[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    y = df['label'].values
    train_idx = np.array(sp['train_idx'], dtype=int)
    val_idx = np.array(sp['val_idx'], dtype=int)
    test_idx = np.array(sp['test_idx'], dtype=int)

    X_tr, y_tr = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_te, y_te = X[test_idx], y[test_idx]

    # LONG vs not-LONG
    y_tr_l = (y_tr == 1).astype(int)
    y_te_l = (y_te == 1).astype(int)

    m_l = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=6, random_state=42,
        eval_metric='logloss',
    )
    m_l.fit(X_tr, y_tr_l, eval_set=[(X_val, (y_val == 1).astype(int))], verbose=False)
    p_l = m_l.predict_proba(X_te)[:, 1]
    pred_l = (p_l >= 0.5).astype(int)

    # SHORT vs not-SHORT
    y_tr_s = (y_tr == 0).astype(int)
    y_te_s = (y_te == 0).astype(int)

    m_s = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        n_jobs=6, random_state=42,
        eval_metric='logloss',
    )
    m_s.fit(X_tr, y_tr_s, eval_set=[(X_val, (y_val == 0).astype(int))], verbose=False)
    p_s = m_s.predict_proba(X_te)[:, 1]
    pred_s = (p_s >= 0.5).astype(int)

    report = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'split': {'train': int(len(train_idx)), 'val': int(len(val_idx)), 'test': int(len(test_idx))},
        'long': {
            'accuracy': float(accuracy_score(y_te_l, pred_l)),
            'f1': float(f1_score(y_te_l, pred_l, zero_division=0)),
            'auc': float(roc_auc_score(y_te_l, p_l)) if len(np.unique(y_te_l)) > 1 else None,
            'predicted_positive': int(pred_l.sum()),
            'actual_positive': int(y_te_l.sum()),
        },
        'short': {
            'accuracy': float(accuracy_score(y_te_s, pred_s)),
            'f1': float(f1_score(y_te_s, pred_s, zero_division=0)),
            'auc': float(roc_auc_score(y_te_s, p_s)) if len(np.unique(y_te_s)) > 1 else None,
            'predicted_positive': int(pred_s.sum()),
            'actual_positive': int(y_te_s.sum()),
        },
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print('[BP] saved', OUT)


if __name__ == '__main__':
    main()
