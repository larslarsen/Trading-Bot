#!/usr/bin/env python3
"""
Falsifiable comparison: no regime gate vs equities-adjusted regime gate.

Same model, same features, same split. If regime gating improves gated
accuracy or Sharpe per trade, macro regime filtering is useful.
"""
import json
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
from equities_regime import build_equities_regime

MODEL_DIR = Path(__file__).parent / 'models'


def build_frame(regime_filter=False):
    df = fetch_data()
    df = df.join(add_resampled_features(df), how='left')
    macro = load_macro_data(df.index)
    df = add_macro_signals(df, macro)
    if USE_MULTI_ASSET and Path(MULTI_ASSET_FILE).exists():
        df = add_multi_asset_features(df, MULTI_ASSET_FILE)
    df = derive_features(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.sort_index()
    df = detect_regime(df)
    eq = build_equities_regime(df)
    df = df.join(eq, how='left')
    df['label'] = triple_barrier_labels(df)['label']

    # regime mask: trade only when equities are not in risk-off + VIX-spike
    regime_ok = pd.Series(True, index=df.index)
    if regime_filter:
        risk_off = df.get('eq_risk_off', 0)
        vix_spike = df.get('eq_vix_spike', 0)
        regime_ok = (risk_off == 0) | (vix_spike == 0)
    df = df[regime_ok]

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


def gate_eval(preds, probs, y_true, thr=0.65):
    raw = preds != 2
    conf = probs.max(axis=1)
    gated = raw & (conf >= thr)
    return {
        'accuracy': float((preds == y_true).mean()),
        'raw_trades': int(raw.sum()),
        'gated_trades': int(gated.sum()),
        'gated_accuracy': float((preds[gated] == y_true[gated]).mean()) if gated.sum() else None,
    }


def summarize(records):
    accs = [r['accuracy'] for r in records]
    raw = sum(r['raw_trades'] for r in records)
    gated = sum(r['gated_trades'] for r in records)
    g_accs = [r['gated_accuracy'] for r in records if r['gated_accuracy'] is not None]
    return {
        'folds': len(records),
        'accuracy_mean': round(float(np.mean(accs)), 4),
        'accuracy_std': round(float(np.std(accs)), 4),
        'raw_trades': int(raw),
        'gated_trades': int(gated),
        'gated_accuracy_mean': round(float(np.mean(g_accs)), 4) if g_accs else None,
        'gated_accuracy_std': round(float(np.std(g_accs)), 4) if g_accs else None,
    }


def run_case(df, feature_cols, splits):
    records = []
    for fold_idx, sp in enumerate(splits):
        train_idx = np.array(sp['train_idx'], dtype=int)
        val_idx = np.array(sp['val_idx'], dtype=int)
        test_idx = np.array(sp['test_idx'], dtype=int)
        if len(train_idx) < 200 or len(test_idx) < 20:
            continue
        X = np.nan_to_num(df[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
        y = df['label'].values
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_te, y_te = X[test_idx], y[test_idx]
        model = xgb.XGBClassifier(
            objective='multi:softmax', num_class=3,
            max_depth=4, learning_rate=0.05, n_estimators=200,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0,
            n_jobs=3, random_state=42,
            eval_metric='mlogloss', class_weight='balanced',
        )
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict(X_te)
        probs = model.predict_proba(X_te)
        stats = gate_eval(preds, probs, y_te)
        stats['fold'] = fold_idx
        stats['train_rows'] = int(len(X_tr))
        stats['test_rows'] = int(len(X_te))
        records.append(stats)
    return records


def main():
    t0 = time.time()
    OUT = MODEL_DIR / 'regime_compare_report.json'
    report = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'cases': {},
        'elapsed_sec': 0,
    }
    for name, fn in [
        ('no_regime_gate', lambda: build_frame(False)),
        ('equities_regime_gate', lambda: build_frame(True)),
    ]:
        print(f'[REGIME] Building {name}...')
        df, feature_cols = fn()
        splits = walk_forward_splits(df)
        if not splits:
            report['cases'][name] = {'error': 'no valid splits'}
            continue
        records = run_case(df, feature_cols, splits)
        report['cases'][name] = {
            'feature_count': len(feature_cols),
            'per_fold': records,
            'summary': summarize(records),
        }
    report['elapsed_sec'] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print('[REGIME] saved', OUT)


if __name__ == '__main__':
    import time
    import warnings
    warnings.filterwarnings('ignore')
    main()
