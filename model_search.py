#!/usr/bin/env python3
import os
"""
Falsifiable model/strategy comparison on one fixed walk-forward split.

Compares:
- XGBoost baseline
- XGBoost + Optuna tuning
- LightGBM default
- CatBoost default
- MLP on scaled features
- Class-weighted tuned XGBoost
- Feature-pruned tuned XGBoost

Outputs /models/model_search_report.json for direct before/after evidence.
"""
from pathlib import Path
import json, time, warnings
from datetime import datetime, timezone
from collections import Counter

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from pipeline import (
    fetch_data, add_resampled_features, load_macro_data, add_macro_signals,
    derive_features, detect_regime, walk_forward_splits, ALL_FEATURES,
    USE_MULTI_ASSET, MULTI_ASSET_FILE, triple_barrier_labels,
)
from multi_asset_features import add_multi_asset_features

warnings.filterwarnings('ignore')
MODEL_DIR = Path(__file__).parent / 'models'
THRESHOLD = 0.65


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
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.sort_index()
    df = detect_regime(df)
    return df, multi_cols


def make_model(name, params, n_classes=3):
    if name == 'xgb':
        return xgb.XGBClassifier(**params, n_jobs=6, eval_metric='mlogloss', use_label_encoder=False)
    if name == 'lgb':
        return lgb.LGBMClassifier(**params, n_jobs=os.cpu_count() or 4, verbose=-1)
    if name == 'cat':
        return CatBoostClassifier(**params, loss_function='MultiClass', verbose=False, allow_writing_files=False)
    if name == 'mlp':
        return MLPClassifier(**params, max_iter=400, random_state=42)
    raise ValueError(name)


def gate_eval(preds, probs, y_true, thr=THRESHOLD):
    raw = preds != 2
    gated = raw & (probs.max(axis=1) >= thr)
    return {
        'accuracy': float((preds == y_true).mean()),
        'raw_trades': int(raw.sum()),
        'gated_trades': int(gated.sum()),
        'gated_accuracy': float((preds[gated] == y_true[gated]).mean()) if gated.sum() else None,
        'precision_long': float((preds[gated] == 1).mean()) if ((preds[gated] == 1) | (preds[gated] == 0)).sum() else None,
        'precision_short': float((preds[gated] == 0).mean()) if ((preds[gated] == 1) | (preds[gated] == 0)).sum() else None,
    }


def eval_case(model_fn, X_tr, y_tr, X_val, y_val, X_te, y_te, sample_weight=None, scaler_fn=None, thr=THRESHOLD):
    if scaler_fn:
        X_tr = scaler_fn().fit_transform(X_tr)
        s = scaler_fn()
        X_val = s.transform(X_val)
        X_te = s.transform(X_te)
    m = model_fn()
    fit_kwargs = dict(eval_set=[(X_val, y_val)])
    if sample_weight is not None:
        fit_kwargs['sample_weight'] = sample_weight
    m.fit(X_tr, y_tr, **fit_kwargs)
    preds = m.predict(X_te)
    probs = m.predict_proba(X_te) if hasattr(m, 'predict_proba') else np.eye(3)[preds]
    stats = gate_eval(preds, probs, y_te, thr)
    stats['val_score'] = float(m.score(X_val, y_val))
    return stats


def main():
    t0 = time.time()
    print('[MS] Building frame...')
    df, multi_cols = build_frame()
    df['label'] = triple_barrier_labels(df)['label']
    df = df[df['label'].notna()]

    feature_cols = [f for f in ALL_FEATURES if f in df.columns] + multi_cols
    for col in ['regime_high_vol', 'regime_trending']:
        if col not in df.columns:
            df[col] = 0
            if col not in feature_cols:
                feature_cols.append(col)
    required = [c for c in feature_cols if df[c].notna().any()]
    df = df.dropna(subset=required)

    X = np.nan_to_num(df[feature_cols].values, nan=0.0, posinf=0.0, neginf=0.0)
    y = df['label'].values
    splits = walk_forward_splits(df)
    sp = splits[-1]
    train_mask = np.zeros(len(df), dtype=bool)
    val_mask = np.zeros(len(df), dtype=bool)
    test_mask = np.zeros(len(df), dtype=bool)
    train_mask[sp['train_idx']] = True
    val_mask[sp['val_idx']] = True
    test_mask[sp['test_idx']] = True

    X_tr, y_tr = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    X_te, y_te = X[test_mask], y[test_mask]

    baseline_params = dict(n_estimators=200, max_depth=4, learning_rate=0.05)
    results = []
    def add(name, fn, **kwargs):
        stats = eval_case(fn, X_tr, y_tr, X_val, y_val, X_te, y_te, **kwargs)
        stats['case'] = name
        results.append(stats)
        print(f"  {name}: acc={stats['accuracy']:.3f} val={stats['val_score']:.3f} gated={stats['gated_trades']} gated_acc={stats['gated_accuracy']}")
        return stats

    # XGB baseline
    add('xgb_baseline', lambda: make_model('xgb', baseline_params))

    # XGB Optuna
    print('[MS] Optuna tuning XGBoost...')
    def objective(trial):
        p = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.7, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.7, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 0.5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 5.0),
        }
        m = make_model('xgb', p)
        m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
        return float(m.score(X_val, y_val))

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20, show_progress_bar=False)
    best_xgb = study.best_trial.params
    add('xgb_optuna', lambda: make_model('xgb', best_xgb))

    # LightGBM
    add('lightgbm_default', lambda: make_model('lgb', dict(n_estimators=200, learning_rate=0.05, max_depth=4)))

    # CatBoost
    add('catboost_default', lambda: make_model('cat', dict(iterations=200, depth=4, learning_rate=0.05)))

    # MLP
    add('mlp_default', lambda: make_model('mlp', dict(hidden_layer_sizes=(64, 32), activation='relu', solver='adam')), scaler_fn=StandardScaler)

    # Class-weighted XGBoost
    cw = Counter(y_tr)
    sw = np.array([1.0 / cw[v] for v in y_tr])
    add('xgb_class_weighted', lambda: make_model('xgb', best_xgb), sample_weight=sw)

    # Pruned features XGBoost
    base_model = make_model('xgb', best_xgb)
    base_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    imp = pd.Series(base_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top_cols = imp.head(max(10, len(imp) // 2)).index.tolist()
    prune_idx = [feature_cols.index(c) for c in top_cols if c in feature_cols]
    X_tr_p = X_tr[:, prune_idx]
    X_val_p = X_val[:, prune_idx]
    X_te_p = X_te[:, prune_idx]
    add('xgb_pruned_top50', lambda: make_model('xgb', best_xgb),  X_tr=X_tr_p, X_val=X_val_p, X_te=X_te_p)

    report = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'split': {'train': int(train_mask.sum()), 'val': int(val_mask.sum()), 'test': int(test_mask.sum())},
        'features': len(feature_cols),
        'threshold': THRESHOLD,
        'cases': results,
        'feature_importance': imp.to_dict(),
        'best_xgb_params': best_xgb,
        'elapsed_sec': round(time.time() - t0, 1),
    }
    (MODEL_DIR / 'model_search_report.json').write_text(json.dumps(report, indent=2))
    print('[MS] saved', MODEL_DIR / 'model_search_report.json')
    print('[MS] done in', round(time.time() - t0, 1), 's')
    return report


if __name__ == '__main__':
    main()