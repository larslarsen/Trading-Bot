#!/usr/bin/env python3
"""
Feature redundancy analysis:
- Spearman correlation matrix
- Variance inflation factor (VIF)
- Recommend dropping Highly correlated or high-VIF columns.
Outputs /models/feature_redundancy_report.json
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from pipeline import (
    fetch_data, add_resampled_features, load_macro_data, add_macro_signals,
    derive_features, detect_regime, ALL_FEATURES,
    USE_MULTI_ASSET, MULTI_ASSET_FILE,
)

MODEL_DIR = Path(__file__).parent / 'models'


def build_frame():
    df = fetch_data()
    df = df.join(add_resampled_features(df), how='left')
    macro = load_macro_data(df.index)
    df = add_macro_signals(df, macro)
    if USE_MULTI_ASSET and Path(MULTI_ASSET_FILE).exists():
        from multi_asset_features import add_multi_asset_features
        df = add_multi_asset_features(df, MULTI_ASSET_FILE)
    df = derive_features(df)
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


def main():
    OUT = MODEL_DIR / 'feature_redundancy_report.json'
    df, feature_cols = build_frame()
    sub = df[feature_cols].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
    sub = sub.loc[:, ~sub.columns.duplicated()]

    corr = sub.corr(method='spearman').abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_corr = []
    for col in upper.columns:
        matches = upper[col][upper[col] > 0.85].sort_values(ascending=False)
        for idx, val in matches.items():
            high_corr.append({'feature_a': col, 'feature_b': idx, 'correlation': round(float(val), 3)})

    # sample VIF on at most 150k rows for speed
    vif_df = sub.sample(n=min(len(sub), 150000), random_state=42).reset_index(drop=True)
    X_const = vif_df.assign(const=1.0)
    vif = {}
    for i, c in enumerate([c for c in vif_df.columns if vif_df[c].var() > 0]):
        try:
            vif[c] = round(float(variance_inflation_factor(X_const.values, i)), 3)
        except Exception:
            vif[c] = None

    drop_candidates = sorted({d['feature_a'] for d in high_corr} | {c for c, v in vif.items() if isinstance(v, float) and v >= 10})

    report = {
        'features': len(feature_cols),
        'high_corr_pairs_count': len(high_corr),
        'high_vif_count': int(sum(1 for v in vif.values() if isinstance(v, float) and v >= 10)),
        'high_corr_pairs': high_corr[:100],
        'vif': vif,
        'drop_candidates': drop_candidates,
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print('[FA] saved', OUT)


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')
    main()
