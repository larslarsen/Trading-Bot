#!/usr/bin/env python3
"""Regenerate feature-redundancy prune on the FULL feature frame the trainer
actually builds (base 99 + cross-asset), on TODAY's data (multi_5m now fixed).
Outputs models/feature_redundancy_full.json with an empirical BASE_DROP set.
Not a model; just produces the canonical drop list."""
import json, warnings, re
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
import model_trainer as mt

OUT = mt.MODEL_DIR / "feature_redundancy_full.json"

# 1) build BTC feature frame exactly as trainer does (base + cross + extras)
df, feats = mt.build_symbol_features("BTC")
print(f"built BTC frame: rows={len(df)} features={len(feats)}")
sub = df[feats].replace([np.inf, -np.inf], np.nan).dropna()

# 2) Spearman > 0.85
corr = sub.corr(method="spearman").abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_corr = []
for col in upper.columns:
    s = upper[col].dropna()
    if len(s):
        s = s.sort_values(ascending=False)
        for idx, val in s.items():
            high_corr.append({"a": col, "b": idx, "r": round(float(val), 3)})

# 3) VIF >= 10 (sample for speed). Dedupe columns first (a stray dup makes a
# column select return a DataFrame and breaks .var()).
vif_df = sub.loc[:, ~sub.columns.duplicated()].copy()
vif_df = vif_df.sample(n=min(len(vif_df), 150000), random_state=42).reset_index(drop=True)
Xc = vif_df.assign(const=1.0)
vif = {}
num_cols = [c for c in vif_df.columns if pd.api.types.is_numeric_dtype(vif_df[c]) and vif_df[c].var() > 0]
for i, c in enumerate(num_cols):
    try:
        vif[c] = round(float(variance_inflation_factor(Xc[num_cols].values, i)), 2)
    except Exception:
        vif[c] = None

# 4) drop = highly collinear OR high VIF, but NEVER cross-asset / regime / label
cross_names = {c for c in feats if re.search(r"(USDT_|_ret_rel|_corr|_ratio|_vol_z)", c)}
protected = cross_names | {"regime_high_vol", "regime_trending", "label"}
drop = sorted({d["a"] for d in high_corr} | {c for c, v in vif.items() if isinstance(v, (int, float)) and v >= 10}
              - protected)
# keep temporal/basic returns/vol that are informative even if correlated at 0.85?
# Literature: drop only when near-perfect (>=0.95) or VIF>=10. Tighten corr to 0.95.
high_corr_strict = [d for d in high_corr if d["r"] >= 0.95]
drop_strict = sorted({d["a"] for d in high_corr_strict}
                     | {c for c, v in vif.items() if isinstance(v, (int, float)) and v >= 10}
                     - protected)

report = {
    "built_rows": int(len(df)),
    "total_features": len(feats),
    "high_corr_pairs_085": len(high_corr),
    "high_corr_pairs_095": len(high_corr_strict),
    "high_vif_count_ge10": int(sum(1 for v in vif.values() if isinstance(v, (int, float)) and v >= 10)),
    "BASE_DROP_085": drop,
    "BASE_DROP_095": drop_strict,
    "cross_asset_features_detected": sorted(cross_names),
    "vif": vif,
    "high_corr_095_pairs": high_corr_strict[:60],
}
OUT.write_text(json.dumps(report, indent=2))
print(f"WROTE {OUT}")
print(f"  BASE_DROP (corr>=0.95 or VIF>=10): {len(drop_strict)} -> {drop_strict}")
print(f"  (looser corr>=0.85 would drop {len(drop)} -> {drop})")
