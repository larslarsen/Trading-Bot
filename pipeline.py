#!/usr/bin/env python3
"""
BTC/USDT 5-min ML trading pipeline v2
- Data: local CSV backfill + live exchange feeds
- Features: 5m technicals + 1h/4h BTC momentum + macro proxies (SPY/GLD/TLT/UUP/VIX)
- Labels: triple-barrier (symmetric TP/SL for label balance)
- Model: XGBoost multiclass (long/short/flat)
- Validation: 27-fold expanding walk-forward
- Execution: cost-aware Bysik filter
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score
from platform import system
from multi_asset_features import add_multi_asset_features
from equities_regime import build_equities_regime
import warnings
warnings.filterwarnings("ignore")
import config as _cfg  # N_JOBS = physical cores - 1 (leave headroom)

# ── CONFIG ─────────────────────────────────────────────────────────────
SYMBOL       = "BTC/USDT"
TIMEFRAME    = "5m"
DATA_FILE    = "btc_5m.csv"
MULTI_ASSET_FILE = "multi_5m.csv"
USE_MULTI_ASSET = True
LOOKBACK_HRS = 24
HORIZON_BARS = 12
TP_PCT       = 0.004       # symmetric 0.4%
SL_PCT       = 0.004
COST         = 0.0005      # 5bp per side
LAMBDA       = 4.0
FOLDS        = 27
TRAIN_MO     = 12
VAL_MO       = 3
TEST_MO      = 3
STEP_MO      = 6  # fewer folds = fewer models to train

# ── INTERNAL CONFIG ────────────────────────────────────────────────────
USE_CUSUM    = False     # toggle for event sampling
N_TREES      = 150       # fewer trees = faster training
MAX_DEPTH    = 4
LR           = 0.05
USE_ATR_BARRIERS = False  # ATR barriers backfired in backtest (0.452 acc)
ATR_MULT     = 1.5       # TP/SL = ATR * multiplier
USE_VOL_SCALING = True   # Oprea/Bâra vol-aware normalization
# ─────────────────────────────────────────────────────────────────────
# ────────────────────────────────────────────────────────────────────────

# ── LISTS ──────────────────────────────────────────────────────────────
TF_5M_FEATURES = [
    # Temporal (Silva et al. full set)
    "year", "month", "day_of_year", "weekday", "hour", "minute",
    "hour_sin", "hour_cos", "month_sin", "month_cos",
    "weekday_sin", "weekday_cos",
    "is_weekend", "is_month_start", "is_month_end",
    "quarter", "is_quarter_start", "is_quarter_end",
    # Returns
    "ret_1", "ret_3", "ret_6", "ret_12", "ret_24",
    # Volatility
    "volatility_12", "volatility_24",
    # Volume
    "vol_ratio", "volume_change",
    # SMAs
    "sma_dist_6", "sma_dist_12", "sma_cross_signal",
    # MACD
    "ema_ratio", "macd_hist", "macd_hist_smooth",
    # RSI / BB / ATR
    "rsi_14", "bb_pos", "bb_width", "atr_ratio",
]
TF_1H_FEATURES = [
    "ret_1h", "vol_1h", "mom_1h_3h", "mom_1h_6h", "range_1h_pct",
]
TF_4H_FEATURES = [
    "ret_4h", "vol_4h", "mom_4h_12h", "range_4h_pct",
]
MACRO_FEATURES = [
    "spy_close_ma20_signal", "gld_close_ma20_signal",
    "tlt_close_ma20_signal", "uup_close_ma20_signal", "vix_close_ma20_signal",
    "btc_spy_ratio_chg", "btc_gld_ratio_chg", "btc_uup_ratio_chg",
]
MICRO_FEATURES = [
    "funding_rate",
    "basis_pct",
    "open_interest",
    "oi_chg_12",
    "insurance_balance",
    "taker_buy_vol",
    "taker_sell_vol",
    "taker_buy_sell_ratio",
    "trade_count",
    "imbalance",
    "spread",
]
MULTI_ASSET_FEATURES = [
    "ETHUSDT_returns",
    "ETHUSDT_btc_ratio",
    "ETHUSDT_btc_ratio_chg",
    "ETHUSDT_volume_z",
    "ETHUSDT_btc_ret_rel_6",
    "ETHUSDT_btc_ret_rel_12",
    "ETHUSDT_btc_ret_rel_24",
    "ETHUSDT_btc_ret_rel_48",
    "btc_corr_ETHUSDT_6",
    "btc_corr_ETHUSDT_12",
    "btc_corr_ETHUSDT_24",
    "btc_corr_ETHUSDT_48",
    "SOLUSDT_returns",
    "SOLUSDT_btc_ratio",
    "SOLUSDT_btc_ratio_chg",
    "SOLUSDT_volume_z",
    "SOLUSDT_btc_ret_rel_6",
    "SOLUSDT_btc_ret_rel_12",
    "SOLUSDT_btc_ret_rel_24",
    "SOLUSDT_btc_ret_rel_48",
    "btc_corr_SOLUSDT_6",
    "btc_corr_SOLUSDT_12",
    "btc_corr_SOLUSDT_24",
    "btc_corr_SOLUSDT_48",
]
EQUITIES_FEATURES = [
    'eq_spy_signal', 'eq_gold_signal',
    'eq_tlt_signal', 'eq_uup_signal', 'eq_vix_signal',
    'eq_vix_spike', 'eq_risk_on', 'eq_risk_off',
    'eq_rates_falling', 'eq_credit_stress',
]
ALL_FEATURES = (
    TF_5M_FEATURES + TF_1H_FEATURES + TF_4H_FEATURES +
    MACRO_FEATURES + MICRO_FEATURES + MULTI_ASSET_FEATURES + EQUITIES_FEATURES
)
# ────────────────────────────────────────────────────────────────────────

# ── DATA LOADING ────────────────────────────────────────────────────────

def fetch_data():
    print(f"Loading {DATA_FILE}...")
    df = pd.read_csv(DATA_FILE, parse_dates=["ts"])
    df.set_index("ts", inplace=True)
    df.index = pd.to_datetime(df.index, utc=True)
    print(f"  Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    return df

# ── MACRO DATA ─────────────────────────────────────────────────────────

def load_macro_data(df_index):
    """Load macro proxies from local CSV files and forward-fill to df index."""
    macro_files = {
        "spy_close": "spy_daily.csv",
        "gld_close": "gld_daily.csv",
        "tlt_close": "tlt_daily.csv",
        "uup_close": "uup_daily.csv",
        "vix_close": "vix_daily.csv",
    }
    series_dict = {}
    macro_dir = Path(__file__).parent
    for col, fname in macro_files.items():
        path = macro_dir / fname
        if not path.exists():
            print(f"[pipeline] macro file missing, skipping: {fname}")
            continue
        try:
            m = pd.read_csv(path)
            m["ts"] = pd.to_datetime(m["ts"], utc=True)
            m = m[["ts", "close"]].rename(columns={"close": col})
            m[col] = pd.to_numeric(m[col], errors="coerce")
            m = m.dropna(subset=[col])
            if m[col].notna().sum() == 0:
                print(f"  Warning: {fname} loaded but all NaN after coercion")
                continue
            m = m.set_index("ts")[col]
            series_dict[col] = m
        except Exception as e:
            print(f"  Warning: {fname} load error: {e}")

    if not series_dict:
        return pd.DataFrame(index=df_index)

    macro = pd.DataFrame(series_dict, index=df_index)
    return macro.ffill().bfill()

# ── FEATURES ────────────────────────────────────────────────────────────

def add_resampled_features(df):
    """Add 1h and 4h BTC resampled features, forward-filled to 5m index."""
    idx_name = df.index.name or 'timestamp'
    df_reset = df.reset_index()
    ohlc_1h = df_reset.resample("1h", on=idx_name).agg({
        idx_name: "last",
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()
    ohlc_1h["ret_1h"] = ohlc_1h["close"].pct_change()
    ohlc_1h["vol_1h"] = ohlc_1h["close"].rolling(24).std()
    ohlc_1h["mom_1h_3h"] = ohlc_1h["close"].pct_change(3)
    ohlc_1h["mom_1h_6h"] = ohlc_1h["close"].pct_change(6)
    ohlc_1h["range_1h_pct"] = (ohlc_1h["high"] - ohlc_1h["low"]) / (ohlc_1h["close"] + 1e-10)

    ohlc_4h = df_reset.resample("4h", on=idx_name).agg({
        idx_name: "last",
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()
    ohlc_4h["ret_4h"] = ohlc_4h["close"].pct_change()
    ohlc_4h["vol_4h"] = ohlc_4h["close"].rolling(6).std()
    ohlc_4h["mom_4h_12h"] = ohlc_4h["close"].pct_change(3)
    ohlc_4h["range_4h_pct"] = (ohlc_4h["high"] - ohlc_4h["low"]) / (ohlc_4h["close"] + 1e-10)

    ohlc_1h = ohlc_1h.set_index(idx_name)
    ohlc_4h = ohlc_4h.set_index(idx_name)
    resampled = ohlc_1h.reindex(df.index, method="ffill")[["ret_1h", "vol_1h", "mom_1h_3h", "mom_1h_6h", "range_1h_pct"]]
    resampled = resampled.join(ohlc_4h.reindex(df.index, method="ffill")[["ret_4h", "vol_4h", "mom_4h_12h", "range_4h_pct"]])
    return resampled


def derive_features(df):
    """Derive 5m technical indicators on the original BTC data."""
    c = df["close"].values if isinstance(df["close"], pd.Series) else df["close"]
    h = df["high"].values if isinstance(df["high"], pd.Series) else df["high"]
    l = df["low"].values if isinstance(df["low"], pd.Series) else df["low"]
    v = df["volume"].values if isinstance(df["volume"], pd.Series) else df["volume"]
    n = len(c)

    # ── TEMPORAL FEATURES (Silva et al. style) ──────────────────────
    ts = df.index
    df["year"] = ts.year
    df["month"] = ts.month
    df["day_of_year"] = ts.dayofyear
    df["weekday"] = ts.weekday
    df["hour"] = ts.hour
    df["minute"] = ts.minute

    # Cyclic transformations
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7)

    # Binary temporal flags
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_month_start"] = (ts.day <= 3).astype(int)
    df["is_month_end"] = (ts.day >= 28).astype(int)
    df["quarter"] = ts.quarter
    df["is_quarter_start"] = ((df["month"] % 3 == 1) & (ts.day <= 5)).astype(int)
    df["is_quarter_end"] = ((df["month"] % 3 == 0) & (ts.day >= 25)).astype(int)

    # ── RETURNS ────────────────────────────────────────────────────
    df["ret_1"] = np.concatenate([[np.nan], np.diff(np.log(c))])
    df["ret_3"] = pd.Series(df["ret_1"].values, index=df.index).rolling(3).sum().values
    df["ret_6"] = pd.Series(df["ret_1"].values, index=df.index).rolling(6).sum().values
    df["ret_12"] = pd.Series(df["ret_1"].values, index=df.index).rolling(12).sum().values
    df["ret_24"] = pd.Series(df["ret_1"].values, index=df.index).rolling(24).sum().values

    # ── VOLATILITY ─────────────────────────────────────────────────
    df["volatility_12"] = pd.Series(df["ret_1"].values, index=df.index).rolling(12).std().values
    df["volatility_24"] = pd.Series(df["ret_1"].values, index=df.index).rolling(24).std().values

    # ── VOLUME ─────────────────────────────────────────────────────
    vma6 = pd.Series(v, index=df.index).rolling(6).mean().values
    vma24 = pd.Series(v, index=df.index).rolling(24).mean().values
    df["vol_ratio"] = vma6 / (vma24 + 1e-10)
    df["volume_change"] = pd.Series(v, index=df.index).pct_change().values

    # ── SMAs ───────────────────────────────────────────────────────
    sma6 = pd.Series(c, index=df.index).rolling(6).mean().values
    sma12 = pd.Series(c, index=df.index).rolling(12).mean().values
    df["sma_dist_6"] = (c - sma6) / (sma6 + 1e-10)
    df["sma_dist_12"] = (c - sma12) / (sma12 + 1e-10)
    df["sma_cross_signal"] = (sma6 - sma12) / (sma12 + 1e-10)

    # ── EMA / MACD ─────────────────────────────────────────────────
    ema12 = pd.Series(c, index=df.index).ewm(span=12, adjust=False).mean()
    ema26 = pd.Series(c, index=df.index).ewm(span=26, adjust=False).mean()
    df["ema_ratio"] = ema12.values / (ema26.values + 1e-10)
    df["macd_hist"] = ema12.values - ema26.values
    df["macd_hist_smooth"] = (
        pd.Series(ema12.values - ema26.values, index=df.index)
        .ewm(span=9, adjust=False).mean().values
    )

    # ── RSI ────────────────────────────────────────────────────────
    delta = pd.Series(c, index=df.index).diff()
    gain = delta.clip(lower=0).rolling(14).mean().values
    loss = (-delta.clip(upper=0)).rolling(14).mean().values
    rs = gain / (loss + 1e-10)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    # ── BOLLINGER ──────────────────────────────────────────────────
    bb_mid = pd.Series(c, index=df.index).rolling(20).mean().values
    bb_std = pd.Series(c, index=df.index).rolling(20).std().values
    df["bb_mid"] = bb_mid
    df["bb_std"] = bb_std
    df["bb_width"] = (4 * bb_std) / (bb_mid + 1e-10)
    df["bb_pos"] = (c - (bb_mid - 2 * bb_std)) / (4 * bb_std + 1e-10)

    # ── ATR ────────────────────────────────────────────────────────
    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))
    atr = pd.Series(tr, index=df.index).rolling(14).mean().values
    df["atr_ratio"] = atr / (c + 1e-10)

    # ── VOL-AWARE NORMALIZATION (Oprea/Bâra) ───────────────────────
    if USE_VOL_SCALING:
        roll_vol = pd.Series(df["ret_1"].values, index=df.index).rolling(24).std().values
        for feat in ["macd_hist", "macd_hist_smooth", "sma_dist_6", "sma_dist_12",
                     "sma_cross_signal", "ema_ratio", "bb_pos", "bb_width"]:
            if feat in df.columns:
                denom = roll_vol + 1e-10
                df[feat] = df[feat].values / denom

    return df


def detect_regime(df):
    """Detect market regime: low-vol / high-vol + trending / range-bound."""
    if "volatility_24" not in df.columns:
        df["regime"] = "unknown"
        return df

    # Realized vol regime
    vol_ma = df["volatility_24"].rolling(24 * 6).mean()  # 6-day median
    vol_med = vol_ma.rolling(24 * 30, min_periods=1).median()
    high_vol = (df["volatility_24"] > vol_med).astype(int)

    # Trend regime: ADX-like using close vs moving average
    if "bb_mid" not in df.columns:
        df["regime"] = "low_vol"  # fallback
        return df
    sma24 = pd.Series(df["close"].values, index=df.index).rolling(24).mean()
    trend_strength = (df["close"].values - sma24.values) / (df["bb_std"].values + 1e-10)
    trending = (abs(trend_strength) > 1.0).astype(int)

    # Combine
    df["regime_high_vol"] = high_vol
    df["regime_trending"] = trending
    df["regime"] = np.where(
        high_vol == 1,
        np.where(trending == 1, "high_vol_trend", "high_vol_range"),
        np.where(trending == 1, "low_vol_trend", "low_vol_range"),
    )
    return df
def add_macro_signals(df, macro=None):
    """Join macro data if not already present, then derive relative signals."""
    if macro is not None:
        cols_to_add = [c for c in macro.columns if c not in df.columns]
        if cols_to_add:
            df = df.join(macro[cols_to_add], how="left")
    # Simple macro trend signal vs 20-day MA
    for col in ["spy_close", "gld_close", "tlt_close", "uup_close", "vix_close"]:
        if col in df.columns:
            ma = df[col].rolling(20).mean().values
            df[f"{col}_ma20_signal"] = np.where(
                ma > 0, (df[col].values - ma) / (ma + 1e-10), np.nan
            )
    # BTC / macro ratios
    if "spy_close" in df.columns:
        ratio = df["close"].values / (df["spy_close"].values + 1e-10)
        df["btc_spy_ratio"] = ratio
        df["btc_spy_ratio_chg"] = pd.Series(ratio, index=df.index).pct_change(12).values
    if "gld_close" in df.columns:
        ratio = df["close"].values / (df["gld_close"].values + 1e-10)
        df["btc_gld_ratio"] = ratio
        df["btc_gld_ratio_chg"] = pd.Series(ratio, index=df.index).pct_change(12).values
    if "uup_close" in df.columns:
        ratio = df["close"].values / (df["uup_close"].values + 1e-10)
        df["btc_uup_ratio"] = ratio
        df["btc_uup_ratio_chg"] = pd.Series(ratio, index=df.index).pct_change(12).values
    return df



# ── TRIPLE-BARRIER LABELS ──────────────────────────────────────────────

def triple_barrier_labels(df, horizon=HORIZON_BARS):
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    n = len(df)
    labels = np.full(n, 2, dtype=int)

    if USE_ATR_BARRIERS and "atr_ratio" in df.columns:
        # Grądzki-style: TP/SL scale with recent volatility
        atr_vals = df["atr_ratio"].values * closes
        tp_mult = ATR_MULT
        sl_mult = ATR_MULT
        for i in range(n - horizon):
            entry = closes[i]
            atr = atr_vals[i] if atr_vals[i] > 0 else entry * TP_PCT
            tp = entry + tp_mult * atr
            sl = entry - sl_mult * atr
            for j in range(1, horizon + 1):
                k = i + j
                if highs[k] >= tp:
                    labels[i] = 1
                    break
                if lows[k] <= sl:
                    labels[i] = 0
                    break
    else:
        # Fixed percentage barriers
        for i in range(n - horizon):
            entry = closes[i]
            tp = entry * (1 + TP_PCT)
            sl = entry * (1 - SL_PCT)
            for j in range(1, horizon + 1):
                k = i + j
                if highs[k] >= tp:
                    labels[i] = 1
                    break
                if lows[k] <= sl:
                    labels[i] = 0
                    break

    df["label"] = labels
    return df

# ── WALK-FORWARD SPLITS ────────────────────────────────────────────────

def walk_forward_splits(df, folds=FOLDS):
    """Expanding-window walk-forward chronological splits."""
    total = len(df)
    target_test = 15000
    val_size = 5000
    step = target_test
    max_possible = max(1, (total - val_size - target_test) // step)
    max_folds = min(folds, max_possible)

    splits = []
    for i in range(max_folds):
        test_end = total - i * step
        test_start = test_end - target_test
        val_end = test_start
        val_start = val_end - val_size
        train_end = val_start

        purge = LOOKBACK_HRS
        test_idx = np.arange(test_start + purge, test_end)

        if train_end <= 0 or val_start <= 0 or len(test_idx) < 50:
            continue

        splits.append({
            "train_idx": np.arange(0, train_end),
            "val_idx": np.arange(val_start, val_end),
            "test_idx": test_idx,
        })
    return splits[::-1]

# ── EXECUTION FILTER ──────────────────────────────────────────────────

def cost_aware_filter(probs, prev_pos, lam=None, cost=None):
    """Bysik-style filter: trade only when confidence exceeds cost-adjusted turnover threshold."""
    lam = lam if lam is not None else LAMBDA
    cost = cost if cost is not None else COST
    p_short = float(probs[0])
    p_long = float(probs[1])
    p_flat = float(probs[2])

    if p_flat >= p_short and p_flat >= p_long:
        desired = 2
    elif p_short > p_long:
        desired = 0
    else:
        desired = 1

    turnover = 0 if desired == prev_pos else 1
    required = lam * cost * turnover

    if desired == 2:
        return 2
    if desired == 0 and (p_short - p_long) > required:
        return 0
    if desired == 1 and (p_long - p_short) > required:
        return 1
    return 2

# ── PIPELINE ──────────────────────────────────────────────────────────

def main():
    # 1. Load BTC 5m data
    df = fetch_data()

    # 2. Multi-timeframe resampled BTC features
    print("Building multi-timeframe features...")
    tf_resampled = add_resampled_features(df)
    df = df.join(tf_resampled, how="left")

    # 3. Macro proxies (forward-filled daily to 5m)
    print("Loading macro features...")
    macro = load_macro_data(df.index)
    df = add_macro_signals(df, macro)

    # 4. Cross-asset features from multi-asset 5m data
    if USE_MULTI_ASSET and Path(MULTI_ASSET_FILE).exists():
        print("Adding cross-asset features...")
        df = add_multi_asset_features(df, MULTI_ASSET_FILE)
    else:
        if USE_MULTI_ASSET:
            print(f"[multi] {MULTI_ASSET_FILE} not found, skipping cross-asset features")

    # 4b. Equities/ETF regime features
    print("Adding equities regime features...")
    eq_regime = build_equities_regime(df)
    df = df.join(eq_regime, how="left")

    # 5. 5m technicals
    print("Deriving 5m technicals...")
    df = derive_features(df)

    # 5. Build feature matrix
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.sort_index()

    # Keep only features that exist
    features = [f for f in ALL_FEATURES if f in df.columns]
    print(f"  Active features ({len(features)}): {features}")

    X = df[features].values
    y = df["label"].values if "label" in df.columns else None

    # 6. Labels (only if not already present after derive)
    if y is None:
        print("Computing triple-barrier labels...")
        df = triple_barrier_labels(df)
    else:
        print("Labels already present")

    counts = df["label"].value_counts().sort_index()
    print(f"Label distribution: short={counts.get(0,0)}, long={counts.get(1,0)}, flat={counts.get(2,0)}")

    # Refresh after label assignment
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    X = df[features].values
    y = df["label"].values

    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 7. Walk-forward
    splits = walk_forward_splits(df)
    print(f"Walk-forward folds: {len(splits)}")

    fold_metrics = []
    for fi, sp in enumerate(splits):
        print(f"  Fold {fi+1}/{len(splits)}: train={len(sp['train_idx'])}, "
              f"val={len(sp['val_idx'])}, test={len(sp['test_idx'])}")
        X_tr, y_tr = X[sp["train_idx"]], y[sp["train_idx"]]
        X_val, y_val = X[sp["val_idx"]], y[sp["val_idx"]]
        X_te, y_te = X[sp["test_idx"]], y[sp["test_idx"]]
        if len(X_tr) < 100 or len(X_te) < 10:
            continue

        model = xgb.XGBClassifier(
            objective="multi:softmax",
            num_class=3,
            max_depth=4,
            learning_rate=0.05,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            n_jobs=_cfg.N_JOBS,  # physical cores - 1 (headroom for OS + control)
            random_state=42,
            early_stopping_rounds=30,
            eval_metric="mlogloss",
            class_weight="balanced",
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        best_trees = getattr(model, "best_ntree_limit", None) or model.n_estimators
        print(f"    Trained ({best_trees} trees)")

        probs = model.predict_proba(X_te)
        preds_raw = model.predict(X_te)

        filtered_pos = []
        margins = []
        prev = 2
        for p in probs:
            prev = cost_aware_filter(p, prev)
            filtered_pos.append(prev)
            margins.append(float(p[1]) - float(p[0]))
        filtered_pos = np.array(filtered_pos)
        margins = np.array(margins)

        acc = accuracy_score(y_te, preds_raw)
        f1 = f1_score(y_te, preds_raw, average="macro", zero_division=0)
        n_trades = np.sum(filtered_pos != 2)
        fold_metrics.append({
            "fold": fi + 1,
            "n_test": len(y_te),
            "n_trades": n_trades,
            "accuracy": acc,
            "f1_macro": f1,
            "pct_trades": n_trades / len(y_te) * 100,
            "mean_margin": margins.mean(),
            "min_margin": margins.min(),
            "max_margin": margins.max(),
            "required": LAMBDA * COST,
        })

    metrics_df = pd.DataFrame(fold_metrics)
    if metrics_df.empty:
        print("\nNo valid folds produced results.")
        return
    print("\n=== Walk-Forward Results ===")
    print(metrics_df.to_string(index=False))
    print("\n=== Summary (mean across folds) ===")
    print(metrics_df[["accuracy", "f1_macro", "pct_trades", "mean_margin", "required"]].mean().to_string())
    print(f"\nTotal trades across all folds: {metrics_df['n_trades'].sum():,}")
    print("Done.")

if __name__ == "__main__":
    main()
