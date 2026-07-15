from __future__ import annotations
"""Canonical trading engine: pure functions for rules, signals, and portfolio mechanics."""
import numpy as np
from scipy.stats import norm
import pandas as pd
from pathlib import Path
import csv

# ---------------------------------------------------------------------------
# Small internal helpers (keep signal bodies readable, no behavior change)
# ---------------------------------------------------------------------------
def _col(df: pd.DataFrame, name: str) -> pd.Series:
    """Extract a price column as a Series sharing ``df``'s index."""
    return pd.Series(df[name].values, index=df.index)


def _ma_recapture_core(close: pd.Series, period: int = 30,
                          extra_filter: pd.Series | None = None,
                          ma: pd.Series | None = None) -> tuple[pd.Series, pd.Series]:
    """Shared MA-recapture entry/exit.

    Entry: price recaptures the MA (<= MA on prior bar, strictly above now).
    Exit:  price closes back below the MA.
    ``ma`` may be passed precomputed (e.g. an EMA) to vary the average
    type; otherwise a rolling SMA of ``period`` is used.
    ``extra_filter`` (a 0/1 or bool Series) ANDs onto the entry when given,
    preserving each variant's distinct gating logic.
    """
    if ma is None:
        ma = close.rolling(period, min_periods=1).mean()
    entry = ((close > ma) & (close.shift(1) <= ma.shift(1))).astype(int)
    if extra_filter is not None:
        entry = (entry & extra_filter).astype(int)
    exit_sig = (close < ma).astype(int)
    return entry, exit_sig


# -----------------------------
# Rule: Donchian breakout
# -----------------------------
def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    plus_si = 100 * plus_dm.rolling(period).mean() / (atr + 1e-12)
    minus_si = 100 * minus_dm.rolling(period).mean() / (atr + 1e-12)
    dx = (plus_si - minus_si).abs() / (plus_si + minus_si + 1e-12) * 100
    return dx.rolling(period).mean()


def default_regime(close_market: pd.Series, day_index: int, adx_threshold: float = 20.0, vol_threshold: float = 0.25) -> str:
    """Market-wide regime gate based on aggregated close series."""
    cur = int(day_index)
    if cur < 25:
        return "trend"
    c = close_market.iloc[: cur + 1]
    high = c.rolling(2).max()
    low = c.rolling(2).min()
    a = float(adx(high, low, c).iloc[cur])
    vol = float(c.pct_change().rolling(20).std().iloc[cur])
    if pd.isna(a) or pd.isna(vol):
        return "trend"
    if a < adx_threshold or vol < vol_threshold:
        return "chop"
    return "trend"


# -----------------------------
# Rule: Donchian breakout
# -----------------------------
def donchian_signal(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int = 40) -> pd.Series:
    """Return long entry signal series: 1 when close > `lookback`-day highest high (shifted 1)."""
    don_high = high.rolling(lookback, min_periods=1).max().shift(1)
    return (close > don_high).astype(int)

# --- Regime signals centralized from research (CCI/REI/Williams %R) ---
def cci_signals(df, period: int = 20):
    """CCI momentum: entry when >0 and rising, exit when <0 and falling."""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    tp = (high + low + close) / 3.0
    sma_tp = pd.Series(tp, index=df.index).rolling(period, min_periods=1).mean()
    mad = pd.Series(tp, index=df.index).rolling(period, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad + 1e-12)
    entry = ((cci > 0) & (cci.diff() > 0)).astype(int)
    exit_sig = ((cci < 0) & (cci.diff() < 0)).astype(int)
    return entry, exit_sig


def rei_signals(df, period: int = 14):
    """REI-style momentum."""
    close = pd.Series(df["close"].values, index=df.index)
    high = pd.Series(df["high"].values, index=df.index)
    low = pd.Series(df["low"].values, index=df.index)
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    up = up_move.where((up_move > 0) & (up_move > down_move), 0).fillna(0)
    down = down_move.where((down_move > 0) & (down_move > up_move), 0).fillna(0)
    rng = (high - low).rolling(period, min_periods=1).mean()
    rei = 100 * (up.rolling(period, min_periods=1).sum() - down.rolling(period, min_periods=1).sum()) / (rng + 1e-12)
    entry = ((rei > 0) & (rei.diff() > 0)).astype(int)
    exit_sig = (rei < -20).astype(int)
    return entry, exit_sig


def williams_r_signals(df, period: int = 14):
    """Williams %R (Larry Williams, 1973): mean-reversion oscillator, 0 to -100.
    Literature-standard usage:
      - Oversold: %R < -80 ; Overbought: %R > -20
      - Entry: %R crosses UP through -80 (price bounces off the oversold floor)
      - Exit:  %R crosses UP through -20 (price has recovered -> sell into strength)
    Exit is gated on recovery, so we sell into strength, never at the oversold bottom.
    NOTE: UNVALIDATED vs the legacy buggy exit (see williams_r_buggy_exit). On a
    3-window chop replay the buggy exit actually outperformed this one
    (+11.8% vs -5.0% mean), though the buggy version sells at the oversold bottom
    (pathological). Treat this as the theoretically-correct form pending further test.
    """
    high = pd.Series(df["high"].values, index=df.index)
    low = pd.Series(df["low"].values, index=df.index)
    close = pd.Series(df["close"].values, index=df.index)
    highest = high.rolling(period, min_periods=1).max()
    lowest = low.rolling(period, min_periods=1).min()
    wr = -100 * (highest - close) / (highest - lowest + 1e-12)
    prev = wr.shift(1)
    entry = ((prev <= -80) & (wr > -80)).astype(int)        # cross UP through -80 (bounce starts)
    exit_sig = ((prev <= -20) & (wr > -20)).astype(int)     # cross UP through -20 (recovered -> sell into strength)
    return entry, exit_sig


def williams_r_buggy_exit(df, period: int = 14):
    """LEGACY / UNVALIDATED. Kept for comparison only — do NOT use live.
    Original pre-2026-07-14 Williams exit: sells whenever %R is below -20 AND
    falling. This fires at the oversold BOTTOM (e.g. wr=-97), crystallizing the
    worst price. It happened to outperform the recovery exit on one 90d alt window
    (acts like a tight churn stop), but is theoretically wrong and pathological.
    Preserved as a named rule so the result is reproducible, not silently lost.
    """
    high = pd.Series(df["high"].values, index=df.index)
    low = pd.Series(df["low"].values, index=df.index)
    close = pd.Series(df["close"].values, index=df.index)
    highest = high.rolling(period, min_periods=1).max()
    lowest = low.rolling(period, min_periods=1).min()
    wr = -100 * (highest - close) / (highest - lowest + 1e-12)
    entry = ((wr > -80) & (wr.diff() > 0)).astype(int)
    exit_sig = ((wr < -20) & (wr.diff() < 0)).astype(int)
    return entry, exit_sig


def tsi_signals(df, fast=13, slow=13, signal=13):
    """True Strength Index signals (common practitioner params).
    Entry: TSI > 0 and rising.
    Exit: TSI < -10 and falling (simple version).
    """
    close = df['close']
    momentum = close.diff()
    ema1 = momentum.ewm(span=fast, adjust=False).mean()
    ema2 = ema1.ewm(span=slow, adjust=False).mean()
    abs1 = momentum.abs().ewm(span=fast, adjust=False).mean()
    abs2 = abs1.ewm(span=slow, adjust=False).mean()
    tsi = 100 * ema2 / (abs2 + 1e-12)
    entry = ((tsi > 0) & (tsi.diff() > 0)).astype(int)
    exit_sig = ((tsi < -10) & (tsi.diff() < 0)).astype(int)
    return entry, exit_sig


def bop_signals(df, smooth=20):
    """Balance of Power signals.
    BOP = (C - L) / (H - L)
    Entry: BOP > 0 and smoothed BOP rising.
    """
    bop = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-12)
    sma_bop = bop.rolling(smooth).mean()
    entry = ((bop > 0) & (sma_bop.diff() > 0)).astype(int)
    exit_sig = (bop < -0.2).astype(int)
    return entry, exit_sig


def mtf_confirm_signals(df):
    """Multi-timeframe confirm signals (sma50 > sma200 and price > sma50).
    Simple trend confirmation used in prior testing.
    """
    close = df['close']
    sma50 = close.rolling(50, min_periods=1).mean()
    sma200 = close.rolling(200, min_periods=1).mean()
    entry = ((sma50 > sma200) & (close > sma50)).astype(int)
    exit_sig = (close < sma50).astype(int)
    return entry, exit_sig


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Average True Range."""
    tr = true_range(high, low, close)
    return tr.rolling(period, min_periods=1).mean()


def chandelier_exit(df: pd.DataFrame, period: int = 22, mult: float = 3.0) -> pd.Series:
    """Chandelier Exit long (Le Beau / Elder,  literature-backed trailing stop).
    Returns 1 on days close < (prior rolling HH - mult * ATR).
    """
    high = df.get('high', df['close'])
    low = df.get('low', df['close'])
    close = df['close']
    atr_val = atr(high, low, close, period)
    hh = high.rolling(period, min_periods=1).max()
    chand = hh - mult * atr_val
    return (close < chand.shift(1)).astype(int)


def rsi_signals(df, period: int = 14):
    """RSI (Wilder 1978): momentum oscillator 0-100.
    Literature-standard mean-reversion usage:
      - Oversold entry: RSI < 30 ; Exit on recovery: RSI crosses UP through 50
    (50 = the neutral midline; exiting there captures the bounce without
    waiting for full overbought >70, which often means the move already faded.)
    """
    close = pd.Series(df["close"].values, index=df.index)
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1/period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/period, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    rsi = 100 - 100 / (1 + rs)
    prev = rsi.shift(1)
    entry = (rsi < 30).astype(int)
    exit_sig = ((prev <= 50) & (rsi > 50)).astype(int)
    return entry, exit_sig


def stochastic_signals(df, k_period: int = 14, d_period: int = 3):
    """Stochastic oscillator %K/%D (Lane, 1950s): bounded 0-100 momentum.
    Same mean-reversion family as CCI/Williams %R. Literature-standard:
      - Oversold entry: %K crosses UP through 20 (price bounces off the floor)
      - Exit: %K crosses UP through 80 (recovered -> sell into strength)
    Exit is gated on recovery, so we sell into strength, never at the bottom
    (mirrors the Williams %R lesson: don't crystallize the oversold low).
    %D is the signal line (SMA of %K); we trigger on %K for timeliness.
    """
    high = _col(df, "high"); low = _col(df, "low"); close = _col(df, "close")
    ll = low.rolling(k_period, min_periods=1).min()
    hh = high.rolling(k_period, min_periods=1).max()
    k = 100 * (close - ll) / (hh - ll + 1e-12)
    prev = k.shift(1)
    entry = ((prev <= 20) & (k > 20)).astype(int)          # cross UP through 20 (bounce)
    exit_sig = ((prev <= 80) & (k > 80)).astype(int)       # cross UP through 80 (recovered)
    return entry, exit_sig


def mfi_signals(df, period: int = 14):
    """Money Flow Index (MFI): volume-weighted RSI, bounded 0-100.
    Adds volume (your friend's angle) to the mean-reversion oscillator family:
    'volume RSI' — money flowing in/out, not just price. Literature-standard:
      - Oversold entry: MFI crosses UP through 20
      - Exit: MFI crosses UP through 80 (recovered -> sell into strength)
    Raw MFI uses a typical-price * volume money-flow ratio.
    """
    high = _col(df, "high"); low = _col(df, "low"); close = _col(df, "close")
    vol = _col(df, "volume")
    tp = (high + low + close) / 3.0
    mf = tp * vol
    pos = mf.where(tp > tp.shift(1), 0.0).rolling(period, min_periods=1).sum()
    neg = mf.where(tp < tp.shift(1), 0.0).rolling(period, min_periods=1).sum()
    mfi = 100 - 100 / (1 + pos / (neg + 1e-12))
    prev = mfi.shift(1)
    entry = ((prev <= 20) & (mfi > 20)).astype(int)        # cross UP through 20
    exit_sig = ((prev <= 80) & (mfi > 80)).astype(int)      # cross UP through 80
    return entry, exit_sig


def inverse_fisher_rsi_signals(df, period: int = 14, smooth: int = 5):
    """Inverse Fisher Transform (IFT) of RSI — Ehlers (MESA Software).
    The IFT normalizes an oscillator to a near-Gaussian +/-1 range, sharpening
    the extremes so oversold/overbought are clearer. Applied to RSI:
      - Entry: IFT crosses UP through -0.5 (deep oversold, beginning to revert)
      - Exit:  IFT crosses UP through +0.5 (recovered -> sell into strength)
    Tests whether NORMALIZING an existing oscillator (RSI) adds edge vs the raw
    oscillator — a within-family sharpening, not a new mechanism.
    """
    close = _col(df, "close")
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    rsi = 100 - 100 / (1 + rs)
    x = (rsi / 100.0 - 0.5) * 2.0                      # map 0..100 -> -1..1
    x = x.rolling(smooth, min_periods=1).mean()         # smooth per Ehlers
    ift = (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)    # inverse Fisher transform
    prev = ift.shift(1)
    entry = ((prev <= -0.5) & (ift > -0.5)).astype(int)  # cross UP through -0.5
    exit_sig = ((prev <= 0.5) & (ift > 0.5)).astype(int)  # cross UP through +0.5
    return entry, exit_sig


def atr_trailing_exit(df: pd.DataFrame, period: int = 14, mult: float = 2.0) -> pd.Series:
    """Pure ATR trailing stop (ratchet from highest high).
    Exit long when close < (cummax(high) - mult * ATR).
    Simpler than Chandelier; no fixed lookback HH.
    """
    high = df.get('high', df['close'])
    low = df.get('low', df['close'])
    close = df['close']
    atr_val = atr(high, low, close, period)
    hh = high.cummax()
    trail = hh - mult * atr_val
    return (close < trail.shift(1)).astype(int)

def apply_trailing_overlay(base_exit: pd.Series, trailing_exit: pd.Series) -> pd.Series:
    """OR base and trailing exits."""
    return ((base_exit.fillna(0) == 1) | (trailing_exit.fillna(0) == 1)).astype(int)


def ma30_recapture_signals(df, period: int = 30):
    """MA30 Recapture rule (simple moving average recapture).

    Entry: price was <= MA30 on previous bar and now closes strictly above it.
           Classic "recapture" / pullback-to-MA continuation long signal.
    Exit: price closes back below the MA30.

    Literature/practitioner context:
    - Used in trend-continuation and breakout systems (e.g. variations of
      MA pullback entries, Turtle-style recapture logic, many swing traders).
    - On daily timeframe for volatile names it acts as a relatively fast
      trend filter / mean-reversion-to-trend entry.
    """
    close = df["close"]
    return _ma_recapture_core(close, period)


# --- MA Recapture family (investigation variants) ---
# Each variant is the same recapture core plus ONE extra entry filter.

def ma30_recapture_ema(df, period: int = 30):
    """MA30 Recapture using EMA (faster reaction than SMA)."""
    close = df["close"]
    ma = close.ewm(span=period, adjust=False).mean()
    return _ma_recapture_core(close, period, extra_filter=None, ma=ma)


def ma30_recapture_rising(df, period: int = 30):
    """SMA30 Recapture + rising MA filter (momentum confirmation)."""
    close = df["close"]
    ma = close.rolling(period, min_periods=1).mean()
    rising = ma > ma.shift(1)
    return _ma_recapture_core(close, period, extra_filter=rising, ma=ma)


def ma30_50_recapture(df):
    """Recapture MA30 while price remains above MA50 (longer-term trend gate)."""
    close = df["close"]
    ma30 = close.rolling(30, min_periods=1).mean()
    ma50 = close.rolling(50, min_periods=1).mean()
    gate = close > ma50
    return _ma_recapture_core(close, period=30, extra_filter=gate, ma=ma30)


def ma30_recapture_lowvol(df, period: int = 30, vol_window: int = 20):
    """MA30 Recapture only in low-volatility conditions (quieter pullbacks)."""
    close = df["close"]
    vol = realized_vol(close, vol_window)
    vol_median = vol.rolling(vol_window * 2, min_periods=vol_window).median()
    low_vol = vol < vol_median
    return _ma_recapture_core(close, period, extra_filter=low_vol)


def ma30_recapture_vol_expand(df, period: int = 30, vol_window: int = 20):
    """MA30 Recapture with expanding volatility filter (vol rising)."""
    close = df["close"]
    vol = realized_vol(close, vol_window)
    expanding = vol > vol.shift(1)
    return _ma_recapture_core(close, period, extra_filter=expanding)


# --- Volume-aware MA recapture variants ---

def ma30_recapture_high_volume(df, period: int = 30, vol_ma: int = 20):
    """MA30 Recapture confirmed by above-average volume."""
    close = df["close"]
    vol = df["volume"]
    vol_ma_s = vol.rolling(vol_ma, min_periods=vol_ma // 2).mean()
    high_vol = vol > vol_ma_s
    return _ma_recapture_core(close, period, extra_filter=high_vol)


def ma30_recapture_volume_surge(df, period: int = 30, vol_ma: int = 20, surge_mult: float = 1.5):
    """MA30 Recapture with volume surge (>= surge_mult x recent avg)."""
    close = df["close"]
    vol = df["volume"]
    vol_ma_s = vol.rolling(vol_ma, min_periods=vol_ma // 2).mean()
    surge = vol >= (vol_ma_s * surge_mult)
    return _ma_recapture_core(close, period, extra_filter=surge)



def get_regime_signals(rule_name: str, df: pd.DataFrame):
    rule = rule_name.lower()
    if rule in ("cci", "cci20"):
        return cci_signals(df)
    if rule in ("rei",):
        return rei_signals(df)
    if rule in ("rsi", "rsi_signals"):
        return rsi_signals(df)
    if rule in ("donchian40", "d40", "donchian"):
        # donchian_signal returns only the entry series; build the matching exit
        # (close below lookback-low) so it satisfies the (entry, exit) contract.
        entry = donchian_signal(df["high"], df["low"], df["close"], lookback=40)
        don_low = df["low"].rolling(40, min_periods=1).min().shift(1)
        exit_sig = (df["close"] < don_low).astype(int)
        return entry, exit_sig
    if rule in ("tsi", "tsi_signals"):
        return tsi_signals(df)
    if rule in ("bop", "bop_signals"):
        return bop_signals(df)
    if rule in ("mtf", "mtf_confirm", "ma_confirm"):
        return mtf_confirm_signals(df)
    if rule in ("williams", "williams_r", "wr"):
        return williams_r_signals(df)
    if rule in ("williams_r_buggy", "williams_buggy", "wr_buggy"):
        return williams_r_buggy_exit(df)
    # MA recapture family
    if rule in ("ma30", "ma30_recapture", "ma30_sma", "recapture", "ma_recapture"):
        return ma30_recapture_signals(df)
    if rule in ("ma30_ema", "ema_recapture"):
        return ma30_recapture_ema(df)
    if rule in ("ma30_rising", "recapture_rising"):
        return ma30_recapture_rising(df)
    if rule in ("ma30_50", "ma30_50_recapture", "dual_ma_recapture"):
        return ma30_50_recapture(df)
    if rule in ("ma30_recapture_lowvol", "lowvol_recapture"):
        return ma30_recapture_lowvol(df)
    if rule in ("ma30_recapture_vol_expand", "vol_expand_recapture"):
        return ma30_recapture_vol_expand(df)
    # Volume-aware MA recapture
    if rule in ("ma30_recapture_high_volume", "high_volume_recapture", "volume_recapture"):
        return ma30_recapture_high_volume(df)
    if rule in ("ma30_recapture_volume_surge", "volume_surge_recapture"):
        return ma30_recapture_volume_surge(df)
    # Bollinger Band Width Percentile (vol-regime rule)
    if rule in ("bbwp", "bbwp_signals", "bband_width_pct"):
        return bbwp_signals(df)
    # --- Mean-reversion oscillator family (CCI-family extensions) ---
    if rule in ("stochastic", "stoch", "stochastic_k", "%k", "stoch_k"):
        return stochastic_signals(df)
    if rule in ("mfi", "money_flow_index", "mfi_signals"):
        return mfi_signals(df)
    if rule in ("ift_rsi", "inverse_fisher_rsi", "fisher_rsi"):
        return inverse_fisher_rsi_signals(df)
    # Hurst regime filter (H > 0.5 = trend)
    if rule in ("hurst_trend", "hurst"):
        # Use hurst_regime as a dynamic rule switch - caller usually pairs with regime_rule_map
        return None  # special case handled via regime_fn
    raise ValueError(f"Unknown regime rule: {rule_name}")


# --- Additional regime features (literature-backed) ---
def kaufman_efficiency_ratio(close: pd.Series, period: int = 20) -> pd.Series:
    """Kaufman Efficiency Ratio: |net change| / sum of absolute changes.
    Higher values = more directional / trending.
    Classic filter for trend vs chop (Kaufman, many practitioners).
    """
    change = close.diff(period).abs()
    volatility = close.diff().abs().rolling(period, min_periods=1).sum()
    er = change / (volatility + 1e-12)
    return er


def choppiness_index(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Choppiness Index: 100 * log10( sum(|H-L| over period) / (period-HH - period-LL) ) / log10(period).
    CI ~100 = maximally choppy; CI ~0 = strongly directional/trending.
    Regime use: CI > threshold (~61.8) => chop; else => trend.
    Robust: clips rng to avoid log(0)/NaN when the window high==low.
    """
    atr1 = (high - low).abs()
    sum_atr = atr1.rolling(period, min_periods=1).sum()
    rng = (high.rolling(period, min_periods=1).max() - low.rolling(period, min_periods=1).min()).clip(lower=1e-12)
    ci = 100.0 * (np.log10(sum_atr + 1e-12) - np.log10(rng)) / np.log10(max(period, 2))
    return ci.replace([np.inf, -np.inf], np.nan).fillna(0.0)



def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Simple realized volatility (std of returns)."""
    return close.pct_change().rolling(window, min_periods=5).std()


def bb_width(close: pd.Series, period: int = 20, std_mult: float = 2.0) -> pd.Series:
    """Bollinger Band width as fraction of the middle band: (upper - lower) / mid."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    return (upper - lower) / sma


def bbwp(close: pd.Series, period: int = 20, std_mult: float = 2.0,
         pct_lookback: int = 50, percentile: bool = True) -> pd.Series:
    """Bollinger Band Width Percentile (BBWP).

    BBWP = rolling percentile rank of the current BB width within its trailing
    `pct_lookback` window. High BBWP (~1.0) = band EXPANDED (volatile / breakout /
    trending regime); low BBWP (~0.0) = band SQUEEZED (calm / chop regime).

    Literature: Bollinger Band Width squeeze is the classic vol-regime signal
    (J. Bollinger; "squeeze" precedes expansions). BBWP percentile form (e.g.
    as used in TTR's `BBands` + percentile rank) is a standard regime/vol filter.
    """
    width = bb_width(close, period=period, std_mult=std_mult)
    if percentile:
        return width.rolling(pct_lookback, min_periods=5).apply(
            lambda x: (x[-1] >= x).mean(), raw=True
        )
    return width


def bbwp_signals(df: pd.DataFrame, period: int = 20, std_mult: float = 2.0,
                 pct_lookback: int = 50, enter_thr: float = 0.80,
                 exit_thr: float = 0.20) -> tuple[pd.Series, pd.Series]:
    """Directional BBWP rule: enter LONG when BB width is EXPANDED (BBWP > enter_thr,
    i.e. volatility breakout / trend regime) AND price breaks above the upper band;
    exit when BBWP contracts below exit_thr (squeeze returns) OR price < lower band.

    This makes BBWP a volatility-gate on a Bollinger breakout, not a bare breakout.
    """
    close = df["close"]
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_mult * std
    lower = sma - std_mult * std
    wp = bbwp(close, period=period, std_mult=std_mult, pct_lookback=pct_lookback)
    entry = ((close > upper) & (wp > enter_thr)).astype(int)
    exit_sig = ((close < lower) | (wp < exit_thr)).astype(int)
    return entry, exit_sig

# --- Portfolio-level Volatility Targeting (literature style: Harvey 2018, Moreira & Muir 2017, Barroso & Santa-Clara 2015) ---
def compute_vol_scale(ret_history: list[float], target_vol: float = 0.15, lookback: int = 20, 
                      bounds: tuple[float, float] = (0.25, 1.5)) -> float:
    """Simple realized-vol scaling to target constant portfolio volatility.
    scale = target_vol / recent_annualized_vol
    Clipped to bounds. Returns 1.0 if insufficient history.
    Matches the spirit of volatility-managed portfolios: reduce exposure when recent vol is high.
    """
    if len(ret_history) < lookback:
        return 1.0
    recent = ret_history[-lookback:]
    vol = float(np.std(recent) * np.sqrt(365))  # annualize assuming daily
    if vol <= 0:
        return 1.0
    scale = target_vol / vol
    return float(np.clip(scale, bounds[0], bounds[1]))


def make_vol_scale_fn(target_vol: float = 0.15, lookback: int = 20, bounds: tuple[float, float] = (0.25, 1.5)):
    """Factory for a vol_scale_fn compatible with simulate_portfolio."""
    def _fn(day_i, equity, ret_history):
        return compute_vol_scale(ret_history, target_vol, lookback, bounds)
    return _fn


# --- Hurst Exponent (persistence / regime filter) ---
def hurst_exponent(series: pd.Series, max_lag: int = 20) -> float:
    """Classic R/S Hurst exponent estimate.
    H > 0.5 → persistent / trending (favor momentum rules)
    H < 0.5 → anti-persistent / mean-reverting
    Returns NaN if not enough data.
    """
    if len(series) < max_lag + 10:
        return np.nan
    lags = range(2, max_lag + 1)
    rs = []
    for lag in lags:
        if lag >= len(series):
            break
        # Rescaled range
        diffs = np.diff(np.log(series.iloc[:lag* (len(series)//lag) ])) if len(series) > lag else np.diff(np.log(series))
        if len(diffs) < lag:
            continue
        # Chunked
        chunks = len(diffs) // lag
        if chunks < 1:
            continue
        rs_chunk = []
        for i in range(chunks):
            chunk = diffs[i*lag:(i+1)*lag]
            if len(chunk) == 0:
                continue
            mean_chunk = np.mean(chunk)
            dev = np.cumsum(chunk - mean_chunk)
            r = np.max(dev) - np.min(dev)
            s = np.std(chunk) if np.std(chunk) > 0 else 1e-12
            rs_chunk.append(r / s)
        if rs_chunk:
            rs.append(np.mean(rs_chunk))
    if len(rs) < 3:
        return np.nan
    # log-log regression
    lags_arr = np.array(list(lags)[:len(rs)])
    rs_arr = np.array(rs)
    if np.any(rs_arr <= 0):
        rs_arr = np.maximum(rs_arr, 1e-12)
    slope, _ = np.polyfit(np.log(lags_arr), np.log(rs_arr), 1)
    h = slope
    return float(np.clip(h, 0.0, 1.0))


def rolling_hurst(series: pd.Series, window: int = 60, max_lag: int = 20) -> pd.Series:
    """Rolling Hurst for regime filtering."""
    return series.rolling(window).apply(lambda x: hurst_exponent(pd.Series(x), max_lag), raw=False)


def hurst_regime(close: pd.Series, day_index: int, window: int = 60, threshold: float = 0.5) -> str:
    """Simple regime using Hurst: > threshold = trend, else chop."""
    if day_index < window:
        return "chop"
    hist = close.iloc[max(0, day_index-window+1):day_index+1]
    h = hurst_exponent(hist)
    if np.isnan(h):
        return "chop"
    return "trend" if h > threshold else "chop"

def ma_crossover_regime(close_market: pd.Series, day_index: int, short: int = 50, long: int = 200) -> str:
    """Simple MA crossover trend filter (friend suggestion).

    When short MA > long MA → "trend" regime (use trend-following strat, e.g. REI).
    Otherwise → "chop" regime (use mean-reversion / oscillation strat, e.g. Williams %R).

    Classic trend filter idea: different behavior/strat in clear uptrend vs everything else.
    Defaults use 50/200 for major trend; faster (20/50) also reasonable for alt daily.
    """
    cur = int(day_index)
    if cur < long:
        return "chop"  # conservative on insufficient history
    c = close_market.iloc[:cur + 1]
    if len(c) < long:
        return "chop"
    short_ma = c.rolling(short, min_periods=short).mean().iloc[cur]
    long_ma = c.rolling(long, min_periods=long).mean().iloc[cur]
    if pd.isna(short_ma) or pd.isna(long_ma):
        return "chop"
    return "trend" if short_ma > long_ma else "chop"




# --- Improved regime detector (literature-grounded) ---
def compute_regime(
    close_market: pd.Series,
    day_index: int,
    adx_threshold: float = 22.0,
    vol_threshold: float = 0.22,
    er_threshold: float = 0.35,
    min_regime_bars: int = 3,
    use_hysteresis: bool = True,
    hysteresis_bars: int = 2,
    method: str = "rule",   # "rule" | "hurst" | "hmm" | "hybrid" | "ma"
    hmm_model=None,
    hurst_window: int = 60,
    ma_short: int = 50,
    ma_long: int = 200,
    hurst_threshold: float = 0.5,
) -> str:
    """
    Improved market regime detector.

    Literature grounding (key sources):
    - ADX for trend strength (Wilder + many: Darwinex, Alpha Architect summaries, LuxAlgo, Reddit algotrading).
    - Realized vol / ATR% for high-vol vs calm (QuantStart HMM papers, volatility filter literature).
    - Kaufman Efficiency Ratio (ER) as a clean directional vs chop measure (repeatedly recommended in practitioner regime filters).
    - Hysteresis / minimum duration to avoid whipsaw (practical consensus across sources).
    - HMM option: GaussianHMM on returns (classic for latent low/high-vol or trend/chop regimes; see QuantStart QSTrader example and multiple HMM regime papers).
    - Hurst exponent for persistence (H > 0.5 = trending/persistent; classic from Mandelbrot, applied in quant regime filters).
    - Theoretical context: Hamilton (1989) Markov switching; Zakamulin & Giner (semi-Markov) showing regime affects optimal trend rules and duration dependence.

    Logic:
      - method="rule": ADX + ER + vol (default)
      - method="hurst": Pure Hurst persistence (H > hurst_threshold -> trend)
      - method="hmm": HMM
      - method="hybrid": rule-based with Hurst confirmation for trend

    Returns: "trend" or "chop" (extendable).
    """
    cur = int(day_index)
    if cur < 30:
        return "trend"

    c = close_market.iloc[:cur + 1]
    if len(c) < 25:
        return "trend"

    # Compute features
    high = c.rolling(2).max()
    low = c.rolling(2).min()
    a = float(adx(high, low, c).iloc[cur])
    vol = float(realized_vol(c).iloc[cur])
    er = float(kaufman_efficiency_ratio(c).iloc[cur])

    if pd.isna(a) or pd.isna(vol) or pd.isna(er):
        return "trend"

    # Simple HMM path (optional, requires fitted hmmlearn GaussianHMM)
    if method in ("hmm", "hybrid") and hmm_model is not None:
        try:
            rets = c.pct_change().dropna().values.reshape(-1, 1)
            if len(rets) > 5:
                states = hmm_model.predict(rets[-min(100, len(rets)):])
                # Convention: treat the most recent state; map high-vol state to chop if desired
                last_state = states[-1]
                # Heuristic: if model has 2 states, we can label post-hoc or use means
                if hasattr(hmm_model, "means_"):
                    means = hmm_model.means_.flatten()
                    if abs(means[last_state]) > 0.01:  # rough vol proxy via mean return magnitude
                        hmm_regime = "trend"
                    else:
                        hmm_regime = "chop"
                else:
                    hmm_regime = "trend" if last_state == 0 else "chop"
                if method == "hmm":
                    return hmm_regime
                # hybrid: require agreement or default to rule
        except Exception:
            pass  # fall through to rule

    if method == "ma":
        # Use MA crossover as the regime filter (short MA > long MA = trend)
        return ma_crossover_regime(close_market, cur, short=ma_short, long=ma_long)

    if method == "choppiness":
        # Choppiness Index on the market proxy: CI high => chop, low => trend.
        hi = c.rolling(2).max(); lo = c.rolling(2).min()
        ci = float(choppiness_index(hi, lo, c, period=14).iloc[cur])
        if pd.isna(ci):
            return "trend"
        return "chop" if ci > 61.8 else "trend"

    if method == "kaufman":
        # Kaufman ER-only adaptive: ER above threshold => trend, else chop.
        # (No ADX/vol gating — pure directional efficiency, the simplest regime filter.)
        er = float(kaufman_efficiency_ratio(c).iloc[cur])
        if pd.isna(er):
            return "trend"
        return "trend" if er >= er_threshold else "chop"

    if method == "mesa":
        # Mesa/British-Bank-style adaptive: ER with an adaptive (vol-scaled) threshold.
        # When volatility is high, require a stronger ER to call trend (avoids whipsaw).
        er = float(kaufman_efficiency_ratio(c).iloc[cur])
        vol = float(realized_vol(c).iloc[cur])
        if pd.isna(er) or pd.isna(vol):
            return "trend"
        adaptive_thr = er_threshold * (1.0 + 2.0 * vol)
        return "trend" if er >= adaptive_thr else "chop"

    if method == "bbwp":
        # Bollinger Band Width Percentile regime filter: band EXPANDED (BBWP high) =>
        # volatile/breakout/trending regime => trend; SQUEEZED (BBWP low) => chop.
        wp = float(bbwp(c, period=20, std_mult=2.0, pct_lookback=50).iloc[cur])
        if pd.isna(wp):
            return "trend"
        return "chop" if wp < 0.20 else "trend"

    # Core rule-based regime (improved)
    is_trending = (a >= adx_threshold) and (er >= er_threshold) and (vol <= vol_threshold)

    regime = "trend" if is_trending else "chop"

    # Basic hysteresis / persistence (prevent single-bar flips)
    if use_hysteresis and cur >= hysteresis_bars:
        prev_regimes = []
        for i in range(1, hysteresis_bars + 1):
            prev = compute_regime(
                close_market, cur - i,
                adx_threshold=adx_threshold, vol_threshold=vol_threshold,
                er_threshold=er_threshold, min_regime_bars=1,
                use_hysteresis=False, method="rule"
            )
            prev_regimes.append(prev)
        if all(r == regime for r in prev_regimes[:hysteresis_bars]):
            return regime
        # If not confirmed, stick with previous stable regime if possible
        # Simple fallback: keep last non-na
        return prev_regimes[0] if prev_regimes else regime

    return regime


# Convenience wrapper that mirrors the old compute_live_regime signature but uses the improved detector
def improved_compute_live_regime(
    price_dfs: dict,
    lookback: int = 40,
    adx_threshold: float = 22.0,
    vol_threshold: float = 0.22,
    er_threshold: float = 0.35,
    method: str = "rule",
    hmm_model=None,
    hysteresis_bars: "int | None" = None,
    ma_short: int = 50,
    ma_long: int = 200,
):
    """Live-friendly wrapper around the improved regime logic using a mean-close market proxy.

    hysteresis_bars=None -> use compute_regime's default (2). Pass an int to
    cross-validate regime-switch persistence (wider = stickier = more lag).
    ma_short/ma_long -> MA-crossover regime pair (method="ma"), the live lever.
    """
    if not price_dfs:
        return "trend"
    closes = [df["close"].iloc[-lookback:].reset_index(drop=True) for df in price_dfs.values() if len(df) > 0]
    if not closes:
        return "trend"
    min_len = min(len(c) for c in closes)
    if min_len < 10:
        return "trend"
    market = pd.concat([c.iloc[-min_len:] for c in closes], axis=1).mean(axis=1).reset_index(drop=True)
    idx = len(market) - 1
    kw = dict(
        adx_threshold=adx_threshold,
        vol_threshold=vol_threshold,
        er_threshold=er_threshold,
        method=method,
        hmm_model=hmm_model,
        ma_short=ma_short,
        ma_long=ma_long,
    )
    if hysteresis_bars is not None:
        kw["hysteresis_bars"] = hysteresis_bars
    return compute_regime(market, idx, **kw)



def compute_live_regime(price_dfs: dict, lookback: int = 30, adx_threshold: float = 20.0, vol_threshold: float = 0.25):
    """Market regime using recent data (mean close across universe)."""
    if not price_dfs:
        return "trend"
    closes = [df["close"].iloc[-lookback:].reset_index(drop=True) for df in price_dfs.values() if len(df) > 0]
    if not closes:
        return "trend"
    min_len = min(len(c) for c in closes)
    if min_len < 5:
        return "trend"
    market = pd.concat([c.iloc[-min_len:] for c in closes], axis=1).mean(axis=1).reset_index(drop=True)
    idx = len(market) - 1
    if idx < 25:
        return "trend"
    high = market.rolling(2).max()
    low = market.rolling(2).min()
    a = float(adx(high, low, market).iloc[idx])
    vol = float(market.pct_change().rolling(20, min_periods=5).std().iloc[idx])
    if pd.isna(a) or pd.isna(vol):
        return "trend"
    return "chop" if (a < adx_threshold or vol < vol_threshold) else "trend"


# -----------------------------
# Rule: Bollinger Band breakout
# -----------------------------
def bollinger_signal(close: pd.Series, period: int = 20, std_mult: float = 2.0) -> tuple[pd.Series, pd.Series]:
    """Return (entry_signal, middle_band)."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + std_mult * std
    entry = (close > upper).astype(int)
    return entry, sma


# -----------------------------
# Rule: Mean-reversion oversold bounce
# -----------------------------
def mr_bounce_signal(close: pd.Series, rsi_period: int = 14, oversold: float = 30.0, drop_lookback: int = 5, drop_threshold: float = -0.10) -> pd.Series:
    """Return signal: 1 when RSI < oversold AND drop_lookback return < drop_threshold, else 0."""
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(50)
    ret = close.pct_change(drop_lookback)
    return ((rsi < oversold) & (ret < drop_threshold)).astype(int)


# -----------------------------
# Portfolio simulator
# -----------------------------

def make_regime_gated_vol_scale_fn(regime_series: pd.Series, target_vol: float = 0.15, lookback: int = 20,
                                   bounds: tuple[float, float] = (0.25, 1.5), apply_in: str = "chop"):
    """Returns a vol_scale_fn that only applies scaling when regime == apply_in (e.g. 'chop').
    regime_series must be indexed by date.
    """
    vol_base_fn = make_vol_scale_fn(target_vol, lookback, bounds)
    def _gated_fn(day_i, equity, ret_history):
        if day_i >= len(regime_series):
            return 1.0
        current_regime = regime_series.iloc[day_i]
        if current_regime == apply_in:
            return vol_base_fn(day_i, equity, ret_history)
        return 1.0
    return _gated_fn


def simulate_portfolio(
    price_df: pd.DataFrame,
    sig_df: pd.DataFrame,
    initial: float = 1000.0,
    max_positions: int = 5,
    max_position_pct: float = 0.20,
    cost_bps: int = 8,
    slippage_bps: int = 5,
    min_equity: float = 100.0,
    exit_signal_df: pd.DataFrame | None = None,
    vol_scale_fn: callable | None = None,
    regime_fn: callable | None = None,
    regime_rule_map: dict | None = None,
    high_df: pd.DataFrame | None = None,
    low_df: pd.DataFrame | None = None,
    fair_compare_path: Path | None = None,
    fair_compare_rule: str | None = None,
    n_trials: int | None = None,
) -> dict:
    """Run equal-weight long-only portfolio simulation.
    Args:
        price_df: close prices, index=dates, columns=symbols
        sig_df: entry signals, 1/0, same shape
        initial: starting equity
        max_positions: max concurrent holdings
        max_position_pct: fraction of equity per position
        cost_bps: round-trip fee in basis points
        slippage_bps: additional slippage in bps
        min_equity: halt if equity falls below
        exit_signal_df: if provided, use this for exits instead of sig_df
        vol_scale_fn: callable(day_index, equity, ret_history) -> scale factor
        regime_fn: callable(high_series, low_series, close_series, day_index) -> "trend"|"chop"
        regime_rule_map: dict(trend=entry_df, chop=entry_df) for swapping entries
        high_df: high prices for regime_fn
        low_df: low prices for regime_fn
        fair_compare_path: optional CSV path to append a result row.
            Writes one row: rule_name, trades, sharpe, max_dd_pct, exposure_ratio, effective_sharpe.
        fair_compare_rule: name to write into the CSV when fair_compare_path is set.
    """
    cash = initial
    positions: dict[str, dict] = {}
    trades = 0
    equity_curve = []
    exposure_history = []
    peak = initial
    max_dd = 0.0
    daily_pnl = 0.0
    ret_history: list[float] = []
    dates = price_df.index.tolist()
    regime_map = None
    if regime_fn is not None and regime_rule_map is not None:
        close_market = price_df.mean(axis=1)
        regimes = []
        for day_i in range(len(dates)):
            regimes.append(regime_fn(close_market, day_i))
        regime_map = pd.Series(regimes, index=dates)

    for day_i, day in enumerate(dates):
        row_prices = price_df.loc[day]
        row_sig = sig_df.loc[day]
        row_exit = exit_signal_df.loc[day] if exit_signal_df is not None else row_sig
        if regime_map is not None:
            regime = regime_map.loc[day]
            row_sig = regime_rule_map[regime].loc[day]

        mtm = 0.0
        for sym, pos in positions.items():
            px = row_prices.get(sym, 0)
            if pd.notna(px) and px > 0:
                mtm += pos["shares"] * px
        equity = cash + mtm
        exposure_history.append(len(positions) / max_positions if max_positions > 0 else 0.0)
        equity_curve.append(equity)
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

        if len(equity_curve) > 1:
            day_ret = equity / equity_curve[-2] - 1
            ret_history.append(day_ret)
            daily_pnl += day_ret

        scale = vol_scale_fn(day_i, equity, ret_history) if vol_scale_fn is not None else 1.0

        # equity floor / daily loss circuit breaker
        if equity < min_equity or daily_pnl < -0.03 * peak:
            for sym in list(positions.keys()):
                px = row_prices.get(sym, 0)
                if pd.notna(px) and px > 0:
                    cash += positions[sym]["shares"] * px * (1 - cost_bps / 10000)
                    trades += 1
            positions.clear()
            daily_pnl = 0.0
            continue

        # exits
        for sym in list(positions.keys()):
            if sym not in positions:
                continue
            if row_exit.get(sym, 0) == 0:
                px = row_prices.get(sym, 0)
                if pd.notna(px) and px > 0:
                    cash += positions[sym]["shares"] * px * (1 - cost_bps / 10000)
                    trades += 1
                    positions.pop(sym)

        # entries
        if len(positions) < max_positions:
            active = [s for s in row_sig.index if row_sig.get(s, 0) == 1 and s not in positions]
            slots = max_positions - len(positions)
            for sym in active[:slots]:
                px = row_prices.get(sym, 0)
                if pd.isna(px) or px <= 0:
                    continue
                size_usd = cash * max_position_pct * scale
                if size_usd <= 0 or cash < size_usd:
                    continue
                fill = px * (1 + (slippage_bps + cost_bps) / 10000)
                positions[sym] = {"shares": size_usd / fill}
                cash -= size_usd
                if len(positions) >= max_positions:
                    break

    eq_arr = np.array(equity_curve, dtype=float)
    ret_arr = np.diff(eq_arr) / eq_arr[:-1] if len(eq_arr) > 1 else np.array([])
    std_r = float(np.std(ret_arr)) if ret_arr.size else 0.0
    sharpe = float(np.nan_to_num(np.mean(ret_arr) / (std_r + 1e-12) * np.sqrt(365), nan=0.0, posinf=0.0, neginf=0.0))
    total_ret = float((float(eq_arr[-1]) / initial - 1) * 100) if len(eq_arr) else 0.0
    peak = np.maximum.accumulate(eq_arr)
    max_dd = float(np.nan_to_num(np.max((peak - eq_arr) / np.where(np.abs(peak) < 1e-12, np.nan, peak)) * 100, nan=100.0, posinf=0.0, neginf=0.0)) if len(eq_arr) else 0.0
    exposure_ratio = float(np.nanmean(exposure_history)) if exposure_history else 0.0
    norm = np.sqrt(max(exposure_ratio, 1e-12))
    eff_sharpe = 0.0 if std_r == 0 else float(np.nan_to_num(np.mean(ret_arr) / (std_r + 1e-12) * np.sqrt(365) * norm, nan=0.0, posinf=0.0, neginf=0.0))

    res = {
        "final_equity": round(float(eq_arr[-1]), 2) if len(eq_arr) else initial,
        "return_pct": round(total_ret, 2),
        "sharpe": round(sharpe, 2),
        "effective_sharpe": round(eff_sharpe, 2),
        "exposure_ratio": round(exposure_ratio, 3),
        "max_dd_pct": round(max_dd, 2),
        "trades": trades,
    }

    # DSR / PSR (Bailey & López de Prado 2014) - always populate
    use_trials = n_trials if n_trials is not None else 1
    sk = float(np.nan_to_num(pd.Series(ret_arr).skew(), nan=0.0))
    ku = float(np.nan_to_num(pd.Series(ret_arr).kurtosis(), nan=0.0)) + 3.0
    T = max(2, len(ret_arr))
    # DSR / PSR (Bailey & López de Prado 2014). Both helpers are NaN/zero
    # guarded internally (np.nan_to_num + sqrt(max(1e-12,...))), so they cannot
    # raise on degenerate inputs -> no try/except, no misleading 0.5 fallback.
    dsr = deflated_sharpe_ratio(sharpe, int(use_trials), T, skew=sk, kurt=ku)
    psr = probabilistic_sharpe_ratio(sharpe, benchmark_sr=0.0, T=T, skew=sk, kurt=ku)
    res["dsr"] = round(dsr, 3)
    res["psr"] = round(psr, 3)
    res["n_trials"] = int(use_trials)
    if fair_compare_path is not None:
        fair_compare_path = Path(fair_compare_path)
        fair_compare_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not fair_compare_path.exists() or fair_compare_path.stat().st_size == 0
        with fair_compare_path.open("a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["rule_name", "trades", "sharpe", "max_dd_pct", "exposure_ratio", "effective_sharpe", "dsr", "psr", "n_trials"])
            w.writerow([fair_compare_rule, trades, res["sharpe"], res["max_dd_pct"], res["exposure_ratio"], res["effective_sharpe"], res.get("dsr", ""), res.get("psr", ""), res.get("n_trials", "")])
    return res


def load_screened_universe(min_bars: int = 60, start_date: str = '2025-01-01', end_date: str = '2026-07-12', as_of: str = None):
    """Load screened altcoin data into {stem: df} dict. Used by both backtests and live.

    as_of: optional 'YYYY-MM-DD' (or filename-parseable) point-in-time cutoff.
      When given, uses the most recent screen CSV *dated on or before* as_of
      instead of the latest file. This removes survivorship bias: a backtest
      slice uses the universe that actually existed then, not today's survivors.
      Default (None) = latest screen = live behavior (unchanged).
    """
    from pathlib import Path
    import pandas as pd
    import re
    ROOT = Path('data')
    OUT = Path('backtest_output')
    screen_files = sorted(OUT.glob('screen_liqu_idio_*.csv'))
    if not screen_files:
        raise FileNotFoundError("no screen_liqu_idio_*.csv found")
    if as_of is not None:
        # parse 'YYYYMMDD' from each filename; keep those <= as_of date
        as_of_d = pd.Timestamp(as_of).date()
        def _fname_date(p):
            m = re.search(r'(\d{8})', p.name)
            return pd.Timestamp(m.group(1)).date() if m else None
        eligible = [p for p in screen_files if (lambda d: d is not None and d <= as_of_d)(_fname_date(p))]
        screen_path = eligible[-1] if eligible else screen_files[0]
    else:
        screen_path = screen_files[-1]
    screen = pd.read_csv(screen_path)
    screen = screen[screen.tier.isin(['large', 'mid', 'tail'])]
    coin_data = {}
    seen = set()
    for _, row in screen.iterrows():
        stem = str(row['stem']).strip().upper()
        if stem in seen:
            continue
        seen.add(stem)
        p = ROOT / f'{stem}_1d_max.csv'
        if not p.exists():
            continue
        df = pd.read_csv(p, parse_dates=['ts']).dropna(subset=['close', 'high', 'low', 'volume'])
        df = df.sort_values('ts').reset_index(drop=True)
        if len(df[(df['ts'] >= start_date) & (df['ts'] <= end_date)]) < min_bars:
            continue
        coin_data[stem] = df
    return coin_data


# --- Regime quality diagnostics ---
def analyze_regime_quality(
    market_close: pd.Series,
    regime_fn: callable = None,
    adx_threshold: float = 22.0,
    vol_threshold: float = 0.22,
    er_threshold: float = 0.35,
    min_regime_bars: int = 3,
):
    """
    Analyze a regime series for distribution, persistence, and feature separation.
    Returns a dict with stats useful for tuning and understanding behavior.
    """
    if regime_fn is None:
        regime_fn = lambda s, i: compute_regime(
            s, i,
            adx_threshold=adx_threshold,
            vol_threshold=vol_threshold,
            er_threshold=er_threshold,
            min_regime_bars=min_regime_bars
        )

    regimes = []
    adx_vals = []
    vol_vals = []
    er_vals = []

    for i in range(len(market_close)):
        reg = regime_fn(market_close, i)
        regimes.append(reg)

        if i >= 25:
            c = market_close.iloc[:i+1]
            h = c.rolling(2).max()
            l = c.rolling(2).min()
            a = float(adx(h, l, c).iloc[i])
            v = float(realized_vol(c).iloc[i])
            e = float(kaufman_efficiency_ratio(c).iloc[i])

            adx_vals.append(a if not pd.isna(a) else np.nan)
            vol_vals.append(v if not pd.isna(v) else np.nan)
            er_vals.append(e if not pd.isna(e) else np.nan)

    reg_series = pd.Series(regimes, index=market_close.index)
    dist = reg_series.value_counts(normalize=True).to_dict()

    # Transitions
    changes = (reg_series != reg_series.shift(1)).sum() - 1
    total_days = len(reg_series)

    # Durations
    durations = []
    current = 1
    for i in range(1, len(reg_series)):
        if reg_series.iloc[i] == reg_series.iloc[i-1]:
            current += 1
        else:
            durations.append(current)
            current = 1
    durations.append(current)

    dur_series = pd.Series(durations)
    avg_duration = dur_series.mean()
    median_duration = dur_series.median()
    max_duration = dur_series.max()

    # Feature separation
    feat_df = pd.DataFrame({
        'regime': regimes[25:],
        'adx': adx_vals,
        'vol': vol_vals,
        'er': er_vals
    }).dropna()

    feat_by_regime = feat_df.groupby('regime')[['adx', 'vol', 'er']].agg(['mean', 'median']).round(4)

    stats = {
        'distribution': {k: round(v, 4) for k, v in dist.items()},
        'transitions': {
            'total_changes': int(changes),
            'change_rate_pct': round(100 * changes / total_days, 2),
            'total_days': int(total_days)
        },
        'durations': {
            'avg': round(avg_duration, 1),
            'median': round(median_duration, 1),
            'max': int(max_duration),
            'num_switches': len(durations) - 1
        },
        'features_by_regime': feat_by_regime.to_dict(),
        'sample_recent_regimes': reg_series.iloc[-10:].tolist()
    }
    return stats


def print_regime_stats(stats: dict):
    """Pretty print the diagnostic output."""
    print("=== Regime Quality Report ===")
    print(f"Distribution: {stats['distribution']}")
    print(f"Transitions: {stats['transitions']['total_changes']} changes "
          f"({stats['transitions']['change_rate_pct']}% of days)")
    print(f"Durations (bars): avg={stats['durations']['avg']}, "
          f"median={stats['durations']['median']}, max={stats['durations']['max']}")
    print("Features by regime:")
    for regime, vals in stats.get('features_by_regime', {}).items():
        print(f"  {regime}: {vals}")
    print(f"Recent regimes (last 10): {stats['sample_recent_regimes']}")


# --- Simple HMM regime (optional, literature-backed) ---
try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False


def fit_hmm_regime(market_close: pd.Series, n_states: int = 2, random_state: int = 42):
    """Fit a GaussianHMM on returns. Returns the fitted model.
    Typical use: low-vol vs high-vol or trend vs chop states.
    """
    if not HAS_HMMLEARN:
        raise ImportError("hmmlearn not installed. pip install hmmlearn")
    rets = market_close.pct_change().dropna().values.reshape(-1, 1)
    model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=100, random_state=random_state)
    model.fit(rets)
    return model


def hmm_regime_fn(market_close: pd.Series, day_index: int, hmm_model):
    """Predict regime using a pre-fitted HMM model.
    Maps states heuristically: higher mean return or lower vol state -> 'trend'.
    """
    if day_index < 25 or hmm_model is None:
        return "trend"
    recent_rets = market_close.pct_change().iloc[max(0, day_index-100):day_index+1].dropna().values.reshape(-1, 1)
    if len(recent_rets) < 5:
        return "trend"
    states = hmm_model.predict(recent_rets)
    last_state = states[-1]
    # Heuristic mapping
    if hasattr(hmm_model, 'means_'):
        means = hmm_model.means_.flatten()
        # Assume the state with larger |mean| or we can inspect variances
        if means[last_state] > 0.0005:  # rough positive drift bias
            return "trend"
    # Fallback: alternate labeling
    
    # Hurst method (new literature-backed option)
    if method == "hurst":
        return hurst_regime(close_market, day_index, window=hurst_window, threshold=hurst_threshold)

    # Hybrid: rule-based + Hurst confirmation (only declare trend if both agree)
    if method == "hybrid":
        rule_reg = "trend" if (a >= adx_threshold and e >= er_threshold and v <= vol_threshold) else "chop"
        h_reg = hurst_regime(close_market, day_index, window=hurst_window, threshold=hurst_threshold)
        if rule_reg == "trend" and h_reg == "trend":
            return "trend"
        return "chop"

    return "trend" if last_state == 0 else "chop"


def compute_hybrid_regime(market_close: pd.Series, day_index: int,
                          rule_params: dict = None, hmm_model=None, weight_rule: float = 0.7):
    """Hybrid: rule-based + HMM. Currently rule-weighted for simplicity."""
    rule_reg = compute_regime(market_close, day_index, **(rule_params or {}))
    if hmm_model is None:
        return rule_reg
    try:
        hmm_reg = hmm_regime_fn(market_close, day_index, hmm_model)
        # Simple agreement bias
        if rule_reg == hmm_reg:
            return rule_reg
        return rule_reg if weight_rule > 0.5 else hmm_reg
    except Exception:
        return rule_reg




# --- Deflated Sharpe Ratio (Bailey & López de Prado 2014) ---
# Corrects for multiple testing (selection bias) + non-normality + sample length.
# DSR is the probability that the observed SR is significant after adjustments.
# Usage: track total number of independent trials (rules/variants tested).
# For best accuracy, estimate skew and kurtosis from the strategy returns.


def deflated_sharpe_ratio(observed_sr: float, n_trials: int, T: int, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Robust DSR (Bailey & López de Prado 2014)."""
    observed_sr = float(np.nan_to_num(observed_sr, nan=0.0))
    n_trials = max(1, int(n_trials))
    T = max(2, int(T))
    skew = float(np.nan_to_num(skew, nan=0.0))
    kurt = float(np.nan_to_num(kurt, 3.0))
    try:
        gamma = 0.5772156649
        if n_trials > 1:
            expected_max = ((1 - gamma) * norm.ppf(1 - 1.0 / n_trials) + gamma * norm.ppf(1 - 1.0 / (n_trials * np.e))) / np.sqrt(2 * np.log(np.log(n_trials) + 1))
        else:
            expected_max = 0.0
        denom = np.sqrt( max(1e-12, (1 - skew * observed_sr + (kurt - 1) * observed_sr**2 / 4 ) / (T - 1)) )
        z = (observed_sr - expected_max) / denom
        dsr = norm.cdf(z)
        return float(np.clip(dsr, 0.0, 1.0))
    except Exception:
        # Conservative fallback
        return 0.5 if observed_sr > 0 else 0.0
def probabilistic_sharpe_ratio(observed_sr: float, benchmark_sr: float = 0.0, T: int = 252, skew: float = 0.0, kurt: float = 3.0) -> float:
    """Probabilistic Sharpe Ratio (PSR) - probability SR > benchmark after non-normality correction."""
    denom = np.sqrt( (1 - skew * observed_sr + ((kurt - 1) * observed_sr**2) / 4 ) / (T - 1) ) if T > 1 else 1.0
    if denom == 0:
        denom = 1e-12
    z = (observed_sr - benchmark_sr) / denom
    return float(norm.cdf(z))



# --- Stepwise SPA helpers (Hsu et al. 2010 / adapted from spa_hsu_focused.py)
# Studentized performance + stepwise critical value for identifying superior rules
# while controlling for data snooping.

def studentized_performance(strat_returns, benchmark_returns):
    """Studentized difference (Hansen/Hsu style)."""
    diff = np.array(strat_returns, dtype=float) - np.array(benchmark_returns, dtype=float)
    n = len(diff)
    if n < 3:
        return np.nan, np.nan, np.nan
    mu = float(diff.mean())
    gamma_0 = float(np.var(diff, ddof=1))
    gamma_sum = sum(float(np.cov(diff[:-k], diff[k:])[0, 1]) for k in range(1, min(5, n)))
    v = max((gamma_0 + 2 * gamma_sum) / n, 0)
    T = mu / np.sqrt(v) if v > 1e-18 else np.nan
    return float(T), mu, v


def stepwise_spa_test(all_strat_returns, benchmark_returns, alpha=0.10):
    """Stepwise SPA test (Hsu et al. 2010).
    Returns dict with significant flag, survivors, T stats, etc.
    """
    T_vals = []
    mu_vals = []
    v_vals = []
    for strat in all_strat_returns:
        T, mu, v = studentized_performance(strat, benchmark_returns)
        T_vals.append(T)
        mu_vals.append(mu)
        v_vals.append(v)

    g = int(np.sum(~np.isnan(T_vals)))
    if g == 0:
        return {'significant': False, 'surviving_indices': [], 'T_max': np.nan, 'T_crit': np.nan, 'n_tested': 0}

    alpha_g = 1 - (1 - alpha) ** (1 / max(1, g))
    t_crit = float(norm.ppf(1 - alpha_g))
    T_max = float(np.nanmax(T_vals)) if np.any(~np.isnan(T_vals)) else np.nan
    significant = bool(T_max > t_crit) if not np.isnan(T_max) else False
    surviving = [i for i, T in enumerate(T_vals) if not np.isnan(T) and T > t_crit]

    stats = {}
    for i in range(len(T_vals)):
        stats[str(i)] = {
            'T': T_vals[i] if not np.isnan(T_vals[i]) else None,
            'mu_diff': float(mu_vals[i]) if not np.isnan(mu_vals[i]) else None,
            'significant': bool(T_vals[i] > t_crit) if not np.isnan(T_vals[i]) else False,
        }

    return {
        'significant': significant,
        'surviving_indices': surviving,
        'T_max': T_max,
        'T_crit': t_crit,
        'alpha': alpha,
        'n_tested': g,
        'stats': stats,
    }
