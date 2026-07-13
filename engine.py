"Canonical trading engine: pure functions for rules, signals, and portfolio mechanics."
import numpy as np
import pandas as pd
from pathlib import Path
import csv


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
def donchian_signal(high: pd.Series, low: pd.Series, close: pd.Series, lookback: int) -> pd.Series:
    """Return long signal series: 1 when close > lookback-high shifted by 1, else 0."""
    don_high = high.rolling(lookback).max().shift(1)
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
    """Williams %R: entry when > -80 and rising, exit when < -20 and falling."""
    high = pd.Series(df["high"].values, index=df.index)
    low = pd.Series(df["low"].values, index=df.index)
    close = pd.Series(df["close"].values, index=df.index)
    highest = high.rolling(period, min_periods=1).max()
    lowest = low.rolling(period, min_periods=1).min()
    wr = -100 * (highest - close) / (highest - lowest + 1e-12)
    entry = ((wr > -80) & (wr.diff() > 0)).astype(int)
    exit_sig = ((wr < -20) & (wr.diff() < 0)).astype(int)
    return entry, exit_sig


def get_regime_signals(rule_name: str, df: pd.DataFrame):
    rule = rule_name.lower()
    if rule in ("cci", "cci20"):
        return cci_signals(df)
    if rule in ("rei",):
        return rei_signals(df)
    if rule in ("williams", "williams_r", "wr"):
        return williams_r_signals(df)
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


def realized_vol(close: pd.Series, window: int = 20) -> pd.Series:
    """Simple realized volatility (std of returns)."""
    return close.pct_change().rolling(window, min_periods=5).std()


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
    method: str = "rule",   # "rule" | "hmm" | "hybrid"
    hmm_model=None,
) -> str:
    """
    Improved market regime detector.

    Literature grounding (key sources):
    - ADX for trend strength (Wilder + many: Darwinex, Alpha Architect summaries, LuxAlgo, Reddit algotrading).
    - Realized vol / ATR% for high-vol vs calm (QuantStart HMM papers, volatility filter literature).
    - Kaufman Efficiency Ratio (ER) as a clean directional vs chop measure (repeatedly recommended in practitioner regime filters).
    - Hysteresis / minimum duration to avoid whipsaw (practical consensus across sources).
    - HMM option: GaussianHMM on returns (classic for latent low/high-vol or trend/chop regimes; see QuantStart QSTrader example and multiple HMM regime papers).
    - Theoretical context: Hamilton (1989) Markov switching; Zakamulin & Giner (semi-Markov) showing regime affects optimal trend rules and duration dependence.

    Logic (rule-based default):
      - "trend" if (ADX >= adx_threshold) AND (ER >= er_threshold) AND (vol <= vol_threshold)
      - Else "chop"
      - High vol can be treated as chop for most momentum rules (or a third "crisis" state in future).

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
):
    """Live-friendly wrapper around the improved regime logic using a mean-close market proxy."""
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
    return compute_regime(
        market, idx,
        adx_threshold=adx_threshold,
        vol_threshold=vol_threshold,
        er_threshold=er_threshold,
        method=method,
        hmm_model=hmm_model,
    )



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
    if fair_compare_path is not None:
        fair_compare_path = Path(fair_compare_path)
        fair_compare_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not fair_compare_path.exists() or fair_compare_path.stat().st_size == 0
        with fair_compare_path.open("a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["rule_name", "trades", "sharpe", "max_dd_pct", "exposure_ratio", "effective_sharpe"])
            w.writerow([fair_compare_rule, trades, res["sharpe"], res["max_dd_pct"], res["exposure_ratio"], res["effective_sharpe"]])
    return res


def load_screened_universe(min_bars: int = 60, start_date: str = '2025-01-01', end_date: str = '2026-07-12'):
    """Load screened altcoin data into {stem: df} dict. Used by both backtests and live."""
    from pathlib import Path
    import pandas as pd
    ROOT = Path('data')
    OUT = Path('backtest_output')
    screen = pd.read_csv(sorted(OUT.glob('screen_liqu_idio_*.csv'))[-1])
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

