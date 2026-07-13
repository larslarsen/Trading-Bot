"""Regime-switching experiment: CCI (trend) | Williams %R (chop)."""
import sys
sys.modules.pop('exp_regime_switch', None)

import numpy as np
import pandas as pd
from pathlib import Path

from engine import simulate_portfolio, donchian_signal, default_regime

ROOT = Path("data")
OUT = Path("backtest_output")
INITIAL = 1000.0
screen = pd.read_csv(sorted(OUT.glob("screen_liqu_idio_*.csv"))[-1])
screen = screen[screen.tier.isin(["large", "mid", "tail"])]


def load_coins():
    coin_data = {}
    seen = set()
    for _, row in screen.iterrows():
        stem = str(row["stem"]).strip().upper()
        if stem in seen:
            continue
        seen.add(stem)
        p = ROOT / f"{stem}_1d_max.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p, parse_dates=["ts"]).dropna(subset=["close", "high", "low", "volume"])
        df = df.sort_values("ts").reset_index(drop=True)
        if len(df[(df["ts"] >= "2025-01-01") & (df["ts"] <= "2026-07-12")]) < 60:
            continue
        coin_data[stem] = df
    return coin_data


def cci_signals(df):
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    tp = (high + low + close) / 3.0
    sma_tp = pd.Series(tp, index=df.index).rolling(20, min_periods=1).mean()
    mad = pd.Series(tp, index=df.index).rolling(20, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad + 1e-12)
    entry = ((cci > 0) & (cci.diff() > 0)).astype(int)
    exit_sig = ((cci < 0) & (cci.diff() < 0)).astype(int)
    return entry, exit_sig


def williams_r_signals(df):
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    highest = pd.Series(high, index=df.index).rolling(14, min_periods=1).max()
    lowest = pd.Series(low, index=df.index).rolling(14, min_periods=1).min()
    wr = -100 * (highest - pd.Series(close, index=df.index)) / (highest - lowest + 1e-12)
    entry = ((wr > -80) & (wr.diff() > 0)).astype(int)
    exit_sig = ((wr < -20) & (wr.diff() < 0)).astype(int)
    return entry, exit_sig


print("Loading...", flush=True)
coin_data = load_coins()
all_dates = sorted(set(d for c in coin_data.values() for d in c.loc[(c["ts"] >= "2025-01-01") & (c["ts"] <= "2026-07-12"), "ts"]))
print("Coins:", len(coin_data), "Dates:", len(all_dates), flush=True)

price_df = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=float)
sig_d40 = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)
sig_cci_entry = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)
sig_cci_exit = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)
sig_wr_entry = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)
sig_wr_exit = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)

for stem, df in coin_data.items():
    close = df["close"].values
    e_cci, x_cci = cci_signals(df)
    e_wr, x_wr = williams_r_signals(df)
    mask = (df["ts"] >= "2025-01-01") & (df["ts"] <= "2026-07-12")
    s = donchian_signal(df["high"], df["low"], df["close"], 40)
    for i in range(len(df)):
        if mask.iloc[i]:
            ts = df["ts"].iloc[i]
            price_df.loc[ts, stem] = close[i]
            sig_d40.loc[ts, stem] = int(s.iloc[i]) if pd.notna(s.iloc[i]) else 0
            sig_cci_entry.loc[ts, stem] = int(e_cci.iloc[i]) if pd.notna(e_cci.iloc[i]) else 0
            sig_cci_exit.loc[ts, stem] = int(x_cci.iloc[i]) if pd.notna(x_cci.iloc[i]) else 0
            sig_wr_entry.loc[ts, stem] = int(e_wr.iloc[i]) if pd.notna(e_wr.iloc[i]) else 0
            sig_wr_exit.loc[ts, stem] = int(x_wr.iloc[i]) if pd.notna(x_wr.iloc[i]) else 0

price_df = price_df.sort_index()
sig_d40 = sig_d40.sort_index()
sig_cci_entry = sig_cci_entry.sort_index()
sig_cci_exit = sig_cci_exit.sort_index()
sig_wr_entry = sig_wr_entry.sort_index()
sig_wr_exit = sig_wr_exit.sort_index()

print("Baseline d40 (full)", flush=True)
res_d40 = simulate_portfolio(price_df, sig_d40, initial=INITIAL, max_positions=5)
print("baseline_d40: ret=" + str(round(res_d40["return_pct"], 1)) + "% sharpe=" + str(round(res_d40["sharpe"], 2)) + " dd=" + str(round(res_d40["max_dd_pct"], 1)) + "% trades=" + str(res_d40["trades"]), flush=True)

print("CCI only (full)", flush=True)
res_cci = simulate_portfolio(price_df, sig_cci_entry, initial=INITIAL, max_positions=5, exit_signal_df=sig_cci_exit)
print("cci_only: ret=" + str(round(res_cci["return_pct"], 1)) + "% sharpe=" + str(round(res_cci["sharpe"], 2)) + " dd=" + str(round(res_cci["max_dd_pct"], 1)) + "% trades=" + str(res_cci["trades"]), flush=True)

print("Williams %R only (full)", flush=True)
res_wr = simulate_portfolio(price_df, sig_wr_entry, initial=INITIAL, max_positions=5, exit_signal_df=sig_wr_exit)
print("wr_only: ret=" + str(round(res_wr["return_pct"], 1)) + "% sharpe=" + str(round(res_wr["sharpe"], 2)) + " dd=" + str(round(res_wr["max_dd_pct"], 1)) + "% trades=" + str(res_wr["trades"]), flush=True)

print("Regime switch CCI↔Williams %R (full, 5-position cap)", flush=True)
market_close = price_df.mean(axis=1)
regime_series = pd.Series([default_regime(market_close, i) for i in range(len(all_dates))], index=all_dates)

# Build regime-aware entry/exit frames
combined_entry = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)
combined_exit = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)
for day in all_dates:
    regime = regime_series.loc[day]
    if regime == "trend":
        combined_entry.loc[day] = sig_cci_entry.loc[day]
        combined_exit.loc[day] = sig_cci_exit.loc[day]
    else:
        combined_entry.loc[day] = sig_wr_entry.loc[day]
        combined_exit.loc[day] = sig_wr_exit.loc[day]

res_switch = simulate_portfolio(price_df, combined_entry, initial=INITIAL, max_positions=5,
                                regime_fn=None, regime_rule_map=None,
                                exit_signal_df=combined_exit)
print("switch: ret=" + str(round(res_switch["return_pct"], 1)) + "% sharpe=" + str(round(res_switch["sharpe"], 2)) + " dd=" + str(round(res_switch["max_dd_pct"], 1)) + "% trades=" + str(res_switch["trades"]), flush=True)

print("Regime breakdown:", flush=True)
print("trend days:", (regime_series == "trend").sum(), "chop days:", (regime_series == "chop").sum(), flush=True)
