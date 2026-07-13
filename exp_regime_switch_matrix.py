"""Multi-rule regime-switching runner: tests all trend/chop combinations."""
import sys
import itertools
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


def rei_signals(df):
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    vol = df["volume"].values
    tp = (high + low + close) / 3.0
    sma_tp = pd.Series(tp, index=df.index).rolling(20, min_periods=1).mean()
    mad = pd.Series(tp, index=df.index).rolling(20, min_periods=1).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    cci = (tp - sma_tp) / (0.015 * mad + 1e-12)
    e = ((cci > -100) & (cci.diff() > 0)).astype(int)
    x = ((cci < 100) & (cci.diff() < 0)).astype(int)
    return e, x


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
sig_rei_entry = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)
sig_rei_exit = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)

for stem, df in coin_data.items():
    close = df["close"].values
    e_cci, x_cci = cci_signals(df)
    e_wr, x_wr = williams_r_signals(df)
    e_rei, x_rei = rei_signals(df)
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
            sig_rei_entry.loc[ts, stem] = int(e_rei.iloc[i]) if pd.notna(e_rei.iloc[i]) else 0
            sig_rei_exit.loc[ts, stem] = int(x_rei.iloc[i]) if pd.notna(x_rei.iloc[i]) else 0

price_df = price_df.sort_index()
sig_d40 = sig_d40.sort_index()
sig_cci_entry = sig_cci_entry.sort_index()
sig_cci_exit = sig_cci_exit.sort_index()
sig_wr_entry = sig_wr_entry.sort_index()
sig_wr_exit = sig_wr_exit.sort_index()
sig_rei_entry = sig_rei_entry.sort_index()
sig_rei_exit = sig_rei_exit.sort_index()

market_close = price_df.mean(axis=1)
all_dates = price_df.index.tolist()

print("Baseline d40 (full)", flush=True)
res_d40 = simulate_portfolio(price_df, sig_d40, initial=INITIAL, max_positions=5)
print("baseline_d40: ret=" + str(round(res_d40["return_pct"], 1)) + "% sharpe=" + str(round(res_d40["sharpe"], 2)) + " dd=" + str(round(res_d40["max_dd_pct"], 1)) + "% trades=" + str(res_d40["trades"]), flush=True)

# Single-rule baselines
single_rules = {
    "CCI": (sig_cci_entry, sig_cci_exit),
    "Williams %R": (sig_wr_entry, sig_wr_exit),
    "REI": (sig_rei_entry, sig_rei_exit),
}
for name, (entry_df, exit_df) in single_rules.items():
    res = simulate_portfolio(price_df, entry_df, initial=INITIAL, max_positions=5, exit_signal_df=exit_df)
    print(f"{name} only: ret={res['return_pct']:.1f}% sharpe={res['sharpe']:.2f} dd={res['max_dd_pct']:.1f}% trades={res['trades']}", flush=True)

# Multi-rule regime combinations
rules = {
    "CCI": (sig_cci_entry, sig_cci_exit),
    "Williams %R": (sig_wr_entry, sig_wr_exit),
    "REI": (sig_rei_entry, sig_rei_exit),
}
trend_names = ["CCI", "REI"]
chop_names = ["Williams %R", "REI"]
combinations = list(itertools.product(trend_names, chop_names))

results = []
print("\n=== Multi-rule regime combinations ===", flush=True)
for trend_name, chop_name in combinations:
    t_entry, t_exit = rules[trend_name]
    c_entry, c_exit = rules[chop_name]
    regime_series = pd.Series([default_regime(market_close, i) for i in range(len(all_dates))], index=all_dates)
    combined_entry = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)
    combined_exit = pd.DataFrame(index=all_dates, columns=list(coin_data.keys()), dtype=int)
    for day in all_dates:
        if regime_series.loc[day] == "trend":
            combined_entry.loc[day] = t_entry.loc[day]
            combined_exit.loc[day] = t_exit.loc[day]
        else:
            combined_entry.loc[day] = c_entry.loc[day]
            combined_exit.loc[day] = c_exit.loc[day]
    trend_days = (regime_series == "trend").sum()
    chop_days = (regime_series == "chop").sum()
    res = simulate_portfolio(price_df, combined_entry, initial=INITIAL, max_positions=5, exit_signal_df=combined_exit)
    results.append({
        "trend": trend_name,
        "chop": chop_name,
        "return_pct": res["return_pct"],
        "sharpe": res["sharpe"],
        "max_dd_pct": res["max_dd_pct"],
        "trades": res["trades"],
        "trend_days": trend_days,
        "chop_days": chop_days,
    })
    print(f"{trend_name} trend / {chop_name} chop: ret={res['return_pct']:.1f}% sharpe={res['sharpe']:.2f} dd={res['max_dd_pct']:.1f}% trades={res['trades']} (trend={trend_days}, chop={chop_days})", flush=True)

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("sharpe", ascending=False)
print("\n=== Ranked by Sharpe ===", flush=True)
print(results_df.to_string(index=False), flush=True)
results_df.to_csv(OUT / "regime_switch_combinations.csv", index=False)
print("\nSaved to backtest_output/regime_switch_combinations.csv", flush=True)
