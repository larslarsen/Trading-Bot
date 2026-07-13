"""Simple regime-switched backtest using the improved detector.

Trend: CCI signals
Chop: Williams %R signals
Baseline: Donchian 40

Uses the tuned regime (ADX 22 / vol 0.22 / ER 0.35).
"""
import pandas as pd
import numpy as np
from engine import (
    load_screened_universe, compute_regime, cci_signals, williams_r_signals,
    simulate_portfolio
)

def main():
    print("Loading data...")
    coin_data = load_screened_universe(min_bars=120)
    all_dates = pd.Index([])
    for df in coin_data.values():
        all_dates = all_dates.union(df['ts'])
    all_dates = pd.DatetimeIndex(sorted(all_dates))

    price_df = pd.DataFrame(index=all_dates)
    high_df = pd.DataFrame(index=all_dates)
    low_df = pd.DataFrame(index=all_dates)
    close_df = pd.DataFrame(index=all_dates)

    for stem, df in coin_data.items():
        df = df.set_index('ts').sort_index()
        price_df[stem] = df['close'].reindex(all_dates)
        high_df[stem] = df['high'].reindex(all_dates)
        low_df[stem] = df['low'].reindex(all_dates)
        close_df[stem] = df['close'].reindex(all_dates)

    price_df = price_df.dropna(axis=0, how='all')
    market_close = close_df.mean(axis=1).ffill().bfill()

    regimes = [compute_regime(market_close, i, adx_threshold=22, vol_threshold=0.22, er_threshold=0.35) 
               for i in range(len(market_close))]
    regime_series = pd.Series(regimes, index=market_close.index)

    print(f"Regime split: {(regime_series == 'trend').mean()*100:.1f}% trend / {(regime_series == 'chop').mean()*100:.1f}% chop")

    def make_sig(rule_fn):
        sigs = {}
        for col in price_df.columns:
            sub = pd.DataFrame({'close': price_df[col], 'high': high_df[col], 'low': low_df[col]}).dropna()
            if len(sub) < 30:
                sigs[col] = pd.Series(0, index=price_df.index)
                continue
            entry, _ = rule_fn(sub)
            sigs[col] = entry.reindex(price_df.index).fillna(0).astype(int)
        return pd.DataFrame(sigs, index=price_df.index)

    def d40(df):
        dh = df['high'].rolling(40).max().shift(1)
        return (df['close'] > dh).astype(int), pd.Series(0, index=df.index)

    print("Computing signals...")
    base_sig = make_sig(d40)
    cci_sig = make_sig(cci_signals)
    will_sig = make_sig(williams_r_signals)

    def reg_fn(cm, i):
        return regime_series.iloc[min(i, len(regime_series)-1)]

    print("Simulating baseline...")
    b = simulate_portfolio(price_df, base_sig, initial=1000, max_positions=5, max_position_pct=0.2, cost_bps=8, slippage_bps=5)

    print("Simulating regime (CCI trend / Williams chop)...")
    r = simulate_portfolio(price_df, cci_sig, initial=1000, max_positions=5, max_position_pct=0.2, cost_bps=8, slippage_bps=5,
                           regime_fn=reg_fn, regime_rule_map={'trend': cci_sig, 'chop': will_sig})

    print("\n=== RESULTS ===")
    print(f"Baseline Donchian40: equity={b['final_equity']}, return={b['return_pct']}%, sharpe={b['sharpe']}, eff_sharpe={b['effective_sharpe']}, max_dd={b['max_dd_pct']}%, trades={b['trades']}, exposure={b['exposure_ratio']}")
    print(f"Regime switched   : equity={r['final_equity']}, return={r['return_pct']}%, sharpe={r['sharpe']}, eff_sharpe={r['effective_sharpe']}, max_dd={r['max_dd_pct']}%, trades={r['trades']}, exposure={r['exposure_ratio']}")

if __name__ == "__main__":
    main()
