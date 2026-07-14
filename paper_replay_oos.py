"""
Fixed strict day-by-day OOS paper trader replay using the SHARED PortfolioEngine.

This drives portfolio_engine.PortfolioEngine (the exact same cash/MTM/risk logic
the live trader uses) instead of a separate SimPosition implementation. The two
code paths now share one source of truth, so a replay verifies the live math.

- Causal regime and signals using only data up to the day.
- Explicit cash tracking inside the engine: open deducts cash, close credits cash,
  equity = cash + position value.
- Applies same DD halt, position caps, 20% sizing as live.
- Baseline = pure Donchian40; regime = improved regime + cci/williams or rei/williams.
"""

import pandas as pd
import numpy as np
from portfolio_engine import PortfolioEngine, EngineConfig

from engine import load_screened_universe, improved_compute_live_regime, get_regime_signals, atr_trailing_exit

COST_BPS = 8 / 10000.0
SLIP_BPS = 5 / 10000.0
MAX_POS = 5
POS_PCT = 0.20


def load_data():
    coin_data = load_screened_universe(min_bars=120)
    data = {stem: df.set_index('ts').sort_index() for stem, df in coin_data.items()}
    dates = pd.DatetimeIndex(sorted(set(ts for df in data.values() for ts in df.index)))
    return data, dates


def prefix_up_to(data, day):
    return {s: d.loc[:day] for s, d in data.items() if len(d.loc[:day]) > 0}


def d40_latest_entry(dfp):
    if len(dfp) < 40:
        return 0
    dh = dfp['high'].rolling(40).max().shift(1)
    return int(dfp['close'].iloc[-1] > dh.iloc[-1])


def latest_entry(rule, dfp):
    try:
        ent, _ = get_regime_signals(rule, dfp.reset_index())
        return int(ent.iloc[-1]) if len(ent) > 0 else 0
    except Exception:
        return 0


def compute_regime(prefix):
    try:
        return improved_compute_live_regime(prefix)
    except Exception:
        return 'trend'


def replay_oos(data, all_dates, start, end, mode='regime', trend_rule='cci', initial=10000.0, use_trailing=False):
    oos_dates = all_dates[(all_dates >= pd.Timestamp(start)) & (all_dates <= pd.Timestamp(end))]

    cfg = EngineConfig(
        initial_capital=initial,
        max_daily_loss_pct=0.03,
        max_drawdown_pct=0.20,
        max_positions=MAX_POS,
        max_position_pct=POS_PCT,
        min_equity_to_trade=100.0,
        flash_crash_bars=5,
        flash_crash_pct=0.50,
        extreme_move_pct=0.90,
        cost_bps=COST_BPS,
        slippage_bps=SLIP_BPS,
        enable_vol_target=False,
    )
    eng = PortfolioEngine(cfg)
    equity_hist = []
    pos_count_hist = []

    for day in oos_dates:
        pre = prefix_up_to(data, day)
        if not pre:
            equity_hist.append(eng.equity)
            pos_count_hist.append(len(eng.positions))
            continue

        # Decide active coins for THIS day using only prefix data
        if mode == 'baseline':
            active = [s for s in pre if d40_latest_entry(pre[s])]
        else:
            reg = compute_regime(pre)
            rule = trend_rule if reg == 'trend' else 'williams_r'
            active = [s for s in pre if latest_entry(rule, pre[s])]

        current_prices = {s: float(pre[s]['close'].iloc[-1]) for s in pre}

        # Daily bar start: reset daily pnl + flash window (counts in DAYS now)
        ref = next(iter(current_prices.values()), None)
        eng.start_daily_bar(ref)

        # Circuit breaker check (uses current equity)
        ok, reason = eng.check_circuit_breakers()
        if not ok:
            eng.flatten_all(current_prices)
            equity_hist.append(eng.equity)
            pos_count_hist.append(0)
            continue

        # Closes: not in active OR explicit exit -> close
        to_close = [s for s in list(eng.positions.keys()) if s not in active]

        # ATR trailing (regime-gated to trend)
        if use_trailing:
            reg = compute_regime(pre)
            if reg == 'trend':
                for sym in list(eng.positions.keys()):
                    if sym not in to_close:
                        sub = pre.get(sym)
                        if sub is not None and len(sub) >= 5:
                            try:
                                recent = sub.iloc[-max(14 * 2, 30):]
                                if atr_trailing_exit(recent, 14, 2.0).iloc[-1] == 1:
                                    to_close.append(sym)
                            except Exception:
                                pass

        for sym in set(to_close):
            px = current_prices.get(sym)
            if px is not None and px > 0:
                eng.close_position(sym, px)

        # Opens (rank-free here; ranking lives in the live trader / can be added)
        for sym in active:
            if sym in eng.positions or len(eng.positions) >= eng.config.max_positions:
                continue
            px = current_prices.get(sym)
            if px is None or px <= 0:
                continue
            ok, reason = eng.check_circuit_breakers()
            if not ok:
                break
            size_usd = eng.equity * POS_PCT  # equity-based sizing (matches live)
            eng.open_position(sym, px, size_usd)

        # Mark to market at end of day
        eq = eng.mark_to_market(current_prices)
        equity_hist.append(eq)
        pos_count_hist.append(len(eng.positions))

    if not equity_hist:
        return {'return_pct': 0, 'sharpe': 0, 'effective_sharpe': 0, 'max_dd_pct': 0, 'trades': 0,
                'avg_pos': 0, 'exposure': 0, 'days': 0}

    final = equity_hist[-1]
    rets = pd.Series(equity_hist).pct_change().dropna()
    sr = (rets.mean() / rets.std() * np.sqrt(252)) if len(rets) > 1 and rets.std() > 0 else 0.0
    ret_pct = (final / initial - 1) * 100
    avg_pos = float(np.mean(pos_count_hist)) if pos_count_hist else 0
    exp = avg_pos / MAX_POS if MAX_POS > 0 else 0
    eff_sr = sr * np.sqrt(max(exp, 1e-6))

    peak = initial
    max_dd = 0.0
    for e in equity_hist:
        peak = max(peak, e)
        dd = (peak - e) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    oos_dates_used = oos_dates[:len(equity_hist)]
    curve = [(str(d.date()), float(e)) for d, e in zip(oos_dates_used, equity_hist)]
    return {
        'return_pct': round(ret_pct, 1),
        'sharpe': round(sr, 2),
        'effective_sharpe': round(eff_sr, 2),
        'max_dd_pct': round(max_dd * 100, 1),
        'trades': len(eng.trades),
        'avg_pos': round(avg_pos, 1),
        'exposure': round(exp, 2),
        'days': len(equity_hist),
        'final_equity': round(equity_hist[-1], 2) if equity_hist else initial,
        'equity_curve': curve,
    }


def main():
    print("Loading screened data for fixed paper replay...")
    data, dates = load_data()
    print(f"Coins: {len(data)}, {dates[0].date()} to {dates[-1].date()}")

    lbl, st, en = "last_90d", "2026-04-14", "2026-07-12"
    print(f"\n=== Simulated Paper Trade Replay: {lbl} ({st} to {en}) ===\n")

    variants = [
        ("baseline_d40", "baseline", None, False),
        ("regime_cci_will", "regime", "cci", False),
        ("regime_rei_will", "regime", "rei", False),
        ("regime_rei_atr", "regime", "rei", True),
    ]

    results = {}
    for name, mode, trend, trail in variants:
        if mode == "baseline":
            res = replay_oos(data, dates, st, en, mode="baseline", use_trailing=False)
        else:
            res = replay_oos(data, dates, st, en, mode="regime", trend_rule=trend, use_trailing=trail)
        results[name] = res
        print(f"{name:16s} | Start: $10000 | End: ${res['final_equity']:.2f} | Total Return: {res['return_pct']}%")

    import pandas as pd
    df_list = []
    for name in results:
        curve = results[name].get("equity_curve", [])
        if curve:
            ddf = pd.DataFrame(curve, columns=["date", name]).set_index("date")
            df_list.append(ddf)
    if df_list:
        out_df = pd.concat(df_list, axis=1)
        out_df.to_csv("backtest_output/paper_replay_90d_equity.csv")
        print(f"\nFull equity curve saved to backtest_output/paper_replay_90d_equity.csv ({len(out_df)} days)")


if __name__ == "__main__":
    main()
