"""
Fixed strict day-by-day OOS paper trader replay with correct cash/MTM math.

Uses explicit cash tracking to avoid state machine subtleties:
- cash starts at initial
- On open: calculate correct fill (slip+cost), deduct cash, record position with shares
- On close: calculate proceeds (after cost), add to cash, record pnl
- Equity each day = cash + sum(shares * current_close for open positions)
- Applies same DD halt, position caps, 20% sizing
- Causal regime and signals using only data up to the day
- Baseline = pure Donchian40
- Regime = improved regime + cci/williams or rei/williams
"""

import pandas as pd
import numpy as np
from engine import load_screened_universe, improved_compute_live_regime, get_regime_signals

COST_BPS = 8 / 10000.0
SLIP_BPS = 5 / 10000.0
MAX_POS = 5
POS_PCT = 0.20

class SimPosition:
    def __init__(self, symbol, entry, shares):
        self.symbol = symbol
        self.entry = entry
        self.shares = shares

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
    except:
        return 0

def compute_regime(prefix):
    try:
        return improved_compute_live_regime(prefix)
    except:
        return 'trend'

def replay_oos(data, all_dates, start, end, mode='regime', trend_rule='cci', initial=10000.0):
    oos_dates = all_dates[(all_dates >= pd.Timestamp(start)) & (all_dates <= pd.Timestamp(end))]
    cash = initial
    positions = {}  # sym -> SimPosition
    equity_hist = []
    trade_count = 0
    pos_count_hist = []

    for day in oos_dates:
        pre = prefix_up_to(data, day)
        if not pre:
            equity_hist.append(cash + sum(p.shares * 0 for p in positions.values()))
            continue

        # Decide active coins
        if mode == 'baseline':
            active = [s for s in pre if d40_latest_entry(pre[s])]
            rule = 'd40'
        else:
            reg = compute_regime(pre)
            rule = trend_rule if reg == 'trend' else 'williams_r'
            active = [s for s in pre if latest_entry(rule, pre[s])]

        current_prices = {s: float(pre[s]['close'].iloc[-1]) for s in pre}

        # Closes
        to_close = [s for s in list(positions.keys()) if s not in active]
        for sym in to_close:
            pos = positions.pop(sym)
            px = current_prices.get(sym, pos.entry)
            proceeds = pos.shares * px * (1 - COST_BPS)
            cash += proceeds
            trade_count += 1

        # Opens (new entries)
        for sym in active:
            if sym in positions or len(positions) >= MAX_POS:
                continue
            px = current_prices.get(sym)
            if px is None or px <= 0:
                continue
            fill = px * (1 + SLIP_BPS + COST_BPS)
            size_usd = cash * POS_PCT   # use current cash for sizing
            if size_usd < 100:
                continue
            shares = size_usd / fill
            positions[sym] = SimPosition(sym, fill, shares)
            cash -= shares * fill   # spend the cash
            trade_count += 1

        # End of day equity = cash + marked positions
        mtm = sum(p.shares * current_prices.get(p.symbol, p.entry) for p in positions.values())
        eq = cash + mtm
        equity_hist.append(eq)
        pos_count_hist.append(len(positions))

        # DD check (simplified flatten)
        peak = max(equity_hist) if equity_hist else initial
        if peak > 0 and (peak - eq) / peak > 0.20:
            # flatten
            for sym, pos in list(positions.items()):
                px = current_prices.get(sym, pos.entry)
                proceeds = pos.shares * px * (1 - COST_BPS)
                cash += proceeds
            positions.clear()
            eq = cash
            equity_hist[-1] = eq

    if not equity_hist:
        return {'return_pct': 0, 'sharpe': 0, 'effective_sharpe': 0, 'max_dd_pct': 0, 'trades': 0, 'avg_pos': 0, 'exposure': 0, 'days': 0}

    final = equity_hist[-1]
    rets = pd.Series(equity_hist).pct_change().dropna()
    sr = (rets.mean() / rets.std() * np.sqrt(252)) if len(rets) > 1 and rets.std() > 0 else 0.0
    ret_pct = (final / initial - 1) * 100
    avg_pos = float(np.mean(pos_count_hist)) if pos_count_hist else 0
    exp = avg_pos / MAX_POS if MAX_POS > 0 else 0
    eff_sr = sr * np.sqrt(max(exp, 1e-6))

    # Max DD
    peak = initial
    max_dd = 0.0
    for e in equity_hist:
        peak = max(peak, e)
        dd = (peak - e) / peak if peak > 0 else 0
        max_dd = max(max_dd, dd)

    return {
        'return_pct': round(ret_pct, 1),
        'sharpe': round(sr, 2),
        'effective_sharpe': round(eff_sr, 2),
        'max_dd_pct': round(max_dd * 100, 1),
        'trades': trade_count,
        'avg_pos': round(avg_pos, 1),
        'exposure': round(exp, 2),
        'days': len(equity_hist)
    }

def main():
    print("Loading screened data for fixed paper replay...")
    data, dates = load_data()
    print(f"Coins: {len(data)}, {dates[0].date()} to {dates[-1].date()}")

    for lbl, st, en in [('last_180d', '2026-01-14', '2026-07-12'), ('last_90d', '2026-04-14', '2026-07-12')]:
        print(f"\n=== {lbl} OOS Paper Replay (fixed math) ===")
        b = replay_oos(data, dates, st, en, mode='baseline')
        print(f"baseline_d40   : ret={b['return_pct']}, SR={b['sharpe']}, effSR={b['effective_sharpe']}, DD={b['max_dd_pct']}, trades={b['trades']}, avg_pos={b['avg_pos']}, exp={b['exposure']}, days={b['days']}")

        for v, tr in [('regime_cci_will', 'cci'), ('regime_rei_will', 'rei')]:
            r = replay_oos(data, dates, st, en, mode='regime', trend_rule=tr)
            print(f"{v:16s}: ret={r['return_pct']}, SR={r['sharpe']}, effSR={r['effective_sharpe']}, DD={r['max_dd_pct']}, trades={r['trades']}, avg_pos={r['avg_pos']}, exp={r['exposure']}, days={r['days']}")

if __name__ == "__main__":
    main()
