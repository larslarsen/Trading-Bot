# Live Candidate Summary
Generated: 2026-07-12

## Baseline
d40 Donchian — full-history penalized effective Sharpe: ~0.57

## Top Rule Families (full-history OOS)
Selected by penalized effective Sharpe, 50bps-robust, cap-5 portfolio.

1. CCI — penalty 0.84, eff Sharpe 4.85, DD 35.6%, 370 trades
2. REI — penalty 0.76, eff Sharpe 4.39, DD 19.0%, 354 trades
3. TSI — penalty 0.66, eff Sharpe ~4.0, DD ~38%, ~445 trades
4. BOP — penalty 0.56, eff Sharpe ~3.2, DD ~38%, ~445 trades
5. MTF confirm — penalty 0.55, eff Sharpe ~3.1, DD ~16%, ~283 trades

* Williams %R — penalty ~0.70, eff Sharpe ~2.9, DD ~18%, ~214 trades
* ROC, Ichimoku, TRIX, Price-Volume div, RVI — secondary candidates

All above passed 50bps cost sweep.

## Regime Switcher (approved config)
- trend: CCI or REI
- chop: Williams %R
- gate: default ADX/volatility filter from `engine.py` (`default_regime`)
- cap: 5 positions, 20% equity/trade
- risk: 20% DD halt, $100 floor, 3% daily loss limit

## Decision
Freeze the regime switcher as the live candidate runner:
1. Compute market regime daily via `default_regime`
2. Apply trend-rule entries/exits when trend
3. Apply Williams %R entries/exits when chop
4. Write results into canonical `fair_compare_full_oos.csv` via `fair_compare_path`

## Next Tuning Step
- Run penalty/Sharpe rebalance on the 3 candidate objective comparison approaches if desired.
- Optionally build the multi-rule runner that cycles among the top 5 rules under cap-5.
