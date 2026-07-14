# Live Candidate Summary (UPDATED 2026-07-14)

## FINAL LIVE CONFIG (validated, running)
- Trend leg: REI + ATR trailing (14/2.0, trend-gated) — walk-forward validated lift.
- Chop leg: **donchian40 (primary, 40d breakout) + ma30_ema recapture (fill-in when donchian silent)**.
- Regime: improved detector (ADX/vol/ER + hysteresis) + MA crossover (50/200).
- Risk: 5 positions, 20% sizing, 20% DD halt, $100 floor, 3% daily loss limit.
- Sizing base = stable capital base (not declining equity).

## How rules are now compared (methodology, fixed 2026-07-14)
Earlier we ranked rules on ENTRY quality only (signal separation) — wrong, it hid bad
exits. Then full entry+exit walk-forward on return% alone — better but no statistical teeth.
Current method (test_rule_scorecard.py):
1. FULL STRATEGY: each rule runs its own entry+exit via shared PortfolioEngine (live math).
2. PANEL per walk-forward slice: ret%, exposure-adjusted Sharpe (effSR), maxDD%, Calmar,
   win%, trades — not just return.
3. PAIRED statistics: each slice is the same market period, so rule differences are paired.
   Exact binomial sign test → "A beats B at p<0.05", not just "higher mean".
4. STABILITY: worst-slice return + positive fraction + std of slice returns.

## Chop-leg scorecard (trend=REI, 6 non-overlapping 20d WF slices, 68 coins)
Ranked by mean return, with panel + paired-vs-live (donchian40):

| candidate      | meanRet | worst  | pos/n | effSR  | meanDD | calmar | win%  |
|----------------|--------:|-------:|------:|-------:|------:|------:|------:|
| tsi            | +38.0%  | -19.1% | 3/6   | +1.37  | 17.7% | 15.02 | 36.9% |
| d40+ma30_ema   | +31.4%  | -19.1% | 5/6   | +3.44  | 10.6% |  5.64 | 35.0% |  <- LIVE (best effSR/consistency)
| bop            | +28.0%  | -19.1% | 4/6   | +1.75  | 13.8% | 23.91 | 43.9% |
| mtf            | +19.0%  | -19.1% | 3/6   | +0.90  | 12.5% |  2.77 | 33.1% |
| donchian40     | +16.2%  | -19.1% | 4/6   | +2.53  | 12.2% |  4.33 | 40.6% |  <- LIVE base (no fill)
| ma30_50        | +14.1%  | -19.1% | 2/6   | +0.14  | 10.9% |  3.98 | 24.9% |
| rsi            |  +8.9%  | -19.1% | 3/6   | +0.93  | 12.8% |  2.21 | 50.2% |
| cci            |  +6.9%  | -19.1% | 3/6   | +0.25  | 11.4% |  2.01 | 28.7% |
| ma30_ema       |  +2.7%  | -19.1% | 3/6   | +0.35  |  9.9% |  0.81 | 20.0% |
| ma30           |  -0.0%  | -19.1% | 3/6   | -0.04  | 11.0% |  0.52 | 21.4% |
| ma30_rising    |  -5.1%  | -19.1% | 1/6   | -1.03  | 18.0% | -0.54 | 19.7% |

## Paired vs LIVE donchian40 (exact sign test on slice returns)
- NO candidate beats donchian40 at p<0.05 (6 slices too few for significance).
- tsi beats live in 2/6 slices (p=1.000) — its +38% is 2 outlier slices, NOT robust.
- LIVE combo (d40+ma30_ema) beats donchian40-alone in 1/6 (p=1.000) but dominates on
  effSR (+3.44 vs +2.53), consistency (5/6 vs 4/6), and DD (10.6% vs 12.2%).

## Decisions
- KEEP LIVE config. It is best on risk-adjusted panel (effSR, consistency, DD) even though
  no rule is statistically distinguishable from donchian40 on 6 slices.
- REJECT tsi swap despite higher mean: edge is 2 lucky slices; paired stats not significant.
- REJECT ma30_ema as sole chop rule (worst standalone); keep only as fill-in.
- Williams %R: lit-correct version kept (unvalidated, flagged); buggy version preserved as
  named rule `williams_r_buggy` (do-not-use, pathological bottom-sell). NOT in live.

## Next step to get statistical power
- 120d of history is too short for significance. Need more bars (backfill older 1d history)
  or finer slices to reach ~20+ WF windows before any rule can be called "proven".
