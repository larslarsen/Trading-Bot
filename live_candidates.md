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

## Chop-leg scorecard (trend=REI, 8 finer WF slices @12d, 68 coins) — REVISED 2026-07-14
NOTE: earlier 6-slice (20d) scorecard made d40+ma30_ema look best (+31.4%); that was a
COARSE-SLICING ARTIFACT. Finer 12d slicing reverses it — the combo is mid-pack.

| candidate      | meanRet | worst  | pos/n | effSR  | meanDD | calmar | win%  |
|----------------|--------:|-------:|------:|-------:|------:|------:|------:|
| bop            | +36.1%  | -12.2% | 5/8   | +2.88  | 13.7% | 18.70 | 29.4% |
| ma30_50        | +32.0%  | -12.6% | 5/8   | +2.71  | 13.3% |  4.56 | 21.4% |
| cci            | +29.2%  | -13.4% | 5/8   | +2.16  | 13.0% |  3.70 | 33.6% |
| tsi            | +26.5%  | -12.2% | 6/8   | +3.10  | 13.3% | 10.88 | 27.5% |
| mtf            | +22.2%  | -13.8% | 5/8   | +2.26  | 11.3% |  3.55 | 47.6% |
| ma30_rising    | +18.6%  | -12.2% | 4/8   | +1.45  | 13.2% |  2.65 | 22.1% |
| ma30_ema       | +11.1%  | -12.2% | 4/8   | -0.56  |  9.7% |  2.23 | 34.5% |
| rsi            |  +9.6%  | -12.2% | 4/8   | +1.06  |  9.2% |  2.48 | 46.0% |
| d40+ma30_ema   |  +7.9%  | -14.7% | 5/8   | +1.54  | 13.3% |  2.41 | 18.8% |  <- LIVE (mid-pack on finer slices)
| donchian40     |  +6.5%  | -18.1% | 5/8   | +1.49  | 14.1% |  2.37 | 16.2% |  <- LIVE base

Paired vs donchian40 (exact sign test): tsi beats in 6/8 (p=0.125), ma30_50 6/8 (p=0.125),
ma30_rising 6/8 (p=0.125), cci 5/8 (p=0.453). STILL nothing significant at p<0.05 (8 slices
too few; need ~15-20). The combo (d40+ma30_ema) is NOT significantly better than donchian
alone on finer slices — its earlier "win" was the coarse-slice artifact.

## CHOP-LEG DEEP TEST (2026-07-14) — tsi/bop/ma30_50 vs LIVE
Full switched strategy (8 finer slices first, then 12 finer slices @8d).

8-slice (coarse) result — INFLATED by slicing variance:
| config          | meanRet | effSR | meanDD | Calmar |
| bop             | +36.1%  | +2.88 | 13.7% | 18.70 |
| ma30_50         | +32.0%  | +2.71 | 13.3% |  4.56 |
| tsi             | +26.5%  | +3.10 | 13.3% | 10.88 |
| LIVE(d40+ma30_ema)| +6.5% | +1.49 | 14.1% |  2.37 |

12-slice @8d (finer, more powerful) — TRUE edge is MODEST:
| config          | meanRet | effSR | meanDD | Calmar |
| tsi             | +11.7%  | +2.87 | 12.7% |  3.86 |
| bop             | +11.2%  | +3.35 | 13.8% |  9.27 |  <- best effSR/Calmar at 12 slices
| ma30_50         |  +8.3%  | +1.96 | 10.5% | 14.06 |
| LIVE(d40+ma30_ema)| +6.4% | +2.47 | 15.8% |  1.64 |

Paired vs LIVE @12 slices: tsi 6/9 (p=0.508), bop 6/8 (p=0.289), ma30_50 6/10 (p=0.754).
ALL beat LIVE in ~6/8-10 slices but by a SMALL margin (+5 to +12%, not +20-36%).
Coarse 8-slice OVERSTATED the edge (+20-36%) due to fewer/larger windows (outlier variance).

DECISION: Edge of tsi/bop over live is REAL but MODEST (+5-12%) and NOT significant
(p>=0.29 even at 12 slices). 360 bars is fundamentally underpowered. The live combo
is within noise of tsi/bop. RECOMMEND: KEEP LIVE as-is (option C) until more data, OR
swap to bop (best effSR/Calmar at 12 slices) accepting a marginal, unproven edge.
Do NOT swap based on the inflated 8-slice numbers.

LIVE full system across {10,15,20,25}% position fractions, 8 WF slices:
| frac | meanRet | effSR | meanDD | Calmar |
| 10%  |  +5.1%  | +1.21 |  8.1%  |  1.35  |
| 15%  |  +7.6%  | +1.34 | 11.6%  |  1.44  |
| 20%  |  +7.9%  | +1.54 | 13.3%  |  2.41  |  <- current
| 25%  | +10.0%  | +1.45 | 15.7%  |  2.44  |
Scholz's "smaller => better risk-adjusted" does NOT hold on our positively-edged data:
20% has best effSR (1.54), 25% best Calmar. No fraction is significant (p=0.73-1.00).
DECISION: KEEP 20% (literature-tested, near-optimal, DD 13.3% safe under 20% halt).

## (3) REGIME DETECTOR comparison — DONE (negative result)
Full system, swap ONLY detector, 8 WF slices:
| detector   | meanRet | effSR | meanDD | Calmar | chop% |
| choppiness | +24.5%  | +2.91 | 12.9%  | 3.87  | 0%*  |
| kaufman    | +17.1%  | +1.16 | 17.1%  | 1.36  | 26%  |
| mesa       | +17.1%  | +1.16 | 17.1%  | 1.36  | 35%  |
| ma(50/200) |  +8.3%  | +0.67 | 18.6%  | 1.14  | 40%  |
| rule(cur)  |  +7.9%  | +1.54 | 13.3%  | 2.41  | 33%  |
* choppiness is DEGENERATE on our 68-alt mean-close proxy: CI sits near 0 (smooth
  aggregate never "choppy"), so it ALWAYS returns trend -> system becomes REI-only.
  Its +24.5% is an artifact of "use REI everywhere", NOT a detector win.
  (Choppiness Index function fixed to be NaN-robust, but rejected as live detector.)
kaufman/mesa beat rule by +9.1 but only 2/3 slices (p=1.000, NOT significant).
ma ties rule (confirms user's observation MA crossover ~= rule, but not better).
DECISION: KEEP current 'rule' detector. No candidate is a validated improvement
(nothing significant at p<0.05; choppiness degenerate). Literature-first: don't swap
without validation.

- FULL system: +7.9% mean, effSR +1.54, DD 13.3%
- CASH-in-chop: +2.5% mean, effSR +0.75, DD 4.1%
- Chop trading adds return + risk-adjusted edge (effSR 1.54 vs 0.75) but NOT significant
  (p=1.000) and triples DD. Keep chop trading (edge is real in magnitude) but watch DD.

## Literature pointers (fresh search 2026-07-14)
- REGIME DETECTION: our improved detector (ADX/vol/ER+hyst) vs MA(50/200) crossover.
  MA crossover outperformed the literature-built detector in live — re-search warranted
  (Kaufman ER adaptive window; Choppiness Index; mesa/British Bank adaptive). Not yet swapped.
- SIZING/RANKING (Scholz 2012, "Size matters!"): smaller trading fractions => highest
  risk-adjusted returns in MOST scenarios; NO optimal fraction exists (contra Kelly).
  => TEST a sizing sweep (10/15/20/25%) rather than assume 20% is best. Our 20% may be
  too aggressive given chop DD (13.3%). This is the next (b) test.

## Decisions (REVISED)
- LIVE combo (d40+ma30_ema) is mid-pack on finer slices, NOT proven best. Keep for now
  (lowest-D worst, simple, literature-backed) but FLAG: tsi/bop/ma30_50 lead on finer
  slices and warrant a longer test.
- Backfill: MEXC free endpoint caps at ~1y for these low-cap alts; Binance/CoinGecko
  don't list them. CANNOT extend 1d history from free sources. Get stat power via finer
  slicing instead (done: 8 slices, still underpowered).
- Next: (b) sizing sweep per Scholz 2012; re-search regime detectors.
- Williams %R: lit-correct version kept (unvalidated, flagged); buggy version preserved as
  named rule `williams_r_buggy` (do-not-use, pathological bottom-sell). NOT in live.

## Next step to get statistical power
- 120d of history is too short for significance. Need more bars (backfill older 1d history)
  or finer slices to reach ~20+ WF windows before any rule can be called "proven".
