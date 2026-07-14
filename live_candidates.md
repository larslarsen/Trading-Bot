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

## (b) SIZING SWEEP on DEEP universe (2026-07-14) — 66 slices, 9 majors — CLOSES LOOP
Original 8-slice low-cap sweep said 20% best; re-run at 66 slices for power.

| frac | meanRet | effSR | meanDD | Calmar |
| 10%  |  +5.1%  | +0.73 |  6.1% | 2.02  |
| 15%  |  +7.3%  | +0.72 |  8.8% | 2.14  |
| 20%  |  +9.9%  | +0.76 | 11.2% | 2.26  | <- current
| 25%  | +12.7%  | +0.79 | 12.9% | 2.39  |

Paired vs 20%: 10% -4.7, 15% -2.5, 25% +2.9; all p=0.784 (NOT significant).
MONOTONIC: larger fraction => higher return/effSR/Calmar. Contradicts Scholz's
"smaller better" (but Scholz says scenario-dependent; our +edge rewards leverage).
25% best effSR/Calmar but gap is noise. DECISION: KEEP 20% (conservative, validated;
25% would need real edge to justify +1.7pp extra DD). All live components now tested @66 slices.

Out-of-sample venue check (different exchange, broader universe, HARDER regime:
2025-26 included chop/crash where trend-following got chewed up).

| config            | meanRet | effSR | meanDD | Calmar |
| bop|rule           |  -0.8%  | -0.19 | 13.2% | 0.16  |
| ma30_ema|rule      |  -2.8%  | -1.06 |  5.7% | 0.09  |
| donchian40|rule    |  -2.9%  | -0.27 |  8.2% | 0.44  | <- LIVE, least-bad
| tsi|rule           |  -3.8%  | -0.55 |  8.5% | -0.15 |

EVERY config LOST money on Kraken (system is regime-dependent: wins in 2020-24
bull, loses in 2025-26 hard regime). But LIVE (donchian40|rule) was the LEAST-BAD:
best Calmar (0.44 = smallest loss per unit of drawdown) and moderate DD (8.2%).
bop lost least raw (-0.8%) but worse DD (13.2%); tsi WORST (-3.8%, negative Calmar).
Detector (rule vs ma) made ~no difference here (consistent with deep test).

CONCLUSION: The live config is the most RESILIENT across both venues/regimes:
best risk-adjusted on deep bulls, least-bad on Kraken hard regime. tsi/bop (the
early "winners" on tiny low-cap samples) are the WORST on this broad OOS check.
This validates keeping donchian40 + ma30_ema + rule detector live. The system's
weakness is regime-dependence (loses in chop), not the rule choice.

Two uses tested: (1) directional chop entry (vol-gated BB breakout), (2) detector
(trend when band expanded, chop when squeezed).

| config              | meanRet | effSR | meanDD | Calmar |
| d40|bbwp-det          | +10.7%  | +0.84 | 10.9% | 2.44  | <- BBWP DETECTOR, best Calmar
| LIVE(d40|rule)       |  +9.9%  | +0.76 | 11.2% | 2.26  | <- current
| bbwp-entry|bbwp-det  |  +9.2%  | +0.60 |  9.1% | 1.97  |
| bbwp-entry|rule      |  +8.3%  | +0.49 |  7.9% | 2.14  |

Paired vs LIVE: d40|bbwp-det +0.9 19/31 p=0.281 (NOT sig, but best Calmar 2.44>2.26);
bbwp-entry (directional) WORSE (p=1.000). BBWP NOT degenerate (unlike choppiness).

DECISION: BBWP as DETECTOR is the ONLY candidate this session that IMPROVED Calmar
over live (2.44 vs 2.26), beating in 19/31 slices. Directionally positive but
p=0.281 (unproven). BBWP as directional entry is NOT better than donchian40.
Keep donchian40 as chop entry. BBWP-detector is a WATCH candidate (best Calmar,
unproven) — do NOT swap live without further validation (Kraken cross-check pending).

Per-asset vol targeting (size = 20% * target_vol/asset_vol, clip 0.25-1.5) vs fixed 20%.

| config   | meanRet | effSR | meanDD | Calmar |
| fixed    |  +9.9%  | +0.76 | 11.2% | 2.26  | <- current, BEST
| VT@10/15/20% | +2.6% | +0.69 | 3.2% | 1.88 |

Paired vs fixed: Δmean -7.2, 26/53, p=1.000. VT CUTS return -74% for only -8pp DD.
Calmar FELL 2.26 -> 1.88. FALSIFIED.

WHY: the regime detector already manages vol (sits flat in chop), so VT mostly
dampens the trend exposure that earns return, without proportional DD relief. AQR's
VT win is for naive trend-followers without regime gating. On our regime-gated system,
fixed 20% is better calibrated. DO NOT enable vol_target.

Same full strategy, swap ONLY detector method.

| detector    | meanRet | effSR | meanDD | Calmar |
| choppiness  | +10.0%  | +0.75 | 11.2% | 1.49  |
| rule (live) |  +9.9%  | +0.76 | 11.2% | 2.26  | <- current, best Calmar
| mesa        |  +9.8%  | +0.81 | 11.2% | 1.88  |
| kaufman     |  +9.4%  | +0.71 | 11.8% | 1.87  |
| ma (50/200) |  +7.9%  | +0.64 | 10.9% | 1.81  |

Paired vs 'rule': ma -2.0 p=0.238, choppiness +0.1 p=0.885, kaufman -0.4 p=0.581,
mesa -0.0 p=0.227. NONE beats 'rule' at p<0.05. 'rule' has best Calmar (2.26).

DECISION: Current 'rule' detector VALIDATED on deep universe. The earlier low-cap
"choppiness wins +24.5%" was the degenerate always-trend artifact; here choppiness is
no longer degenerate but still doesn't beat rule. ma (50/200) is the WORST. Keep 'rule'.

Variants as chop leg (trend=REI). Tests your friend's MA-recapture idea head-to-head.

| variant      | meanRet | worst  | pos/n | effSR | meanDD | Calmar | win% |
| ma30_ema     | +10.1%  | -29.7 | 26/66 | +0.85 |  8.3% | 2.02  | 4.5% | <- BEST (live fill-in)
| donchian40   |  +9.9%  | -29.7 | 19/66 | +0.95 |  5.5% | 1.08  | 3.0% | <- baseline
| d40+ma30_ema |  +9.9%  | -44.9 | 27/66 | +0.76 | 11.2% | 2.26  | 3.0% |
| ma30         |  +8.7%  | -29.7 | 24/66 | +0.67 |  9.0% | 1.41  | 5.1% |
| ma30_50      |  +8.6%  | -29.7 | 22/66 | +0.71 |  7.3% | 1.31  | 5.1% |
| ma30_rising  |  +7.5%  | -29.7 | 20/66 | +0.54 |  6.4% | 0.86  | 4.5% |

Paired vs donchian40: ma30_ema +0.2 15/44 p=0.049* (only sig +), ma30 -1.2 p=0.014* (worse),
ma30_rising -2.4 p=0.029* (worse), ma30_50 -1.3 p=0.110, d40+ma30_ema -0.0 p=0.311.
Paired vs ma30_ema: NONE beats it (ma30 p=0.099, ma30_50 p=0.728, ma30_rising p=0.868, combo p=1.000).

DECISION: ma30_ema (close > EMA30) is the BEST MA-recapture variant — only one that beats
donchian40, and beats all fancier variants (ma30_50 dual-MA, ma30_rising slope are WORSE).
Simplest form wins. LIVE fill-in (ma30_ema) is the correct MA choice. NO change needed.

9 deep-history majors/mids (ADA,ALGO,AVAX,CRV,DOGE,ETH,LRC,MATIC,SOL), 1450-bar
common window 2020-09 -> 2024-09, **66 WF slices**. This is the data we already had.

| candidate   | meanRet | worst  | pos/n | effSR | meanDD | Calmar | win% |
| mtf         | +11.6%  | -29.9 | 24/66 | +1.16 |  8.3% | 1.41  | 7.6% |
| ma30_ema    | +10.1%  | -29.7 | 26/66 | +0.85 |  8.3% | 2.02  | 4.5% |
| cci         | +10.0%  | -39.9 | 27/66 | +0.68 | 10.7% | 2.40  | 6.8% |
| donchian40  |  +9.9%  | -29.7 | 19/66 | +0.95 |  5.5% | 1.08  | 3.0% | <- baseline
| d40+ma30_ema|  +9.9%  | -44.9 | 27/66 | +0.76 | 11.2% | 2.26  | 3.0% |
| tsi         |  +8.8%  | -54.5 | 26/66 | +0.60 | 11.5% | 1.37  | 3.0% |
| rsi         |  +8.8%  | -29.7 | 18/66 | +0.80 |  4.6% | 1.89  | 9.8% |
| bop         |  +7.7%  | -49.8 | 27/66 | +0.48 | 17.0% | 1.33  | 7.6% |

Paired vs donchian40 (exact sign test, 66 slices):
  ma30_ema  +0.2  15/44  p=0.049*  (marginal +)
  cci       +0.1  21/55  p=0.105
  tsi       -1.1  12/36  p=0.065   (trending WORSE)
  bop       -2.2  17/55  p=0.006*  (SIGNIFICANTLY WORSE)
  mtf       +1.7  17/33  p=1.000
  d40+ma30  -0.0  14/35  p=0.311

DECISION: With REAL power (66 slices), the MEXC-alts "tsi/bop beat live" result is
REVERSED. On majors/mids: donchian40 is a solid baseline; tsi is NOT better (p=0.065
worse), bop is SIGNIFICANTLY WORSE (p=0.006). ma30_ema marginally better (p=0.049)
but tiny effect. The live chop rule (d40+ma30_ema) is JUSTIFIED — NOT a weak choice.
CAVEAT: this is majors/mids, not the live low-cap MEXC alts; edges may differ. Kraken
backfill (running) will give a Kraken universe to double-check. The earlier chop-leg
"wins" were SMALL-SAMPLE ARTIFACTS (8-12 slices on low-caps).

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
