# Paper #3 Test Architecture

Goal: OOS-validate ONE more microstructure paper, literature-backed, with
the failures of papers #1/#2 designed OUT.

## Lesson encoded from papers #1/#2 (do NOT repeat)
1. PAPER #1 FAIL: flow cols were 0 non-null on every pair (trade_agg_5m.csv
   MISSING) but the harness trained on zero-filled cols and reported a
   spurious 30/30 "win" (p~1.9e-9) at sub-0.50 magnitude. -> GUARD.
2. PAPER #1/2 SIGN-TEST BUG: k=min gave p=1.0 on lopsided wins. Fixed to
   k=max (p~9.3e-10 / 1.9e-9). Harnesses corrected; reuse corrected version.
3. PAPER #2 n=0: info bars built at ~2000 rows (too few for walk_forward
   >=35000 min). Fixed to ~60000. -> ensure augmented series has enough rows.

## Architecture (generic; target picked below)
For each gated pair:
  baseline = canonical features (113) from Binance 5m  [the CONTROL]
  aug      = baseline + TARGET microstructure features  [the TREATMENT]
  Both: triple_barrier_labels -> PIT walk-forward, 5 folds, XGBoost.
  Paired exact sign test (aug vs baseline) on per-pair OOS dir-acc.
Verdict gate (BOTH required — this is the effect-size floor paper #1 lacked):
  (a) corrected sign test p(aug > baseline) < 0.05
  (b) aug mean OOS dir-acc > 0.50  (above coin-flip; significance without
      effect-size is the paper #1 trap)
Report: n_tested, n_skipped (with skip REASON), wins/losses/ties, p, means.

## MANDATORY DATA-PRESENCE GUARD (the paper #1 fix, built in)
Before training a pair: assert every TARGET feature col is >60% non-null
on that pair's history. If not -> SKIP the pair, record reason
("funding file missing" / "blofin <60% non-null" / etc). Never train on
zero/NaN-filled cols. Print n_tested vs n_skipped at the end.

## Reuse
eval_flow_paper1.py is the template (baseline vs treatment, PIT WF,
paired sign test, corrected k=max). Fork it -> eval_paper3.py. Minimal
new code = the TARGET feature builder + the guard.

## Candidate targets (pick ONE; data reality below)
A. FUNDING-RATE microstructure  [HIGHEST signal, partial coverage]
   Target features: funding_rate (level), funding_z (z-score vs rolling),
   funding_change, funding_x_ret (interaction w/ forward ret).
   Data: data/funding/<SYM>USDT_funding.csv — POPULATED (BTC 901 rows),
   but only ~10 pairs have files. -> eval runs on those 10, reports
   n_skipped=22 (no funding file). Honest, not gutted.
   Literature: funding-rate mean-reversion is the most-cited crypto
   microstructure edge (perpetuals basis predicts short-term reversal).

B. MULTI-VENUE (blofin 2nd-source candles)  [full coverage, lower signal]
   Target features: blofin 5m returns/vol, Binance-blofin midpoint spread,
   vol ratio, lead-lag. Both candles present for ALL gated pairs.
   Data: data/<SYM>USDT_5m_blofin_max.csv — POPULATED, all pairs.
   Literature: cross-venue consensus / independent views improve signal
   (Frontiers 2026: transfer between venues of same asset helps).

C. (BLOCKED) LOB depth (Kuznetsov 2025 / Raffaelli 2026) — orderbook_5m.csv
   MISSING. Untestable until we collect L2 snapshots. Do NOT run; would
   repeat paper #1's gut.

## Decision needed from user
Pick A (funding, 10 pairs, high signal) or B (multi-venue, 32 pairs,
lower signal). Then I implement eval_paper3.py (fork of eval_flow_paper1.py
+ guard + target builder) and run it ALONE, watchdog, notify.
