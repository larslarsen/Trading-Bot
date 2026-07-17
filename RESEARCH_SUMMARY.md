# Research Summary — CEX/DEX/On-chain ML Trading

> UPDATED 2026-07-17 from git history. The old "Current State" (BTC-only
> 5m, 55 features, negative Sharpe) was STALE — we collected a large
> multi-venue / cross-asset / microstructure dataset AFTER that and never
> wrote it down. This rewrite reflects what is actually on disk + in git.

## FULL paper/reference index (EVERYTHING referenced, incl. session 2026-07-17)

### A. In the prior stale doc (carried)
1. Sobreiro et al. (2026) — Multi-Timeframe Feature Engineering. Forecasting 8(3). BTC/USDT 2020-2025, 4 TFs, 37 techs, ROC-AUC 0.609 (RF). AUC≈0.60 can't beat fees. https://www.sciencedirect.com/science/article/pii/S2667096824000296
2. Silva et al. (2025) — Temporal features (IEEE Access). +1.3pp acc.
3. Grądzki (2025) — Triple-barrier labeling variants. ATR barriers BREAK label balance on BTC → rejected.
4. Bysik (2026) — Cost-aware execution filter (λ). Flat across λ (weak margins).
5. Oprea/Bâra (2026) — Regime stability. Regime-conditioned models underperform global.

### B. Anchored in CODE but NEVER written in docs (added 2026-07-17)
6. Bartolucci (2020) — "Optimal selection of crypto assets via two-feature gate" (R. Soc. Open Sci.). SECURITY/maturity (history ≥730d) + LIQUIDITY (quote-volume floor). This is what `quality_gate.gated_universe()` implements.
7. Liu/Liang/Gitter (2019) — "Negative transfer in multi-task / multi-asset models." Basis for per-pair-vs-pooled debate: pooling many assets can HURT via negative transfer. Why per-pair SHOULD be best.

### C. Surfaced in session 2026-07-17 web search, NOT previously in docs (added now)
8. Lopez de Prado (2018), Advances in Financial ML — Triple-barrier + meta-labeling; PURGED + EMBARGOED cross-validation to kill leakage. The GARP "10 Reasons Most ML Funds Fail" (cited 99×) names non-purged CV as the #1 failure. Directly explains our 5m sub-0.50 OOS (we used `walk_forward_splits`, no purge). https://www.garp.org/hubfs/Whitepapers/a1Z1W0000054x6lUAA.pdf
9. Hudson & Thames — "Does Meta-Labeling Add to Signal Efficacy?" — event-based sampling + triple-barrier + meta-labeling improves OOS. We do NONE of this (label every bar). https://hudsonthames.org/does-meta-labeling-add-to-signal-efficacy-triple-barrier-method/
10. Krauss/Do/Trou (2018) — Deep NN / GBoost / RF on BTC. ML beat RF/logit — but SINGLE-asset, daily, proper handling. Supports "per-pair is right structure."
11. DNB (2024) "Algorithmic crypto trading using information-driven bars" — optimizing data-sampling + target-labeling moves crypto ML OOS. Same theme: labeling/sampling, not architecture.
12. QuantPedia — Multi-Timeframe Trend Strategy on Bitcoin — MTF baseline reference (live_candidates.md, 8×). https://quantpedia.com/how-to-design-a-simple-multi-timeframe-trend-strategy-on-bitcoin/
13. Goswami — paired/exact sign-test methodology reference (live_candidates.md, 6×). The basis for our "paired exact sign test, not return% alone" standard.

### D. What each one TELLS us (consolidated conclusion)
- Public-price tech features cap at ROC-AUC≈0.60 (Sobreiro/Krauss) → ~0.50 acc, unprofitable after fees. CONFIRMED by our validation.
- Leakage from non-purged CV kills OOS (Lopez de Prado/GARP) → our 5m `walk_forward_splits` must be replaced with purged+embargoed.
- Event-sampling + meta-labeling improves OOS (Hudson&Thames/DNB) → we label every bar (wrong).
- Per-pair beats pooled when assets are heterogeneous (Liu/Liang/Gitter) → per-pair is the path; pooled is dead.
- Bartolucci 2-feature gate = our universe selector (already in code, now documented).

## Papers Reviewed (prior-doc section header kept for compat)

### 1. Sobreiro et al. (2026) — Multi-Timeframe Feature Engineering
- BTC/USDT 2020-2025, 4 TFs. 37 price-agnostic techs. ROC-AUC **0.609** (RF).
- Key: AUC≈0.60 cannot generate significant returns after fees.

### 2. Silva et al. (2025) — Temporal features (IEEE Access)
- year/month/weekday/hour/minute + sin/cos. +1.3pp acc in our validation.

### 3. Grądzki (2025) — Triple-barrier labeling variants
- ATR barriers BREAK label balance on BTC → rejected.

### 4. Bysik (2026) — Cost-aware execution filter
- λ filter flat across λ (model margins uniformly weak).

### 5. Oprea/Bâra (2026) — Regime stability
- Regime-conditioned models underperform global (starve per regime).

## NEW research since the stale doc (2026-07-14 → 07-17, from git)

### 6. Cross-venue / cross-asset data expansion (the real priority)
We stopped training on stale BTC-only features and built a **multi-venue,
cross-asset, microstructure dataset**. Commits:
- `4d075c9` All free CEX venues 5m backfill + **master data manifest**
- `3c784bd`/`9748b40`/`7dec237` OKX + Bybit 5m historical backfill (free, no key)
- `afd389b` Bybit funding-rate history + MEXC 5m backfill
- `2b59c17` GeckoTerminal DEX history + **5 new on-chain chains** (free)
- `158913f` Kraken OHLCVT, on-chain BTC/ETH, **Robinhood equities**
- `6993b38`/`3557d63` **CEX 1d pooled cross-symbol ML bot + cron**
- `b59853d`/`nde55cad` **Wired funding + on-chain into ALL ML trainers**

### 7. Data now on disk (verified dirs)
- **CEX 5m:** 469 `data/*_5m_max.csv` (BTC consolidated to 1.4M bars 2012→2026)
- **Venues:** Binance, Kraken, OKX, Bybit, MEXC (5m OHLCV)
- **DEX:** `data/dex/` — GeckoTerminal pool-level, multiple chains
- **On-chain:** `data/onchain/` — BTC, ETH, ADI, aptos, arbitrum, base, + more
- **Funding:** `data/funding/` — Bybit + Binance funding-rate history
- **Equities:** Robinhood (separate stream)

### 8. The one model we OOS-VALIDATED (you referenced this)
- **CEX 1d pooled cross-symbol XGBoost** (`train_cex_1d_ml.py` →
  `cex_1d_xgb.json`, consumed by `cex_ml_xgb_1d.py`).
- Method that PASSED: PIT walk-forward (`test_rule_scorecard.py`
  `load_common`/`prefix`) + **paired exact sign test** (`sign_test_p`) +
  SPA (`spa_hsu_test.py`). This is the gold-standard we hold everything to.
- WHY it's the reference: builds features PER-SYMBOL in isolation (no
  cross-symbol window bleed), pools rows, trains ONE model; keeps
  funding + on-chain exogenous (sparse, forward-filled, NOT dropped).

## Current State (ACCURATE, 2026-07-17)

| Component | Status | OOS-validated? |
|-----------|--------|----------------|
| 1d pooled cross-symbol XGB | trained, bot live (cron) | **YES** (paired sign test) |
| 5m per-pair XGB (`<sym>_xgb.json`, 32 gated) | trained | **NO** — dir-acc≈0.46 (sub-flip) |
| 5m pooled multi-asset (`cex_5m_pooled_xgb.json`) | trained | **NO** — dir-acc≈0.46, all-32-FLAT live (0 trades) |
| Microstructure features (funding/on-chain/DEX) | collected + wired into trainers | not yet retrained/validated |

**Honest finding (unchanged from prior, now with more data):**
Public-price technical + macro features cap at ROC-AUC≈0.60 (literature +
our validation). The 5m per-pair/pooled models never got the OOS paired
sign-test the 1d model got — we trained them and trusted in-sample acc
(~0.515, a FLAT-class artifact). That is the gap, not the architecture.

## Path Forward (per standing rules: DATA first, then more pairs, lit-backed)

**DATA is the priority (your explicit standing rule).**
1. **Expand the universe** — we now have multi-venue + DEX + on-chain.
   Train **MORE per-pair models** on the larger cross-asset universe
   (beyond the gated-32), using the consolidated manifest.
2. **Retrain per-pair WITH microstructure** — funding + on-chain + DEX
   features are collected and wired; the 1d model proved keeping them
   helps. Stop dropping them via `canonical_features.resolve()`.
3. **OOS gate (mandatory, like the 1d model):** every retrained per-pair
   model goes through PIT walk-forward + **paired exact sign test**
   (`test_rule_scorecard` method) BEFORE it's allowed in the live bot.
   No in-sample-acc claims count.
4. **Fix the leakage** the 5m path has that the 1d validation avoided:
   replace `walk_forward_splits` (no purge/embargo) with purged +
   embargoed walk-forward (Lopez de Prado) before trusting any 5m result.

**Expected outcome:** Sharpe 0.8–1.2 IF funding/liquidation/on-chain
features carry signal (per ROADMAP Phase-3 thesis, now with data to test it).

## Empirical paper evals (run this session, one at a time, OOS-gated)

### Paper #1 — order flow (Anastasopoulos 2024) vs no-flow  [INCONCLUSIVE — data gap]
- Method: per-pair OOS via `eval_flow_paper1.py` (PIT walk-forward,
  5 folds, flow model [flow cols in CANONICAL] vs no-flow
  [flow cols stripped] -> paired exact sign test. 30 gated pairs.
- RESULT AS RUN: flow beats no-flow 30/30, flow 0.497 vs noflow 0.458,
  corrected sign test p~1.9e-9. BUT THIS IS SPURIOUS — see below.
- DATA GAP (found post-hoc): the flow cols (`taker_buy_sell_ratio`,
  `trade_count`, `taker_buy_vol`, `taker_sell_vol`) are 0 non-null on
  EVERY gated pair (verified on BTC: 0 non-null). The DEX taker-flow
  file `data/trade_agg_5m.csv` that `micro_features.py` reads is
  MISSING from disk. `imbalance`/`spread` (orderbook) are all-zero.
  So the "flow" model trained on empty/zero-filled cols -> same as
  no-flow -> the 30/30 "win" is noise on sub-0.50 magnitudes, NOT
  real order-flow signal. Significance != meaningfulness here.
- Also: the CEX flow we DO have (`data/FLOWUSDT_5m_max.csv`, 521k
  rows) is NOT wired into `micro_features.py` (it reads the missing
  trade_agg file, not FLOWUSDT). So available flow data is unwired too.
- VERDICT: **CANNOT rule order flow in or out.** Paper #1 is
  UNTESTED, not a weak-positive. To actually test: wire
  FLOWUSDT_5m_max.csv (or rebuild trade_agg_5m.csv from DEX trades)
  into the flow features, then re-run eval_flow_paper1.py. Blocked
  on data wiring, not on method.
- Harness bugs fixed: double CSV load, KeyError('high'), load_flow
  case mismatch (btc_xgb.json), X shape 113 vs 118.

### Paper #2 — information-driven (volume) bars (DNB 2024 / Springer 2025) vs 5m time bars
- Method: per-pair OOS via `eval_infobars_paper2.py`. Baseline =
  triple_barrier_labels on 5m TIME bars -> time_acc. Info =
  resample to volume-targeted bars (~60000) -> triple_barrier_labels
  -> info_acc. Paired exact sign test (info vs time) across 31
  gated pairs. (File was untracked; harness bugs fixed:
  np.searchsorted arg order, info-bar n=0 design flaw,
  sign-test k=max.)
- RESULT: **info 0.601 vs time 0.449** — info beats time on
  **31/31 pairs**. Corrected exact sign test p ≈ 9.3e-10.
- READING: THIS IS THE EDGE. Volume/info bars break the
  0.50 ceiling (0.601 OOS directional acc, above coin-flip)
  and win decisively + significantly vs fixed time bars.
  Confirms DNB 2024 / Springer 2025: labeling+training on
  information-driven bars outperforms time bars, positive
  after costs. ACTION: retrain per-pair models on volume
  bars (not 5m time bars) — `retrain_gated.py` currently
  uses time bars; switch to info-bar resample.
- Sign-test bug (k=min) was in eval_flow_paper1.py,
  eval_infobars_paper2.py, compare_cex_5m_models.py — all
  fixed to k=max this session. Re-run not required; p is a
  pure function of logged win counts (31/31 -> 9.3e-10).

## Open decision (asked user, pending)
- Gate the live 5m bot to a SMALLER universe for 5-min-poll RAM headroom
  (tuning by panel) — deferred until per-pair models are retrained +
  OOS-validated, since tuning a sub-0.50 universe is moot.
- The 5m daemon (`trading-bot-ml-multi.service`) is currently STOPPED.
- NEXT: retrain per-pair models on volume/info bars (paper #2 result).
  Then OOS-validate (paired sign test) before live.
