# Multi-Timeframe (MTF) Trading — Literature Review & Test Design

*Literature-first, falsifiable discipline. Built so we TEST MTF before coding it,
not assume it. Updated 2026-07-14 from a live literature search.*

## 1. What the evidence actually says

### 1a. EMPIRICAL STUDY — Goswami (2026), SSRN, "Multi-Timeframe Signal
Confirmation in Algorithmic Cryptocurrency Trading: A Backtest Study on
ETH/USDT" (independent; CCXT/KuCoin; 1yr ETH/USDT Apr2025–Apr2026)
- 5-min strategy WITHOUT MTF confirmation: 1,969 trades, profit factor
  **0.106**, max DD **99.17%** (lost nearly all capital).
- Same 5-min strategy WITH MTF confirmation (take 5m signal only if the 15m
  chart independently agrees): 9 trades, profit factor **0.722**, max DD
  **1.46%**.
- Statistical test on the 9 trades: **p = 0.687** → sample far too small to
  conclude anything.
- Author's takeaway: MTF confirmation works as a NOISE FILTER in principle, but
  needs more data to validate.
- WHY THIS MATTERS TO US: this is the "which TF pair is arbitrary / under-tested"
  critique made EMPIRICAL. MTF slashed trades 99.7% and DD 99%→1.5%, but with
  n=9 the result is NOT validated. Direct proof we must WF+OOS test, not trust a
  single backtest.

### 1b. EMPIRICAL STUDY — QuantPedia (Nov 2025), "How to Design a Simple
Multi-Timeframe Trend Strategy on Bitcoin" (Gemini; Dec2018–Nov2025;
long-only MACD; D1H1 Elder "Triple Screen" filter)
- Pure 1H MACD (no filter): 2,262 trades, ann return 4.6%, Sharpe **0.33**,
  Calmar 0.19, max DD –23.9%.
- + D1 (daily) trend filter (trade only in direction of higher-TF trend):
  ~1,000 trades, ann return **6.6%**, max DD **–12.4%**, Sharpe **0.80**.
- Mechanism (critical, no lookahead): the daily signal is applied ONLY after the
  daily candle closes (after midnight). Entries taken only in the D1 trend
  direction; counter-trend noise eliminated.
- WHY THIS MATTERS: this is the BETTER-EVIDENCED MTF pattern (1,000 trades,
  Sharpe 0.33→0.80) and it uses a LOGICAL filter (Elder principle), not a
  data-mined parameter. Aligns with our "literature-backed, no ad-hoc filters"
  rule. The D1H1 structure = higher-TF trend gate + lower-TF trigger.

### 1c. Practitioner corpus (Tradeciety, VTMarkets, Investopedia, TradingView,
Brian Shannon "Technical Analysis Using Multiple Timeframes")
- Convergent claim: higher-TF = trend/context (FEWER, more reliable signals);
  lower-TF = entry timing (MORE noise). Dominant pattern = higher-TF FILTER +
  lower-TF TRIGGER. 3-tier (trend/setup/trigger) is common.
- CAUTION (Markets4you): stacking MTF with redundant indicators invites
  CONFIRMATION BIAS — a warning against over-layering filters that all say the
  same thing.

## 2. How this maps to OUR engine
- Our existing REI trend gate (14/2.0 ATR trailing) IS already a higher-TF-style
  trend filter at one timeframe. MTF would ADD a SECOND-timeframe confirmation:
  e.g. only take the 1d donchian/MA entry when 4h (or 1d-vs-4h) also agrees.
- QuantPedia's D1H1 is the closest validated template: higher-TF trend gate
  (after its candle closes — NO lookahead) + lower-TF entry trigger.

## 3. CRITICAL data constraint (drives collector_daemon.py)
- 1d bars RESAMPLE to 3d / 1w / 1M from the 1d we already have → NO new
  collection needed for those.
- 4h / 8h / 1h CANNOT be derived from 1d → MUST be collected forward
  continuously or lookback is lost forever. This is why the daemon exists:
  it appends 1h/4h/1d (MEXC+BloFin) + 8h (BloFin only) so the window compounds
  with no gaps. By the time we test MTF, we'll have real intraday history.

## 4. Falsifiable test design (NOT opinion — this is what we run)
When intraday history is sufficient (~backtest from 2025-07 forward on 1h/4h):
- Variants to compare, each as a FULL strategy (own entry+exit), scored by a
  panel: ret / exposure-adj Sharpe / DD / Calmar / win%, plus a paired exact
  sign test vs the canonical single-TF baseline (Donchian40 on 1d):
  1. Baseline: current single-TF canonical (1d only).
  2. D1 filter + H4 entry trigger (mirror QuantPedia D1H1).
  3. D1 filter + H1 entry trigger.
  4. H4 filter + H1 entry trigger.
- Hypothesis to falsify: adding MTF confirmation improves exposure-adjusted
  Sharpe/Calmar on WF + OOS (per Goswami/QuantPedia mechanism: fewer false
  signals → smoother equity). NULL if it merely reduces trades without lifting
  risk-adjusted return, or if it overfits (OOS collapses).
- Guardrails (your standing rules): no ad-hoc filters; WF-validated; OOS paper
  on the ACTUAL engine; paired sign test, not return% alone. Watch for the
  Goswami trap — if a variant yields very few trades, report n and a sign-test
  p-value, don't celebrate a high Sharpe from tiny n.

## 5. Open questions the literature does NOT settle
- WHICH TF pair is optimal — explicitly unresolved (Goswami n=9; QuantPedia only
  tested D1H1). Must be empirically selected via the panel above, not chosen by
  intuition.
- Confirmation (both TFs must agree) vs cascade (lower-TF entry only allowed if
  higher-TF in trend) — QuantPedia uses cascade (direction-only); Goswami uses
  agreement. Test both.
- Does MTF beat our canonical Donchian40 on 1d on WF+OOS, or is it overfitting
  noise? Falsify, don't assume.

## Sources
- Goswami, R. (2026). Multi-Timeframe Signal Confirmation in Algorithmic
  Cryptocurrency Trading: A Backtest Study on ETH/USDT. SSRN 6683818.
  https://ssrn.com/abstract=6683818
- QuantPedia (2025). How to Design a Simple Multi-Timeframe Trend Strategy on
  Bitcoin. https://quantpedia.com/how-to-design-a-simple-multi-timeframe-trend-strategy-on-bitcoin/
- Tradeciety — How To Perform A Multi TimeFrame Analysis.
- Investopedia — Master Trading With Multiple Time Frames.
- Markets4you — Technical Analysis Using Multiple Timeframes (confirmation-bias caution).
- Shannon, B. — Technical Analysis Using Multiple Timeframes (book).
