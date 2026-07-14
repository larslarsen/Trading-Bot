# CURRENT PRODUCTION FOCUS (as of 2026-07-14)

Core: altcoin daily regime-based long-only rule system.
- FINAL LIVE CONFIG: trend = REI (no fill-in; REI silent on only 1.6% of trend
  days) + ATR trailing (14/2.0, trend-gated); chop = CCI primary + ma30_ema
  recapture fill-in when CCI is silent (rare — 0/97 chop days silent on 68-coin set).
  Held until MTF 4h data (~Oct 2026) achieves stat significance.
- Centralized in engine.py (donchian, cci, rei, williams_r [lit-correct + buggy-named],
  tsi, rsi, bop, mtf, ma30 recapture family, chandelier/atr trailing exits).
- paper_trader_multi.py + order_manager_multi.py for execution (5 pos, ATR trail, vol-target flag).
- Strict causal walk-forward (test_rule_scorecard.py): full entry+exit strategy via shared
  PortfolioEngine, PANEL metrics (ret/effSR/DD/calmar/win%), PAIRED exact sign test.

Verified: ATR trailing (trend-gated) gives lift on 90d OOS vs no-trail.
CCI is the best of the bounded-oscillator (mean-reversion) chop family tested on
68 coins / 8 WF slices: cci +29.2% (effSR +2.16, 33.6% win) vs d40 +6.5%, bop +36.1%,
tsi +26.5%, rei +24.5%. New CCI-family members (Stochastic, MFI, IFT-RSI) did NOT beat
cci: stochastic +5.8%, mfi +4.3%, ift_rsi +8.0% — all with negative entry contribution
(-5 to -7) vs cci's +14.1. MFI (volume-weighted) added nothing vs price-based cci.
REI is the only entry rule significant vs d40 (p=0.031). No chop rule distinguishable
from d40 at p<0.05 except rei; cci is best non-significant.
Live entry: chop->cci (+ma30_ema silent fill), trend->rei, in engine.py + config.py.

Tests: green (35 passed, 4 skipped). Live trader RUNNING on final config.
See live_candidates.md for the full scorecard and decisions.
Previous BTC ML work is archived; current is simple falsifiable rules + regime.

---
# Roadmap — BTC/USDT ML Trading Bot

## Current State: Validated Baseline

**What we built:**
- 1.25M 5-min BTC bars (2012–2025) via backfill
- 55 features: 5m technicals + 1h/4h resampled + macro proxies + full temporal (Silva et al.)
- Triple-barrier labels, symmetric 0.4%
- 27-fold expanding walk-forward validation
- Bysik cost-aware filter
- Execution simulator with slippage/fees/timing

**Validated results:**
| Config | Accuracy | F1-macro | Sharpe net | Verdict |
|--------|----------|----------|------------|---------|
| 39 features (baseline) | 0.530 | 0.444 | negative | signal too weak |
| 55 features (+full temporal) | 0.543 | 0.459 | negative | marginal improvement |
| 55 features + ATR barriers | 0.452 | 0.324 | negative | label balance broken |
| 56 features + regime detection | 0.543 | 0.459 | negative | no value added |
| Regime-conditioned models | 0.450 | 0.330 | negative | worse than global |

**Honest finding:** Public-price technical + macro features on BTC 5-min bars cannot produce
profitable directional signals after transaction costs. The literature ROC-AUC ≈ 0.60 ceiling is real.

---

## Research Reference Documents

Create `/home/lars/trading-bot/RESEARCH_SUMMARY.md` from the papers we've reviewed:
- Sobreiro et al. 2026 (multi-timeframe, 37 features, ROC-AUC 0.609)
- Bysik 2026 (execution filter)
- Grądzki 2025 (triple-barrier labels, ATR-normalized barriers)
- Silva et al. 2025 (temporal features: year/month/weekday/hour/minute/sin-cos)
- Oprea/Bâra 2026 (regime stability, vol-aware normalization)

## Phase 1: Low-Hanging Feature Optimization
- [x] Add full temporal features (Silva et al.): year, month, day_of_year, weekday,
      hour, minute, hour_sin/cos, month_sin/cos, weekday_sin/cos,
      is_weekend, is_month_start/end, is_quarter_start/end
- [x] Add multi-timeframe features: 1h, 4h resampled
- [x] Add macro proxy features: SPY, GLD, TLT, UUP, VIX + BTC ratios
- [x] Grądzki ATR-normalized barriers → tested, rejected (breaks label balance)
- [x] Vol-aware feature scaling (Oprea/Bâra) → tested, rejected (unnecessary for trees)
- [x] Regime-conditioned models → tested, rejected (worse than global)

## Phase 2: Execution Infrastructure (build regardless of signal)
This is where we have real optionality — execution code is reusable across any signal.

1. **Live data feeders**
   - Kraken spot REST for BTC/USDT (public, no VPN needed)
   - CCXT normalization for future exchange support (Binance, Bybit, MEXC)
   - Hourly backfill keep-alive to maintain warm feature windows

2. **Real-time feature pipeline**
   - Same derive_features() but streaming
   - Rolling-window computation on live bars
   - Triple-barrier in progress tracking (labels formed in real-time)

3. **Model serving**
   - Weekly retrain from scratch on expanding window
   - Serialize XGBoost to JSON or ONNX
   - Inference server: accept latest feature row, output position signal
   - Fallback to last known prediction if data stale

4. **Order management**
   - Exchange REST API integration for order placement
   - Limit orders with fallback to market
   - Position tracking and reconciliation
   - Circuit breakers: max daily loss, max drawdown, kill-switch

5. **Monitoring & logging**
   - Trade journal: timestamp, side, size, fill price, fees
   - Prediction log: probabilities, selected action, regime
   - Daily summary: PnL, win rate, Sharpe rolling

## Phase 3: Structural Edge Data Collection
Microstructure features that actually have signal:

1. **Funding rates** (highest priority)
   - Binance USDT-margined futures: `fundingRate` endpoint
   - Collect every 8h, build 6-month history
   - Derive: funding basis (funding vs spot premium), funding momentum, funding z-score

2. **Liquidations**
   - Bybit/MEXC liquidation endpoints
   - Long vs short liquidation imbalance = forced-flowage indicator

3. **Order book depth**
   - Top-10 bid/ask imbalance
   - Depth slope (distance from mid to cumulative $X depth)

4. **Cross-exchange basis**
   - BTC/USDT price on Kraken vs Coinbase vs Binance
   - Basis convergence/divergence signals

## Phase 4: Regime-Constrained Strategy
- Restrict trading to high_vol_trend regime (highest val_acc = 0.476)
- VIX filter: reduce position size when VIX > 20-day MA
- Circuit breaker: halt trading during flash crashes (5-bar drawdown > 3σ)

## Phase 5: Live Trading (cautious)
- Paper trading on Kraken spot first
- Start with $100, track against buy-and-hold benchmark
- Scale to real only after 90 days positive alpha
- Max position: 10% equity per trade initially

---

## Realistic Expectations

- Signal ceiling for public-price features: ROC-AUC ≈ 0.60 (confirmed by literature + our validation)
- Net Sharpe with microstructure edge: maybe 0.8-1.2 if funding capture works
- Timeline: 6-12 months of data collection before microstructure features are useful
- This is a long-term infrastructure project, not a quick profit
