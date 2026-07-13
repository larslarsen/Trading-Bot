# Research Summary — BTC/USDT ML Trading

## Papers Reviewed

### 1. Sobreiro et al. (2026)
- **Title:** Multi-Timeframe Feature Engineering for Bitcoin Market Prediction
- **Journal:** Forecasting 2026, 8(3), 40
- **Data:** BTC/USDT 2020-2025, 4 timeframes (15m, 4h, daily, 3-day)
- **Features:** 37 price-agnostic technical indicators (RSI, StochRSI, MACD, BB, ATR, EMA, volume)
- **Labels:** Binary classification via majority-vote across 54 TP/SL combos
- **Validation:** Expanding-window TimeSeriesSplit, 5 folds
- **Results:** ROC-AUC 0.609 (RF best), Sharpe 0.14 gross, negligible net of fees
- **Key finding:** Models with AUC≈0.60 cannot generate significant returns after fees
- **Source:** https://www.mdpi.com/2571-9394/8/3/40

### 2. Silva et al. (2025)
- **Title:** A Novel Cryptocurrency Trend Prediction Framework Powered by Innovative Feature Engineering
- **Journal:** IEEE Access
- **Contribution:** Temporal feature set for crypto ML
- **Features:** year, month, day_of_year, weekday, hour, minute, hour_sin/cos, month_sin/cos, weekday_sin/cos, is_weekend, is_month_start/end, is_quarter_start/end, quarter
- **Impact:** +1.3pp accuracy in our validation (0.530 → 0.543)

### 3. Grądzki (2025)
- **Title:** Triple-barrier labeling variants for financial ML
- **Focus:** Adaptive barriers based on ATR instead of fixed percentage
- **Our finding:** ATR barriers break label balance on BTC because low-vol periods have fewer flats

### 4. Bysik (2026)
- **Title:** Cost-aware execution filter for ML trading signals
- **Focus:** Lambda filter: trade only when confidence exceeds λ × cost threshold
- **Our finding:** Filter is flat across λ because model margins are uniformly weak

### 5. Oprea/Bâra (2026)
- **Title:** Feature stability across market regimes
- **Focus:** Feature importance shifts, regime-aware modeling
- **Our finding:** Regime-conditioned models (4 regimes) underperform global model due to insufficient data per regime

## Validated Experiment Results

| Config | Features | Accuracy | F1-macro | Regimes | Verdict |
|--------|----------|----------|----------|---------|---------|
| Baseline XGB | 21 | 0.530 | 0.444 | single | Weak signal |
| + temporal (Silva) | 55 | 0.543 | 0.459 | single | Marginal gain |
| + ATR barriers | 55 | 0.452 | 0.324 | single | Broke balance |
| + vol scaling | 55 | ~0.54 | ~0.46 | single | No improvement |
| + regime features | 56 | 0.543 | 0.459 | single | No value added |
| Regime-conditioned | 56 | 0.450 | 0.330 | 4 regimes | Worse than global |
| Model bake-off* | 54 | 0.452 | 0.320-0.367 | single | XGB/LGB similar |

*Bake-off used ATR barriers, fixed to 200k subsample, 3 folds

## Honest Conclusions

1. **Signal ceiling:** Public-price technical features on BTC 5-min bars: ROC-AUC ≈ 0.55-0.60
2. **Regime split:** Not viable with current data — per-regime models starve
3. **ATR barriers:** Break label balance on BTC; only useful for asymmetric label schemes
4. **Execution:** No realistic execution config turns 54% accuracy into profit
5. **Model choice:** XGBoost, LightGBM, Random Forest all equivalent within variance

## Path Forward

### Recommended: Infrastructure + Data Collection

**Phase 1:** Build execution infrastructure
- Live data feeder (CCXT)
- Rolling feature computation
- Model serving (weekly retrain, serialize to JSON)
- Order management + circuit breakers

**Phase 2:** Collect microstructure data (6-12 months)
- Funding rates (Binance fapi)
- Liquidations (Bybit/MEXC)
- Order book depth
- Cross-exchange basis spreads

**Phase 3:** Retrain with structural edge features
- Funding basis signals
- Liquidation imbalance
- Order book imbalance
- USD liquidity measures

**Expected outcome:** Sharpe 0.8-1.2 if funding/liquidation features have signal
