#!/usr/bin/env python3
"""
Compare the multi-asset POOLED model vs the PER-PAIR models on the SAME gated
universe, using the established PIT walk-forward methodology
(test_rule_scorecard.py: load_common/prefix/run_strategy engine + sign_test_p).

Question (Bartolucci 2020 two-feature selection; Liu/Liang/Gitter 2019 negative
transfer): is a WEAK multi-asset model covering MANY coins better than STRONGER
per-pair models covering FEWER coins we can actually run at once?

Both strategies trade the SAME 32 gated coins, same PortfolioEngine config
(cost 8bp, slip 5bp, 5 pos, 20% size, 20% DD, 3% daily-loss), same WF slices ->
paired exact sign test on slice returns + panel (ret/effSR/DD/Calmar/win%).

A = multi-asset: one pooled model (cex_5m_pooled_xgb.json), P(LONG)>=thresh.
B = per-pair: 32 per-pair models (models/<sym>_xgb.json), canonical features.

CAVEAT (model FIXED): both models trained on full history; features are PIT-
truncated per slice, so this tests SIGNAL EFFICACY OOS, not a deployable
re-retrained strategy. Flagged, not hidden.
"""
import sys, json, time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).parent
sys.path.insert(0, str(REPO))

from test_rule_scorecard import (load_common, prefix, sign_test_p,
                                  TRAIN, STEP, OOS)
from portfolio_engine import PortfolioEngine, EngineConfig
import model_trainer as mt
from canonical_features import resolve
import xgboost as xgb

COST_BPS = 8.0 / 10000.0
SLIP_BPS = 5.0 / 10000.0
MAX_POS, POS_PCT = 5, 0.20
THRESH = 0.60  # P(LONG) entry threshold (matches live bot)

MODELS = REPO / "models"
POOLED_PATH = MODELS / "cex_5m_pooled_xgb.json"
POOLED_META = MODELS / "cex_5m_pooled_meta.json"


# ── PIT feature build (mirror model_trainer.build_symbol_features on a
#    provided truncated df, so no full-file re-fetch / lookahead) ──────────
def build_features_pit(df, symbol, canonical):
    """Replicate model_trainer.build_symbol_features pipeline on `df` (already
    PIT-truncated). canonical=True -> apply canonical 98-block resolve (per-pair
    models); False -> return model_trainer 113-set (pooled model)."""
    from pipeline import (add_resampled_features, add_macro_signals,
                          add_multi_asset_features, derive_features, detect_regime)
    from micro_features import load_micro, load_funding
    from onchain_features import load_onchain
    df = df.copy()
    if df.index.duplicated().any():
        df = df[~df.index.duplicated(keep="last")]
    tf = add_resampled_features(df); df = df.join(tf, how="left")
    macro = mt.load_macro_data(df.index); df = add_macro_signals(df, macro)
    multi_cols = []
    if mt.USE_MULTI_ASSET and Path(mt.MULTI_ASSET_FILE).exists():
        df, multi_cols = add_multi_asset_features(df, mt.MULTI_ASSET_FILE)
    micro = load_micro(df.index)
    if not micro.empty and micro.notna().any().any():
        df = df.join(micro, how="left")
    if "funding_rate" in df.columns:
        df = df.drop(columns=["funding_rate"])
    fund = load_funding(symbol, df.index)
    if fund is not None and not fund.empty:
        df = df.join(fund, how="left")
    oc = load_onchain(df.index, symbol)
    if oc is not None and not oc.empty:
        df = df.join(oc, how="left")
    try:
        from dex_features import add_dex_features
        df, dex_cols = add_dex_features(df)
    except Exception:
        dex_cols = []
    df = derive_features(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.sort_index()
    df = detect_regime(df)
    if canonical:
        df, feats = resolve(df)
    else:
        feats = [f for f in mt.ALL_FEATURES if f in df.columns] + multi_cols + dex_cols \
                + ["regime_high_vol", "regime_trending"]
        extra = ["funding_rate"] + [c for c in df.columns if c.startswith("oc_")]
        for c in extra:
            if c in df.columns:
                feats.append(c)
    return df, feats


def load_pooled():
    m = xgb.XGBClassifier(); m.load_model(str(POOLED_PATH))
    feats = json.loads(POOLED_META.read_text()).get("features")
    return m, feats


def load_perpair(gated):
    out = {}
    for p in gated:
        sym = p.replace("USDT", "")
        cand = [MODELS / f"{sym}_xgb.json", MODELS / f"{p}_xgb.json"]
        if p == "BTCUSDT":
            cand.append(MODELS / "latest_xgb.json")
        path = next((c for c in cand if c.exists()), None)
        if path is None:
            continue
        mm = xgb.XGBClassifier(); mm.load_model(str(path))
        out[p] = mm
    return out


def signals_multiasset(data, dates, gated, model, feat_cols):
    active_by_day = {}
    for day in dates:
        pre = prefix(data, day)
        if not pre:
            active_by_day[day] = []; continue
        act = []
        for p in gated:
            sym = "BTC" if p == "BTCUSDT" else p
            dfp = pre.get(sym) if sym in pre else pre.get(p)
            if dfp is None or len(dfp) < 50:
                continue
            try:
                df, feats = build_features_pit(dfp, sym, canonical=False)
            except Exception:
                continue
            if df.empty:
                continue
            fvec = pd.DataFrame(0.0, index=[df.index[-1]], columns=feat_cols)
            present = [c for c in feat_cols if c in df.columns]
            fvec[present] = df[present].iloc[[-1]].values
            fvec = fvec.fillna(0.0)
            X = np.nan_to_num(fvec.values, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                pl = float(model.predict_proba(X)[0][1])
            except Exception:
                continue
            if pl >= THRESH:
                act.append(p)
        active_by_day[day] = act
    return active_by_day


def signals_perpair(data, dates, perpair):
    gated = list(perpair.keys())
    active_by_day = {}
    for day in dates:
        pre = prefix(data, day)
        if not pre:
            active_by_day[day] = []; continue
        act = []
        for p in gated:
            sym = "BTC" if p == "BTCUSDT" else p
            dfp = pre.get(sym) if sym in pre else pre.get(p)
            if dfp is None or len(dfp) < 50:
                continue
            try:
                df, feats = build_features_pit(dfp, sym, canonical=True)
            except Exception:
                continue
            if df.empty:
                continue
            model = perpair[p]
            exp = getattr(model, "feature_names_in_", None)
            if exp is not None:
                miss = [c for c in exp if c not in df.columns]
                if miss:
                    continue
                fvec = df[list(exp)].tail(1).copy()
            else:
                fvec = df[feats].tail(1).copy()
            fvec = fvec.fillna(0.0)
            X = np.nan_to_num(fvec.values, nan=0.0, posinf=0.0, neginf=0.0)
            try:
                cls = int(model.predict(X)[0])
                conf = float(model.predict_proba(X)[0].max())
            except Exception:
                continue
            sig = {0: "SHORT", 1: "LONG", 2: "FLAT"}.get(cls, "FLAT")
            if sig == "LONG" and conf >= THRESH:
                act.append(p)
        active_by_day[day] = act
    return active_by_day


def score_signals(active_by_day, data, dates):
    cfg = EngineConfig(
        initial_capital=10000.0, max_daily_loss_pct=0.03, max_drawdown_pct=0.20,
        max_positions=MAX_POS, max_position_pct=POS_PCT, min_equity_to_trade=100.0,
        flash_crash_bars=5, flash_crash_pct=0.50, extreme_move_pct=0.90,
        cost_bps=COST_BPS, slippage_bps=SLIP_BPS, enable_vol_target=False,
    )
    eng = PortfolioEngine(cfg)
    eq = []
    for day in dates:
        pre = prefix(data, day)
        if not pre:
            eq.append(eng.equity); continue
        prices = {s: float(pre[s]["close"].iloc[-1]) for s in pre if not pre[s].empty}
        eng.start_daily_bar(next(iter(prices.values()), None))
        ok, _ = eng.check_circuit_breakers()
        if not ok:
            eng.flatten_all(prices); eq.append(eng.equity); continue
        active = active_by_day.get(day, [])
        for s in list(eng.positions.keys()):
            if s not in active:
                px = prices.get(s)
                if px and px > 0:
                    eng.close_position(s, px)
        for s in active:
            if s in eng.positions or len(eng.positions) >= MAX_POS:
                continue
            px = prices.get(s)
            if not px or px <= 0:
                continue
            ok, _ = eng.check_circuit_breakers()
            if not ok:
                break
            eng.open_position(s, px, eng.equity * POS_PCT)
        eq.append(eng.mark_to_market(prices))
    eq = pd.Series(eq)
    rets = eq.pct_change().dropna()
    ret = (eq.iloc[-1] / 10000.0 - 1) * 100
    sr = (rets.mean() / rets.std() * np.sqrt(252)) if len(rets) > 1 and rets.std() > 0 else 0.0
    in_mkt = (eq.diff() != 0).astype(int)
    exp = in_mkt.mean() if len(in_mkt) else 0.0
    eff_sr = sr * (exp ** 0.5) if exp > 0 else 0.0
    peak = 10000.0; mdd = 0.0
    for e in eq:
        peak = max(peak, e); mdd = max(mdd, (peak - e) / peak)
    calmar = (ret / 100.0) / mdd if mdd > 0 else 0.0
    wins = sum(1 for t in eng.trades if t.get("pnl", 0) > 0)
    winrate = (wins / len(eng.trades) * 100) if eng.trades else 0.0
    return {"ret": ret, "effSR": eff_sr, "maxDD": mdd * 100, "calmar": calmar,
            "win%": winrate, "trades": len(eng.trades), "exp%": exp * 100}


def main():
    t0 = time.time()
    pooled, feat_cols = load_pooled()
    # gated universe == pooled model's training pairs (the literature gate)
    gated = json.loads(POOLED_META.read_text())["pairs"]
    perpair = load_perpair(gated)
    print(f"GATED universe: {len(gated)} coins")
    print(f"Per-pair models available: {len(perpair)}/{len(gated)}")
    print(f"Multi-asset: 1 pooled model over all {len(gated)}")
    print(f"CAVEAT: models FIXED (trained on full history); features PIT-truncated "
          f"per slice -> tests signal efficacy OOS, not deployable re-retrained strategy.\n")

    data0, dates = load_common(n_min=150)
    slices = []
    i = TRAIN
    while i + OOS <= len(dates):
        slices.append(dates[i:i + OOS]); i += STEP
    print(f"WF slices: {len(slices)} (TRAIN={TRAIN} STEP={STEP} OOS={OOS}, PIT)\n")

    A, B = [], []
    for k, seg in enumerate(slices):
        sdata = load_common(n_min=150, as_of=str(seg[0].date()))[0]
        sdates = pd.DatetimeIndex(sorted(
            set().union(*[set(sdata[s].index) for s in sdata]))) if sdata else seg
        # use seg's own daily grid for consistency with scorer
        actA = signals_multiasset(sdata, seg, gated, pooled, feat_cols)
        actB = signals_perpair(sdata, seg, perpair)
        A.append(score_signals(actA, sdata, seg))
        B.append(score_signals(actB, sdata, seg))
        if (k + 1) % 5 == 0 or k == len(slices) - 1:
            print(f"  slice {k+1}/{len(slices)} done")

    def agg(rs, name):
        print(f"\n=== {name} (n={len(rs)} slices) ===")
        print(f"  meanRet={np.mean([r['ret'] for r in rs]):+6.2f}%  "
              f"worst={min(r['ret'] for r in rs):+6.2f}%  "
              f"pos={sum(1 for r in rs if r['ret']>0)}/{len(rs)}")
        print(f"  meanEffSR={np.mean([r['effSR'] for r in rs]):+6.2f}  "
              f"meanDD={np.mean([r['maxDD'] for r in rs]):.2f}%  "
              f"meanCalmar={np.mean([r['calmar'] for r in rs]):+6.2f}  "
              f"meanWin={np.mean([r['win%'] for r in rs]):.1f}%  "
              f"meanExp={np.mean([r['exp%'] for r in rs]):.1f}%")

    agg(A, "A = MULTI-ASSET pooled (weak, all 32 coins)")
    agg(B, "B = PER-PAIR (stronger, per-coin)")

    print("\n=== PAIRED exact sign test (A vs B) ===")
    for metric in ("ret", "effSR"):
        av = [r[metric] for r in A]; bv = [r[metric] for r in B]
        p = sign_test_p(av, bv)
        better = sum(1 for a, b in zip(av, bv) if a > b)
        d = np.mean(av) - np.mean(bv)
        star = " *" if p < 0.05 else ""
        print(f"  {metric:6s}: Δmean={d:+6.2f}  A beats B in {better}/{len(slices)} slices  "
              f"p={p:.3f}{star}")
    print("\n* = p<0.05 exact binomial sign test, paired by slice.")
    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
