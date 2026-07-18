#!/usr/bin/env python3
"""
Gated-pooled 5m ML paper trader.

ONE daemon that trades the LITERATURE-SCREENED universe (Bartolucci 2020 two-
feature gate: SECURITY/maturity + LIQUIDITY/quote-volume) using the SINGLE
pooled multi-asset model we trained on that exact screened set
(models/cex_5m_pooled_xgb.json -- trained by train_cex_5m_pooled.py --gate).

This is deliberately SEPARATE from cex_ml_xgb_5m.py (the per-pair screener
bot): the pooled model was trained on model_trainer.build_symbol_features
(113-feature set), whereas the per-pair bot ALSO uses the same 113-feature
canonical block (model_trainer.build_symbol_features + canonical_features.resolve).
Different pooling (single shared model vs per-pair) -> different daemon, but the
feature dimension is identical.

Screener == quality_gate.gated_universe() (the literature gate). NOT the
xrank/Kraken or cex_multi_screen universes -- those are a different purpose.

Inference: build the last-row feature vector for every gated pair with the
exact training pipeline, run the shared pooled model, take P(LONG) as the
directional strength, rank top-N, trade long-only via order_manager_multi.

Usage:
  python cex_ml_pooled_5m.py            # loop forever, poll every 300s
  python cex_ml_pooled_5m.py --once     # one ranking + execution pass
"""
import argparse
import gc
import json
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import xgboost as xgb

from pipeline import fetch_data
from model_trainer import build_symbol_features
from order_manager_multi import MultiPositionState

REPO = Path(__file__).parent
MODELS_DIR = REPO / "models"
POOLED_MODEL = MODELS_DIR / "cex_5m_pooled_xgb.json"
POOLED_META = MODELS_DIR / "cex_5m_pooled_meta.json"

# Live daemon: THROTTLE (skip a pass), never hard-exit, so trading keeps running.
MEM_GUARD_MB = 6144  # above this RSS, skip the heavy build for one cycle

POLL_SEC = 300
STATE_FILE = REPO / "execution_state_ml_pooled.json"
JOURNAL_FILE = REPO / "trade_journal_ml_pooled.json"

# Live settings (mirror the per-pair bot / paper_trader_multi):
MAX_POSITIONS = 5            # hold top-5 by signal strength
CONFIDENCE_THRESHOLD = 0.60  # min P(LONG) to take a long
FEE_BP = 0.60
SLIPPAGE_BP = 5
SIZE_USD = 2000.0            # 20% of $10k equity per position
DD_HALT = 0.20               # flatten + halt at 20% drawdown


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    return 0


def mem_guard_abort(cap_mb):
    """Hard in-cycle backstop: abort this cycle's build if a single pass would
    blow the cap (catches between-cycle spikes that OOM-killed us before)."""
    if rss_mb() > cap_mb:
        raise RuntimeError(f"MEM GUARD: RSS={rss_mb()}MB > {cap_mb}MB abort cycle")


def gated_pairs():
    """The literature-screened universe (Bartolucci two-feature gate)."""
    from quality_gate import gated_universe
    return gated_universe()


def load_pooled():
    if not POOLED_MODEL.exists():
        raise RuntimeError(f"[pooled] FATAL: {POOLED_MODEL} missing -- train with train_cex_5m_pooled.py --gate")
    m = xgb.XGBClassifier()
    m.load_model(str(POOLED_MODEL))
    meta = json.loads(POOLED_META.read_text()) if POOLED_META.exists() else {}
    feats = meta.get("features")
    if feats is None:
        # fall back to the model's own recorded feature names
        feats = list(getattr(m, "feature_names_in_", []))
    print(f"[pooled] loaded {POOLED_MODEL.name}: n_features={getattr(m,'n_features_in_',None)} "
          f"universe={meta.get('n_pairs','?')} pairs")
    return m, feats


def fetch_symbol(pair):
    """BTCUSDT -> 'BTC' (root btc_5m.csv); everything else -> the pair."""
    return "BTC" if pair == "BTCUSDT" else pair


def last_feature_row(pair, feat_cols):
    """Build the exact training feature vector (last bar) for `pair`.

    Uses model_trainer.build_symbol_features -- the SAME pipeline the pooled
    model trained on. Aligns to the model's recorded feature order, 0-filling
    any column the pair lacks. Returns (fvec_df, last_close) or (None, None).
    """
    sym = fetch_symbol(pair)
    try:
        df, feats = build_symbol_features(sym)
    except Exception as e:
        print(f"[{datetime.now(timezone.utc).isoformat()}] {pair} build error: {e!r}"[:160])
        return None, None
    if df is None or df.empty:
        return None, None
    df = df.replace([np.inf, -np.inf], np.nan)
    # align to the model's feature order; 0-fill missing
    out = pd.DataFrame(0.0, index=[df.index[-1]], columns=feat_cols)
    present = [c for c in feat_cols if c in df.columns]
    out[present] = df[present].iloc[[-1]].values
    out = out.fillna(0.0)
    last_close = float(df["close"].iloc[-1]) if "close" in df.columns else None
    return out, last_close


def run_once(model, feat_cols, state, pairs):
    mem_guard_abort(4096)
    ranked = []
    for pair in pairs:
        try:
            fvec, last_close = last_feature_row(pair, feat_cols)
            if fvec is None or last_close is None:
                continue
            X = np.nan_to_num(fvec.values, nan=0.0, posinf=0.0, neginf=0.0)
            probs = model.predict_proba(X)[0]
            # multi:softmax classes 0=SHORT,1=LONG,2=FLAT (matches training labels)
            p_long = float(probs[1])
            ranked.append({"pair": pair, "signal": "LONG" if p_long >= CONFIDENCE_THRESHOLD else "FLAT",
                           "conf": p_long, "strength": p_long if p_long >= CONFIDENCE_THRESHOLD else 0.0})
        except Exception as e:
            print(f"[{datetime.now(timezone.utc).isoformat()}] {pair} predict error: {e!r}"[:160])
    ranked.sort(key=lambda r: -r["strength"])
    print(f"[{datetime.now(timezone.utc).isoformat()}] ranked(gated pooled): " +
          ", ".join(f"{r['pair']}={r['signal']}({r['conf']:.2f})" for r in ranked))

    desired = [r for r in ranked if r["signal"] != "FLAT" and r["conf"] >= CONFIDENCE_THRESHOLD][:MAX_POSITIONS]
    desired_pairs = {r["pair"]: r for r in desired}

    # close positions no longer desired
    for sym in list(state.positions.keys()):
        if sym not in desired_pairs:
            px = price_for(sym)
            if px:
                state.close_position(sym, px)
                print(f"CLOSE {sym} @ {px:.6f} (not in top-{MAX_POSITIONS})")

    # open top-N desired
    for r in desired:
        sym = r["pair"]
        if sym in state.positions:
            continue
        if len(state.positions) >= MAX_POSITIONS:
            break
        px = price_for(sym)
        if not px or px <= 0:
            continue
        ok, reason = state.check_circuit_breakers()
        if not ok:
            print(f"CIRCUIT BREAKER: {reason}, staying flat")
            break
        pos = state.open_position(sym, px, SIZE_USD)
        if pos:
            print(f"OPEN {sym} @ {px:.6f}, size=${SIZE_USD:.2f} (LONG conf={r['conf']:.2f})")

    # mark to market + save
    try:
        mtm = {r["pair"]: price_for(r["pair"]) for r in ranked}
        eq = state.mark_to_market({k: v for k, v in mtm.items() if v})
        dd = (state.peak_equity - eq) / state.peak_equity if state.peak_equity > 0 else 0
        print(f"MTM equity: ${eq:.2f}  Peak: ${state.peak_equity:.2f}  DD: {dd:.2%}  Positions: {len(state.positions)}")
        if dd > DD_HALT:
            print(f"Flattening: DD {dd:.2%} > {DD_HALT:.0%}")
            state.flatten_all({k: v for k, v in mtm.items() if v})
            state.halt(f"live_drawdown_flatten_{pd.Timestamp.now().isoformat()}")
    except Exception as e:
        print(f"MTM error: {e!r}"[:160])


def price_for(symbol):
    df = fetch_data(fetch_symbol(symbol))
    return float(df["close"].iloc[-1]) if not df.empty else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args()
    model, feat_cols = load_pooled()
    pairs = gated_pairs()
    print(f"[pooled] gated universe ({len(pairs)} pairs): {pairs}")
    state = MultiPositionState(initial_capital=10000.0, state_file=STATE_FILE, journal_file=JOURNAL_FILE)
    if args.once:
        run_once(model, feat_cols, state, pairs)
        return
    while True:
        rss = rss_mb()
        if rss > MEM_GUARD_MB:
            print(f"[{datetime.now(timezone.utc).isoformat()}] MEM THROTTLE: "
                  f"RSS={rss}MB > cap={MEM_GUARD_MB}MB -- skipping build, gc + cooldown")
            gc.collect()
            time.sleep(POLL_SEC + 60)
            continue
        try:
            run_once(model, feat_cols, state, pairs)
        except Exception as e:
            print(f"[{datetime.now(timezone.utc).isoformat()}] error: {e!r}"[:200])
        gc.collect()
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
