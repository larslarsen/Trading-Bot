#!/usr/bin/env python3
"""Train per-pair single XGBoost models for the literature-screened universe.

Loops model_trainer.train_and_save(symbol) over every pair in
universe_gated.json (Bartolucci quality gate: maturity + liquidity). Writes
each to models/<SYM>_xgb.json (BTC -> latest_xgb.json), exactly as
retrain_all.py does, so the live screener bot auto-discovers and trades them.

Run AFTER quality_gate has produced universe_gated.json (the gated pooled
trainer writes it; or run `python quality_gate.py` first).

Usage:
  python retrain_gated.py
"""
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import model_trainer as mt

REPO = Path(__file__).parent
GATE_JSON = REPO / "universe_gated.json"


def main():
    if not GATE_JSON.exists():
        # fall back to generating the gate on the fly
        from quality_gate import gated_universe
        pairs = gated_universe()
    else:
        pairs = json.loads(GATE_JSON.read_text())["pairs"]
    # model_trainer uses fsym convention: BTCUSDT -> "BTC", else the pair
    syms = ["BTC" if p == "BTCUSDT" else p for p in pairs]
    syms = list(dict.fromkeys(syms))
    print(f"[retrain_gated] {len(syms)} pairs from screened universe")
    t0 = time.time()
    ok = 0
    for s in syms:
        try:
            r = mt.train_and_save(symbol=s)
            if r:
                ok += 1
        except Exception as e:
            print(f"  {s} FAILED: {e!r}")
    print(f"\n[retrain_gated] done. {ok}/{len(syms)} trained in "
          f"{time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
