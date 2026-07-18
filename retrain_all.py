#!/usr/bin/env python3
"""Retrain ALL screener models on the ONE canonical 113-feature set.

Run ONCE after the new CPU cooler is installed (or anytime). Each pair trains
on an identical input dimension (canonical_features.CANONICAL) so the serving
bot's feature block matches every model exactly. Outputs land in models/<sym>_xgb.json
(BTC keeps latest_xgb.json). Safe to re-run.

Usage:
  python retrain_all.py              # BTC + all pairs in screener_ml_multi.txt
  python retrain_all.py --symbols BTC,ETH,DOGE
"""
import argparse, sys, warnings
warnings.filterwarnings("ignore")
import model_trainer as mt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=None,
                    help="comma list; default = BTC + screener pairs")
    args = ap.parse_args()
    if args.symbols:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        syms = ["BTC"]
        sp = mt.MODEL_DIR.parent / "screener_ml_multi.txt"
        if sp.exists():
            for line in sp.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    syms.append(line)
    syms = list(dict.fromkeys(syms))  # dedupe, keep order
    print(f"[retrain_all] pairs={syms}  canonical_n_features={mt_canonical_n()}")
    for s in syms:
        print(f"\n=== {s} ===")
        mt.train_and_save(symbol=s)
    print("\n[retrain_all] done.")

def mt_canonical_n():
    import canonical_features as cf
    return cf.N_FEATURES

if __name__ == "__main__":
    main()
