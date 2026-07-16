#!/usr/bin/env python3
"""Focused mean-reversion scorecard on CEX universe.

Reuses test_rule_scorecard's PIT walk-forward + paired sign test verbatim
(same engine, same panel: ret/effSR/maxDD/calmar/win%). Only the candidate
set changes: pure mean-reversion family (bounded oscillators), no trend rules.

Runs against the CEX universe (load_common -> data/ + latest screen_liqu_idio_*),
the established, longer-history dataset -- NOT the thin DEX set.
"""
import test_rule_scorecard as sc

MR = [
    ("rsi", "chop"),
    ("cci", "chop"),
    ("williams_r", "chop"),
    ("stochastic", "chop"),
    ("mfi", "chop"),
    ("bbwp", "chop"),
    ("ift_rsi", "chop"),
    ("bop", "chop"),
]

if __name__ == "__main__":
    # baseline = donchian40 (the LIVE rule), run the same way for paired compare
    sc.CANDIDATES = MR + [("donchian40", "chop")]
    sc.main()
