# Papers #1 & #2 — Re-Architecture (with failures designed out + new data)

Context: original paper #1 result is VOID (flow cols 0 non-null; trained on
empty features). Paper #2 result (0.601 vs 0.449) STANDS but should be
re-confirmed with the guard + combined-volume bars. We also found new data:
  - data/FLOWUSDT_5m_max.csv  : market-wide AGGREGATE flow, 521k rows, 5m.
    This is the Anastasopoulos "world order flow" signal (aggregate, not
    per-coin) — the CORRECT data for paper #1, which the original eval
    never used (it read the missing trade_agg_5m.csv instead).
  - data/<SYM>USDT_5m_blofin_max.csv : 2nd-source (blofin) OHLCV, ALL pairs.
    -> per-coin volume proxy + combined-volume info bars for paper #2.
  - data/funding/, data/onchain/ : populated (see PAPER3_ARCHITECTURE.md).

## COMMON architecture (both papers)
Fork eval_flow_paper1.py (baseline vs treatment, PIT walk-forward 5-fold,
corrected k=max sign test). Add:

GUARD (paper #1's lesson — never train on empty features):
  For each pair, after building TREATMENT features, assert every TREATMENT
  col is >60% non-null over the pair history. If not -> SKIP pair, log
  reason ("FLOWUSDT missing" / "blofin <60% non-null" / etc). Report
  n_tested and n_skipped at end. No zero-fill, no silent "win".

VERDICT GATE (effect-size floor paper #1 lacked — BOTH required):
  (a) corrected exact sign test p(treatment > baseline) < 0.05
  (b) treatment mean OOS dir-acc > 0.50   (above coin-flip)
  Significance without effect-size = paper #1 trap. Reject if (b) fails.

OUTPUT per pair: "SYM: base=X treat=Y" + final block with n_tested/
n_skipped, wins/losses/ties, p, means, VERDICT (HELPFUL / not).

## PAPER #1 re-arch — order flow (VOID -> re-run on REAL flow)
Target features (TREATMENT), added to canonical baseline:
  - FLOWUSDT aggregate flow, time-aligned to each pair (global feature):
      net_flow = (buy_vol - sell_vol) or signed flow; flow_z = rolling
      z-score; flow_ma = EMA. (Derive from FLOWUSDT_5m_max.csv columns.)
  - Per-coin flow PROXY from blofin: vol_ratio = blofin_vol / binance_vol
      (relative activity / flow pressure per coin). Present for ALL pairs.
Baseline: canonical (113), NO flow.
Treatment: baseline + [net_flow, flow_z, flow_ma, vol_ratio].
Why this matches the paper: Anastasopoulos 2024 = WORLD/aggregate order
flow dominates -> FLOWUSDT (aggregate) is the right signal; vol_ratio is a
per-coin flow-pressure proxy from the 2nd venue we actually have.
Guard: FLOWUSDT present (yes); if a pair lacks blofin -> run with
FLOWUSDT-only, log "blofin absent for SYM".
Files: eval_paper1_v2.py (fork + flow builder + guard).
Expected: if flow carries signal, treatment > 0.50 and beats baseline
significantly. If still sub-0.50 -> genuinely no edge (correctly rejected,
not a spurious win).

## PAPER #2 re-arch — information-driven bars (STANDS -> enhance + re-confirm)
Original used Binance 5m volume for the volume-target (~60000 bars) ->
0.601 vs 0.449. Enhance with new data:
  - COMBINED volume for bar target: vol_target uses (binance_vol +
    blofin_vol) per pair -> "true" volume bars (both venues). More robust
    than Binance-only.
  - Cross-venue robustness check: also build info bars on blofin source
    alone (separate treatment) to see if the edge is venue-specific.
Baseline: triple_barrier on 5m time bars (Binance).
Treatment A: info bars on combined (binance+blofin) volume, ~60000 bars.
Treatment B (optional): info bars on blofin volume alone.
Guard: info-bar row count >= 35000 (walk_forward_splits min). If pair has
too little combined vol -> skip (log reason). Already fixed the ~60000
target; keep it.
Files: eval_paper2_v2.py (fork + combined-volume bar builder + guard).
Expected: re-confirm 0.601-class edge on combined volume; check if blofin
alone also works (cross-venue transfer, per Frontiers 2026).

## Run order
1. eval_paper1_v2.py  (paper #1, REAL flow now) — ALONE, watchdog, notify.
2. eval_paper2_v2.py  (paper #2, combined-volume enhance) — ALONE, notify.
Each ~12-16 min. No stacking.

## Decision needed
Approve running both v2 evals (paper #1 first — it's the void one). Or
say "paper #1 only" / "paper #2 only". On your go I implement the two
forks (reuse eval_flow_paper1.py shape) and run them.
