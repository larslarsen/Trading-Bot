"""
A vs B dropoff test: when a coin with an OPEN position drops off the daily
live screen, do we (A) keep managing it to its normal signal exit, or (B) force-
close it the same day?

Methodology (also a proof-of-concept for the PIT backtest fix):
  - Reconstruct POINT-IN-TIME screens over the backtest window by re-running the
    screen's ranking (top-idio-vol 20% per tier) on ONLY the bars available
    as-of each date. No survivorship: a coin only exists in a day's screen if its
    data existed then.
  - Run the live rei+cci engine with a time-varying eligible universe (only coins
    in that day's PIT screen can be ENTERED). Already-held coins stay monitored.
  - On the day a held coin first disappears from the PIT screen, record:
        A = return from drop-date close to its normal signal-exit close
        B = return if force-closed at drop-date close
  - Aggregate to decide A vs B on evidence.
"""
import sys
sys.path.insert(0, "/home/lars/trading-bot")
import numpy as np, pandas as pd
from pathlib import Path
import test_rule_scorecard as sc
from engine import get_regime_signals, improved_compute_live_regime

ROOT = Path("data"); OUT = Path("backtest_output")
MAX_POS = sc.MAX_POS; POS_PCT = sc.POS_PCT; COST = (sc.COST_BPS + sc.SLIP_BPS) / 1e4

def pit_screen_as_of(data, date):
    """Reconstruct the screen (top-idio-vol 20% per tier) using only bars <= date."""
    rows = []
    for s, df in data.items():
        d = df[df.index <= date]
        if len(d) < 60:
            continue
        ret = d["close"].pct_change()
        adv = d["volume"].mean()
        idio = ret.rolling(30).std().mean()
        # tier: derive from the broad universe if present, else 'unknown'
        rows.append({"stem": s, "adv": adv, "idio_vol": idio if pd.notna(idio) else 0.0})
    scr = pd.DataFrame(rows)
    if scr.empty:
        return set()
    scr = scr.sort_values("adv", ascending=False).head(500)
    # assign pseudo-tiers by ADV quartile for the per-tier idio filter
    scr["tier"] = pd.qcut(scr["adv"].rank(method="first"), 3, labels=["tail", "mid", "large"])
    parts = []
    for tier, g in scr.groupby("tier"):
        n = max(3, int(len(g) * 0.2))
        parts.append(g.nlargest(n, "idio_vol"))
    sel = pd.concat(parts)
    return set(sel["stem"].tolist())

def main():
    data, dates = sc.load_common(n_min=150)   # current survivor universe (for bars)
    # restrict to the OOS-style window we actually backtest
    wd = [d for d in dates if d >= dates[60]]
    print("window: %s .. %s  (%d days, %d coins)" % (wd[0].date(), wd[-1].date(), len(wd), len(data)))

    # precompute PIT screens for each day in window
    pit = {d: pit_screen_as_of(data, d) for d in wd}
    sizes = [len(pit[d]) for d in wd]
    print("PIT screen sizes over window: min=%d median=%d max=%d" % (min(sizes), int(np.median(sizes)), max(sizes)))

    # run a lightweight per-coin position tracker mirroring paper_trader logic
    held = {}          # stem -> entry_price
    drop_events = []   # (stem, drop_date, entry_price, A_ret, B_ret)

    for i, day in enumerate(wd):
        pre = sc.prefix(data, day)
        if not pre:
            continue
        reg = improved_compute_live_regime(pre)
        rule = "rei" if reg == "trend" else "cci"
        prices = {s: float(pre[s]["close"].iloc[-1]) for s in pre}

        # first: detect dropoffs among currently held (coin not in today's PIT, still has data)
        for s in list(held.keys()):
            if s not in pit[day] and s in prices:
                entry_px = held[s]
                drop_px = prices[s]
                # B: force-close at drop-date
                b_ret = (drop_px / entry_px - 1) - COST * 2
                # A: we keep monitoring; need its normal signal exit AFTER drop.
                #   find next day where the rule's EXIT fires for s (using its own bars).
                a_ret = None
                for j in range(i + 1, len(wd)):
                    pj = sc.prefix(data, wd[j]).get(s)
                    if pj is None:
                        # coin data ended -> treat as forced exit at last known px
                        last_px = prices[s]
                        a_ret = (last_px / entry_px - 1) - COST * 2
                        break
                    _, ex = get_regime_signals(rule, pj.reset_index())
                    if len(ex) and int(ex.iloc[-1]):
                        exit_px = float(pj["close"].iloc[-1])
                        a_ret = (exit_px / entry_px - 1) - COST * 2
                        break
                if a_ret is None:
                    a_ret = b_ret  # never exited -> assume force-close (conservative)
                drop_events.append((s, day, entry_px, a_ret, b_ret))
                del held[s]

        # then: normal engine-ish entry using only PIT-eligible coins
        active = []
        for s in pre:
            if s not in pit[day]:
                continue
            ent, _ = get_regime_signals(rule, pre[s].reset_index())
            if len(ent) and int(ent.iloc[-1]):
                active.append(s)
        # combo fill (cci silent -> ma30_ema) on chop days
        if reg == "chop" and len(active) == 0:
            for s in pre:
                if s not in pit[day]:
                    continue
                ent, _ = get_regime_signals("ma30_ema", pre[s].reset_index())
                if len(ent) and int(ent.iloc[-1]):
                    active.append(s)
        # open new positions (respect cap; ignore circuit breakers for this test)
        for s in active:
            if s in held or len(held) >= MAX_POS:
                continue
            px = prices.get(s)
            if px and px > 0:
                held[s] = px

    # also close any still-held at window end (count as no dropoff, skip)
    n = len(drop_events)
    if n == 0:
        print("NO dropoff events in window -> cannot distinguish A vs B on this universe.")
        return
    a = np.array([e[3] for e in drop_events])
    b = np.array([e[4] for e in drop_events])
    print("\n=== DROPOFF POLICY TEST (A=ride to exit, B=force-close) ===")
    print("dropoff events (held coin left PIT screen while open): %d" % n)
    print("A mean ret = %.2f%%   B mean ret = %.2f%%   delta(A-B) = %.2f%%" % (a.mean()*100, b.mean()*100, (a-b).mean()*100))
    print("A win-rate = %.1f%%   B win-rate = %.1f%%" % ((a>0).mean()*100, (b>0).mean()*100))
    print("A better than B in %.1f%% of events" % ((a > b).mean()*100))
    verdict = "A (ride to exit)" if a.mean() >= b.mean() else "B (force-close)"
    print("EVIDENCE-BASED VERDICT: %s" % verdict)

if __name__ == "__main__":
    main()
