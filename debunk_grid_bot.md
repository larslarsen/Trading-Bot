# Why "Average Down Into Infinity" Is Mathematically Impossible

Your claim: with a large enough balance, you can keep averaging a losing
position down forever. On BloFin futures, 2-3x leverage.

Three independent proofs, all from real market data.

---

## PROOF 1 — Every entry signal still ruins under martingale/grid

Same trades, same market, only the position sizing changes. "Fixed" = sane
20%-risk-per-trade. "mart/grid" = doubling down / averaging down.

| signal | win% | fixed | mart_cap | mart_uncap | grid | ruined? |
|---|---|---|---|---|---|---|
| donchian40 | 19% | +3% | -4% | -4% | 0% | YES |
| ma30_ema | 17% | +920% | -20% | -256% | +32% | YES |
| cci | 21% | +2919% | -116% | -116% | +48% | YES |
| tsi | 25% | +1562% | -9929% | -9929% | +373% | YES |
| rsi | 61% | +48280% | -729% | -512% | +253260% | YES |
| mtf | 23% | +639% | 0% | -235% | +808% | YES |
| ma30 | 17% | +965% | -1% | -7% | +33% | YES |
| williams_r | 55% | +1442% | -652% | -2472% | +251% | YES |
| rei | 27% | +3857% | -960% | -6842% | +71% | YES |
| bbwp | 33% | +84% | -21% | -21% | +9% | YES |
| stochastic | 55% | +1428% | -646% | -2448% | +248% | YES |
| mfi | 56% | +924195% | -1.19M% | -1.19M% | +3.6M% | YES |
| ift_rsi | 65% | +54371% | -5664% | -5664% | +57957% | YES |
| d40+ma30_ema | 24% | +19% | -8% | -8% | 0% | YES |

Even rsi/mfi/stochastic at 55-65% win rate (your "long win streaks" profile)
make +48,000% to +924,000% with sane sizing — and go to NEGATIVE equity
under martingale. The entries are fine. The sizing kills it.

---

## PROOF 2 — A bigger account buys you NOTHING

Under uncapped averaging-down, bet doubles each loss. Consecutive losses to
wipe the account = log2(balance / base_bet + 1). Base bet = 20% of balance,
so this cancels to ~3 REGARDLESS of account size:

| account | consecutive losses to RUIN |
|---|---|
| $1,000 | 3 |
| $1,000,000 | 3 |
| $1,000,000,000 | 3 |

Going from $1k to $1B adds zero extra losses of survival. You need an
INFINITELY large account to survive an infinite streak — which is the
contradiction in "average down into infinity."

---

## PROOF 3 — On BloFin futures, the exchange liquidates you first

Liquidation price of a blended (averaged) long = Pav × (1 − 1/L).
Averaging down does NOT widen this band — the liquidation price tracks the
average entry, so each add moves the cliff with you. The exchange stops you
at 1/L adverse (2x = 50%, 3x = 33%).

Simulated on 7,000+ real CEX price paths (grid enters on RSI<35, adds on
dips, liquidates at 1/L):

| leverage | avg-down step | grids | liquidated | take-profit | adds-to-liq (median) | adverse% @ liq |
|---|---|---|---|---|---|---|
| 2x | 5% | 7042 | 134 | 6908 | 10 | 51.9% |
| 2x | 10% | 7013 | 170 | 6843 | 6 | 51.7% |
| 3x | 5% | 7060 | 288 | 6772 | 6 | 35.7% |
| 3x | 10% | 7035 | 364 | 6671 | 4 | 35.4% |

98-99% of grids hit take-profit (the win streaks you see). But the 1-2% that
liquidate do so at the FULL 1/L band, wiping the entire grid. One liquidation
erases every win streak. The exchange, not you, decides when it ends.

---

## Bottom line

"Average down into infinity" requires surviving infinite adverse move.
The exchange liquidates at 33-50% adverse. Infinity ≠ 50%.
No balance size, no "safeguard," no win streak changes this.
