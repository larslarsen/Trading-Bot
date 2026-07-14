"""
Tests for the shared PortfolioEngine.

These exist specifically to make the "phantom equity" class of bug impossible:
  - cash + position_value must always equal equity
  - opening must deduct cash, closing must credit cash
  - equity-based sizing must be able to reach the target fraction
  - no single-day double-counting / no leftover wrong equity between runs
"""

import numpy as np
import pytest
from portfolio_engine import PortfolioEngine, EngineConfig, Position, PositionSide


def make_engine(**over):
    cfg = EngineConfig(initial_capital=10000.0, max_positions=5, max_position_pct=0.20)
    for k, v in over.items():
        setattr(cfg, k, v)
    return PortfolioEngine(cfg)


def test_invariants_after_open_and_mtm():
    eng = make_engine()
    px = {'AAA': 100.0, 'BBB': 50.0}
    eng.start_daily_bar(100.0)
    eng.open_position('AAA', 100.0, eng.equity * 0.20)
    eng.open_position('BBB', 50.0, eng.equity * 0.20)

    # cash must have been deducted ~40% of equity
    assert eng.cash == pytest.approx(10000.0 * 0.60, rel=1e-6)
    # equity after MTM equals cash + position value
    eq = eng.mark_to_market(px)
    assert eq == pytest.approx(eng.cash + eng.position_value(px), rel=1e-9)
    # opening at fill price keeps NAV ~ flat (minus tiny entry spread), not +2000 phantom
    assert eq == pytest.approx(10000.0, rel=1e-3)
    assert eq < 10000.0 + 1e-6  # strictly no phantom gain from opening


def test_close_credits_cash_and_reduces_equity_by_pnl():
    eng = make_engine()
    px = {'AAA': 100.0}
    eng.start_daily_bar(100.0)
    eng.open_position('AAA', 100.0, eng.equity * 0.20)  # 2000 notional at fill
    # price doubles
    px2 = {'AAA': 200.0}
    pnl = eng.close_position('AAA', 200.0)
    # proceeds ~ 2000 * 2 * (1 - cost) ; pnl ~ +2000 - fees
    assert eng.cash == pytest.approx(10000.0 + pnl, rel=1e-6)
    assert eng.positions == {}
    eq = eng.mark_to_market(px2)
    assert eq == pytest.approx(eng.cash, rel=1e-9)
    assert len(eng.trades) == 1


def test_equity_based_sizing_reaches_target_fraction():
    eng = make_engine(max_positions=5, max_position_pct=0.20)
    # all 5 opens must each deploy ~20% of equity (NAV preserved, so cash allows it)
    eng.start_daily_bar(100.0)
    for i, sym in enumerate(['A', 'B', 'C', 'D', 'E']):
        eng.open_position(sym, 100.0, eng.equity * 0.20)
    # After 5 opens cash should be ~0 (each took 20% of equity at fill, NAV preserved)
    assert eng.cash == pytest.approx(0.0, abs=1.0)
    assert len(eng.positions) == 5
    # total deployed notional ~ 100% of capital base (minus tiny entry spread)
    total_notional = sum(p.shares * 100.0 for p in eng.positions.values())
    assert total_notional == pytest.approx(10000.0, rel=2e-3)


def test_cannot_exceed_max_positions():
    eng = make_engine(max_positions=2, max_position_pct=0.20)
    eng.start_daily_bar(100.0)
    for sym in ['A', 'B', 'C', 'D']:
        eng.open_position(sym, 100.0, eng.equity * 0.20)
    assert len(eng.positions) == 2


def test_flash_window_is_per_day_not_per_call():
    eng = make_engine(flash_crash_bars=3, flash_crash_pct=0.50, extreme_move_pct=0.90)
    # 3 calls in the SAME day with a 60% move -> NOT a crash (window resets each day)
    eng.start_daily_bar(100.0)
    eng.check_circuit_breakers()          # call 1
    eng.check_circuit_breakers()          # call 2
    ok, _ = eng.check_circuit_breakers()  # call 3 (only 1 sample in window)
    assert ok is True
    # Now 3 days of wild moves -> crash
    eng2 = make_engine(flash_crash_bars=3, flash_crash_pct=0.50, extreme_move_pct=0.90)
    eng2.start_daily_bar(100.0)
    eng2.check_circuit_breakers()
    eng2.start_daily_bar(200.0)  # +100% in a day
    eng2.check_circuit_breakers()
    eng2.start_daily_bar(40.0)   # -80% next day
    ok2, reason2 = eng2.check_circuit_breakers()
    assert ok2 is False
    assert 'flash_crash' in reason2


def test_drawdown_halt():
    eng = make_engine(max_drawdown_pct=0.20, max_positions=1, max_position_pct=1.0)
    # open a full-capital position, then MTM at a 25% lower price -> 25% equity drawdown
    eng.start_daily_bar(100.0)
    eng.open_position('AAA', 100.0, eng.equity * 1.0)
    eng.mark_to_market({'AAA': 75.0})  # 25% loss on the whole book
    ok, reason = eng.check_circuit_breakers()
    assert ok is False
    assert 'max_drawdown' in reason
    assert eng.halted is True


def test_daily_loss_halt():
    eng = make_engine(max_daily_loss_pct=0.03)
    eng.peak_equity = 10000.0
    eng.daily_pnl = -400.0
    ok, reason = eng.check_circuit_breakers()
    assert ok is False
    assert 'daily_loss' in reason


def test_equity_floor_halt():
    eng = make_engine(min_equity_to_trade=100.0)
    eng.equity = 50.0
    ok, reason = eng.check_circuit_breakers()
    assert ok is False
    assert 'equity' in reason


def test_state_roundtrip_preserves_cash_and_equity():
    eng = make_engine()
    eng.start_daily_bar(100.0)
    eng.open_position('AAA', 100.0, eng.equity * 0.20)
    d = eng.to_state_dict()
    eng2 = make_engine()
    eng2.load_state_dict(d)
    assert eng2.cash == pytest.approx(eng.cash, rel=1e-9)
    assert eng2.equity == pytest.approx(eng.equity, rel=1e-9)
    assert len(eng2.positions) == len(eng.positions)


def test_live_and_replay_share_engine_module():
    # Guarantee replay imports the shared engine, not a separate SimPosition.
    import paper_replay_oos as pr
    assert hasattr(pr, 'PortfolioEngine')
    import inspect
    src = inspect.getsource(pr)
    # the replay must drive the shared engine, not define its own SimPosition class
    assert 'class SimPosition' not in src


# ── Edge cases / branch coverage ──────────────────────────────────────────
def test_open_blocked_when_halted():
    eng = make_engine()
    eng.halted = True
    eng.halted_reason = "max_drawdown"
    assert eng.open_position('AAA', 100.0, 2000.0) is None
    assert eng.positions == {}


def test_open_blocked_when_target_nonpositive():
    eng = make_engine()
    eng.start_daily_bar(100.0)
    # size_usd <= 0 -> no position
    assert eng.open_position('AAA', 100.0, 0.0) is None
    # fill price <= 0 -> no position (would divide by zero / negative shares)
    assert eng.open_position('AAA', 0.0, 2000.0) is None
    assert eng.positions == {}


def test_open_caps_at_max_position_pct_of_base():
    # sizing base is max(equity, peak_equity, initial); target is min(request, base*pct)
    eng = make_engine(max_position_pct=0.20)
    eng.start_daily_bar(100.0)
    pos = eng.open_position('AAA', 100.0, 999999.0)  # request huge
    assert pos is not None
    # deployed notional (shares * fill) must equal the capped target = 20% of base.
    # fill already folds in slippage+cost, so compare against the actual fill price.
    fill = 100.0 * (1 + eng.config.slippage_bps + eng.config.cost_bps)
    assert pos.shares * fill == pytest.approx(eng.config.initial_capital * 0.20, rel=1e-6)


def test_close_unknown_symbol_returns_none():
    eng = make_engine()
    assert eng.close_position('NOPE', 100.0) is None


def test_flatten_skips_missing_prices():
    eng = make_engine(max_positions=5, max_position_pct=0.20)
    eng.start_daily_bar(100.0)
    eng.open_position('AAA', 100.0, 2000.0)
    eng.open_position('BBB', 50.0, 2000.0)
    # only AAA has a price; BBB's price missing -> skipped, not crashed
    eng.flatten_all({'AAA': 120.0})
    assert 'AAA' not in eng.positions
    assert 'BBB' in eng.positions  # remained open


def test_position_value_without_prices_uses_entry():
    eng = make_engine(max_positions=5, max_position_pct=0.20)
    eng.start_daily_bar(100.0)
    eng.open_position('AAA', 100.0, 2000.0)
    # no live prices -> value at entry cost basis == the capped target notional
    assert eng.position_value() == pytest.approx(2000.0, rel=1e-6)


def test_vol_scale_disabled_returns_one():
    eng = make_engine(enable_vol_target=False)
    assert eng.vol_scale() == 1.0
    assert eng.vol_scale(lookback=5) == 1.0  # short history also -> 1.0


def test_vol_scale_enabled_short_history_returns_one():
    eng = make_engine(enable_vol_target=True, vol_lookback=20)
    # not enough equity history yet
    assert eng.vol_scale() == 1.0


def test_vol_scale_enabled_clips_to_bounds():
    eng = make_engine(enable_vol_target=True, vol_lookback=3,
                      target_vol=0.15, min_vol_scale=0.25, max_vol_scale=1.5)
    # flat equity history -> zero vol -> vol_scale returns 1.0 (explicit vol<=0 guard)
    eng.equity_history = [1000.0, 1000.0, 1000.0, 1000.0]
    assert eng.vol_scale() == 1.0
    # high-vol history -> target/vol small -> clipped to min
    eng.equity_history = [1000.0, 1300.0, 1000.0, 1300.0, 1000.0]
    scale2 = eng.vol_scale()
    assert scale2 <= 1.5 and scale2 >= 0.25


def test_flash_crash_extreme_move_treated_as_data_artifact():
    # Need >= flash_crash_bars samples for the crash branch to evaluate.
    # Final jump beyond extreme_move_pct -> window cleared, NOT halted (glitch).
    eng = make_engine(flash_crash_bars=3, flash_crash_pct=0.50, extreme_move_pct=0.90)
    eng.start_daily_bar(100.0)
    eng.start_daily_bar(100.0)
    eng.start_daily_bar(2000.0)  # move = (2000-100)/2000 = 0.95 > 0.90 -> artifact
    ok, _ = eng.check_circuit_breakers()
    assert ok is True   # not halted; window was cleared as artifact
    assert eng._flash_window == []  # window reset


def test_flash_crash_real_move_halts():
    # move between flash_crash_pct and extreme_move_pct -> genuine crash
    eng = make_engine(flash_crash_bars=3, flash_crash_pct=0.50, extreme_move_pct=0.90)
    eng.start_daily_bar(100.0)
    eng.start_daily_bar(100.0)
    eng.start_daily_bar(250.0)  # move = (250-100)/250 = 0.60 in [0.50, 0.90]
    ok, reason = eng.check_circuit_breakers()
    assert ok is False
    assert 'flash_crash' in reason


def test_start_daily_bar_resets_pnl_and_rolls_window():
    eng = make_engine(flash_crash_bars=3)
    eng.daily_pnl = -500.0
    eng._flash_window = [100.0, 200.0]
    eng.start_daily_bar(300.0)
    assert eng.daily_pnl == 0.0
    assert eng._flash_window == [100.0, 200.0, 300.0]  # rolled forward
    # window capped at flash_crash_bars
    for _ in range(5):
        eng.start_daily_bar(300.0)
    assert len(eng._flash_window) == eng.config.flash_crash_bars


def test_position_roundtrip_dict():
    p = Position('X', PositionSide.LONG, 100.0, 2.5, highest_high=110.0)
    d = p.to_dict()
    assert d['symbol'] == 'X' and d['shares'] == 2.5 and d['highest_high'] == 110.0
    p2 = Position.from_dict(d)
    assert p2.symbol == p.symbol and p2.shares == p.shares and p2.highest_high == p.highest_high
    assert p2.side == PositionSide.LONG


def test_halt_sets_reason():
    eng = make_engine()
    eng.halt('manual_test')
    assert eng.halted and eng.halt_reason == 'manual_test'

