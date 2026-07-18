"""Tests for config.py — EngineConfig defaults + core-count detection."""
import os
from pathlib import Path

import pytest

import config as cfg


def test_engineconfig_defaults():
    c = cfg.EngineConfig()
    assert c.initial_capital == 10000.0
    assert c.max_daily_loss_pct == 0.03
    assert c.max_drawdown_pct == 0.20
    assert c.max_positions == 5
    assert c.max_position_pct == 0.20
    assert c.min_equity_to_trade == 100.0
    assert c.flash_crash_pct == 0.50
    assert c.cost_bps == 8.0 / 10000.0
    assert c.slippage_bps == 5.0 / 10000.0


def test_config_singleton_present():
    assert cfg.CONFIG is not None
    assert isinstance(cfg.CONFIG, cfg.EngineConfig)


def test_n_jobs_positive_and_leaves_headroom(monkeypatch):
    # force a known cpu count
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    # re-import detection path by calling the helper directly
    assert cfg._detect_logical_cores() == 8
    phys = cfg._detect_physical_cores()
    assert phys >= 1
    assert cfg.N_JOBS == max(1, phys - 1)
    assert cfg.N_JOBS >= 1


def test_n_jobs_env_override(monkeypatch):
    monkeypatch.setenv("TRADING_BOT_CORES", "4")
    # reload module to pick up env at import time
    import importlib
    importlib.reload(cfg)
    assert cfg.PHYSICAL_CORES == 4
    assert cfg.N_JOBS == max(1, 4 - 1)
    # cleanup env so other tests aren't affected
    monkeypatch.delenv("TRADING_BOT_CORES", raising=False)
    importlib.reload(cfg)
