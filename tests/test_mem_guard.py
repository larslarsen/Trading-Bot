"""Tests for mem_guard.py — RSS read + abort guard."""
import sys
from pathlib import Path

import pytest

import mem_guard as mg


def test_rss_mb_returns_float():
    v = mg.rss_mb()
    assert isinstance(v, float)
    assert v >= 0.0


def test_guard_noop_when_zero():
    # limit 0 disables; must not exit
    mg.guard(0)


def test_guard_noop_under_limit(monkeypatch):
    monkeypatch.setattr(mg, "rss_mb", lambda: 10.0)
    mg.guard(1000)  # under limit -> no exit


def test_guard_aborts_over_limit(monkeypatch):
    monkeypatch.setattr(mg, "rss_mb", lambda: 99999.0)
    with pytest.raises(SystemExit) as ei:
        mg.guard(100)
    assert ei.value.code == 0
