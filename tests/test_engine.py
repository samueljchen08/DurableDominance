"""The vectorised engine must agree with the readable reference implementation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np

from durable_dominance import engine, metrics, simulation


def test_win_probability_matches_sigmoid_form():
    for s1, s2, c in [(60, 40, 0.9), (50, 50, 0.75), (30, 70, 0.99)]:
        direct = simulation.win_probability(s1, s2, c)
        sigmoid = 1.0 / (1.0 + np.exp(-np.log(c) * (s1 - s2)))
        assert abs(direct - sigmoid) < 1e-12


def test_batch_matches_reference_distribution():
    """Same parameters, same distribution of ranked wins (not the same draws)."""
    t, m, s, c = 20, 40, 6.0, 0.9
    fixture = engine.Fixture.build(t, m)
    schedule = simulation.create_schedule(t, m)
    assert fixture.n_matches == len(schedule)

    rng = np.random.default_rng(1)
    z, u = engine.draw_noise(fixture, 3000, rng)
    fast = engine.ranked_wins_batch(fixture, z, u, s, c).mean(axis=0)

    rng2 = np.random.default_rng(2)
    slow = np.mean(
        [simulation.ranked_wins(t, m, s, c, schedule, rng2) for _ in range(3000)],
        axis=0,
    )
    assert np.allclose(fast, slow, atol=0.6)


def test_wins_conserved():
    """Every match produces exactly one win."""
    fixture = engine.Fixture.build(24, 30)
    rng = np.random.default_rng(3)
    z, u = engine.draw_noise(fixture, 50, rng)
    wins = engine.ranked_wins_batch(fixture, z, u, 5.0, 0.92)
    assert np.all(wins.sum(axis=1) == fixture.n_matches)


def test_effective_spread_degeneracy():
    """Matched lambda gives a matched win distribution; mismatched does not."""
    fixture = engine.Fixture.build(30, 82)
    rng = np.random.default_rng(4)
    z, u = engine.draw_noise(fixture, 2000, rng)

    s0, c0 = 5.306122448979592, 0.9081632653061225
    lam = simulation.effective_spread(s0, c0)
    s1 = 2 * s0
    c1 = float(np.exp(np.log(c0) / 2))
    assert abs(simulation.effective_spread(s1, c1) - lam) < 1e-12

    base = engine.ranked_wins_batch(fixture, z, u, s0, c0).mean(axis=0)
    matched = engine.ranked_wins_batch(fixture, z, u, s1, c1).mean(axis=0)
    mismatched = engine.ranked_wins_batch(fixture, z, u, s0, 0.80).mean(axis=0)

    assert np.allclose(base, matched, atol=1e-9)
    assert np.abs(base - mismatched).max() > 3.0


def test_noll_scully_of_balanced_league_is_about_one():
    rng = np.random.default_rng(5)
    win_pcts = rng.binomial(82, 0.5, 30) / 82
    assert 0.6 < metrics.noll_scully(win_pcts, 82) < 1.5


def test_paper_win_pct_spread_reproduces_original_formula():
    pcts = [0.756, 0.500, 0.244]
    expected = (abs(0.756 - 0.5) + abs(0.5 - 0.5) + abs(0.244 - 0.5)) ** 0.5
    assert abs(metrics.paper_win_pct_spread(pcts) - expected) < 1e-12
