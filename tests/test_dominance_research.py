"""Correctness checks for the strength, playoff and dominance machinery."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
import pytest

from durable_dominance import dominance as DOM
from durable_dominance import playoffs as PO
from durable_dominance import strength as S


def toy_matches():
    """A three-team round robin where A > B > C by construction."""
    rows = []
    for a, b, wins, losses in [("A", "B", 3, 1), ("A", "C", 4, 0), ("B", "C", 3, 1)]:
        rows.append({"team": a, "opponent": b, "series_wins": wins,
                     "series_losses": losses})
        rows.append({"team": b, "opponent": a, "series_wins": losses,
                     "series_losses": wins})
    return pd.DataFrame(rows)


def test_bradley_terry_orders_teams_correctly():
    beta = S.bradley_terry(toy_matches())
    assert list(beta.index) == ["A", "B", "C"]
    assert beta["A"] > beta["B"] > beta["C"]


def test_bradley_terry_is_centred_and_finite():
    beta = S.bradley_terry(toy_matches())
    assert abs(beta.mean()) < 1e-8
    assert np.all(np.isfinite(beta))


def test_equal_teams_get_equal_strength():
    rows = []
    for a, b in [("A", "B"), ("A", "C"), ("B", "C")]:
        rows.append({"team": a, "opponent": b, "series_wins": 2, "series_losses": 2})
        rows.append({"team": b, "opponent": a, "series_wins": 2, "series_losses": 2})
    beta = S.bradley_terry(pd.DataFrame(rows))
    assert beta.abs().max() < 1e-6


def _exact_series_prob(gap: float, hca: float) -> float:
    """Closed form: convolve the seven per-game Bernoullis, then sum P(>=4)."""
    ps = 1.0 / (1.0 + np.exp(-(gap + hca * PO.HOME_PATTERN)))
    dist = np.array([1.0])
    for q in ps:
        dist = np.convolve(dist, [1 - q, q])
    return float(dist[4:].sum())


@pytest.mark.parametrize("gap,hca", [(1.0, 0.0), (0.0, 0.0), (0.0, 0.5),
                                     (0.5, 0.4), (-0.3, 0.4)])
def test_series_matches_closed_form(gap, hca):
    """The random walk must reproduce the analytic best-of-7 probability."""
    n = 200_000
    rng = np.random.default_rng(0)
    empirical = PO.simulate_series(np.full(n, gap), np.zeros(n), n, rng, hca).mean()
    assert empirical == pytest.approx(_exact_series_prob(gap, hca), abs=0.004)


def test_series_favours_the_stronger_team():
    rng = np.random.default_rng(0)
    hi = np.full(20000, 1.0)
    lo = np.zeros(20000)
    rate = PO.simulate_series(hi, lo, 20000, rng, hca=0.0).mean()
    # sigmoid(1) = 0.731 per game, which a best-of-7 amplifies to 0.911.
    assert 0.89 < rate < 0.93


def test_even_series_is_a_coin_flip():
    rng = np.random.default_rng(1)
    z = np.zeros(40000)
    rate = PO.simulate_series(z, z, 40000, rng, hca=0.0).mean()
    assert abs(rate - 0.5) < 0.01


def test_home_court_helps_the_host():
    """Four of seven at home is worth about four points of win probability."""
    rng = np.random.default_rng(2)
    z = np.zeros(60000)
    assert PO.simulate_series(z, z, 60000, rng, hca=0.5).mean() > 0.52


def toy_season(n_per_conf=8):
    teams = [f"T{i}" for i in range(2 * n_per_conf)]
    return pd.DataFrame({
        "bt": np.linspace(1.5, -1.5, len(teams)),
        "conference": ["E"] * n_per_conf + ["W"] * n_per_conf,
        "win_pct": np.linspace(0.8, 0.3, len(teams)),
    }, index=teams)


def test_title_probabilities_sum_to_one():
    p = PO.simulate_season(toy_season(), "bt", 4000,
                           rng=np.random.default_rng(3))
    assert abs(p.sum() - 1.0) < 1e-9
    assert (p >= 0).all()


def test_stronger_team_wins_more_titles():
    p = PO.simulate_season(toy_season(), "bt", 8000,
                           rng=np.random.default_rng(4))
    assert p.idxmax() == "T0"


def test_concentration_and_repeat_rate():
    same = np.zeros(10, dtype=int)
    c = PO.concentration(same, 5)
    assert c["unique_champions"] == 1 and c["hhi"] == pytest.approx(1.0)
    assert PO.repeat_rate(same) == pytest.approx(1.0)

    alternating = np.array([0, 1, 0, 1, 0, 1])
    assert PO.repeat_rate(alternating) == pytest.approx(0.0)
    assert PO.concentration(alternating, 5)["hhi"] == pytest.approx(0.5)


def test_gini_bounds():
    assert DOM.gini(np.ones(10)) == pytest.approx(0.0, abs=1e-9)
    lopsided = np.array([0.0] * 99 + [1.0])
    assert DOM.gini(lopsided) > 0.95


def test_pedigree_never_uses_the_current_season():
    """A first-time champion must show zero prior titles."""
    standings = pd.DataFrame({
        "season": [1, 1, 2, 2, 3, 3],
        "team": ["A", "B"] * 3,
        "rank": [1, 2, 1, 2, 2, 1],
        "wins": [60, 20, 60, 20, 20, 60],
        "losses": [20, 60, 20, 60, 60, 20],
    })
    champions = pd.Series({1: "A", 2: "A", 3: "B"})
    ped = DOM.champion_pedigree(standings, champions).set_index("season")
    assert ped.loc[1, "prior_titles"] == 0      # A's first title
    assert ped.loc[2, "prior_titles"] == 1      # A had one before
    assert ped.loc[3, "prior_titles"] == 0      # B's first title
    assert ped.loc[3, "prior_top4"] == 2        # B finished top-4 twice before
