"""Vectorised season simulator used for parameter fitting.

`simulation.py` holds the readable, match-by-match reference implementation.
This module produces identical draws in bulk so a 50x50 grid can be searched
over 40+ seasons in seconds instead of days.

Two identities do the work:

    P(team 1 wins) = c**s1 / (c**s1 + c**s2) = sigmoid(ln(c) * (s1 - s2))

so no powers are ever evaluated, and win totals come from one matrix product
against the schedule's incidence matrix rather than a Python loop over games.

`tests/test_engine.py` checks this against `simulation.py` directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from durable_dominance.simulation import (
    DEFAULT_LOWER_BOUND,
    DEFAULT_SKILL_MEAN,
    DEFAULT_UPPER_BOUND,
    create_schedule,
)


@dataclass(frozen=True)
class Fixture:
    """A schedule pre-compiled into the arrays the batch simulator needs."""

    n_teams: int
    n_matches: int
    home: np.ndarray          # team index on the left of each match
    away: np.ndarray          # team index on the right
    design: np.ndarray        # (n_matches, n_teams): +1 home, -1 away
    away_counts: np.ndarray   # games each team plays as the away side

    @classmethod
    def build(cls, n_teams: int, n_games: int) -> "Fixture":
        schedule = create_schedule(n_teams, n_games)
        home = np.array([i for i, _ in schedule], dtype=np.intp)
        away = np.array([j for _, j in schedule], dtype=np.intp)
        design = np.zeros((len(schedule), n_teams))
        design[np.arange(len(schedule)), home] = 1.0
        design[np.arange(len(schedule)), away] = -1.0
        away_counts = np.bincount(away, minlength=n_teams).astype(float)
        return cls(n_teams, len(schedule), home, away, design, away_counts)


def draw_noise(fixture: Fixture, replicates: int, rng: np.random.Generator):
    """Standard normal skills and uniform match draws, shared across the grid.

    Reusing one noise block for every (s, c) turns the MSE surface smooth --
    neighbouring grid points then differ because the parameters differ, not
    because the random draws did (common random numbers).
    """
    z = rng.standard_normal((replicates, fixture.n_teams))
    u = rng.random((replicates, fixture.n_matches))
    return z, u


def ranked_wins_batch(
    fixture: Fixture,
    z: np.ndarray,
    u: np.ndarray,
    s: float,
    c: float,
    mean: float = DEFAULT_SKILL_MEAN,
    lower: float = DEFAULT_LOWER_BOUND,
    upper: float = DEFAULT_UPPER_BOUND,
) -> np.ndarray:
    """Sorted win totals for every replicate: shape (replicates, n_teams)."""
    skills = np.clip(mean + s * z, lower, upper)
    diff = skills[:, fixture.home] - skills[:, fixture.away]
    # sigmoid(ln(c) * diff), written to avoid overflow warnings on big margins.
    p_home = 1.0 / (1.0 + np.exp(-np.log(c) * diff))
    home_won = (u < p_home).astype(float)
    wins = home_won @ fixture.design + fixture.away_counts
    wins.sort(axis=1)
    return wins[:, ::-1]


def mse_surface(
    fixture: Fixture,
    real_wins: np.ndarray,
    s_grid: np.ndarray,
    c_grid: np.ndarray,
    replicates: int = 100,
    rng: np.random.Generator | None = None,
    summary: str = "mode",
) -> np.ndarray:
    """MSE of ranked wins against `real_wins` for every (s, c). Shape (|s|,|c|)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    z, u = draw_noise(fixture, replicates, rng)
    real = np.sort(np.asarray(real_wins, dtype=float))[::-1]
    n = min(len(real), fixture.n_teams)

    out = np.empty((len(s_grid), len(c_grid)))
    for a, s in enumerate(s_grid):
        for b, c in enumerate(c_grid):
            sim = ranked_wins_batch(fixture, z, u, s, c)
            per_rep = np.mean((sim[:, :n] - real[:n]) ** 2, axis=1)
            out[a, b] = _summarise(per_rep, summary)
    return out


def _summarise(values: np.ndarray, how: str) -> float:
    if how == "mean":
        return float(values.mean())
    vals, counts = np.unique(values, return_counts=True)
    return float(vals[counts.argmax()])


def lambda_curve(
    fixture: Fixture,
    real_wins: np.ndarray,
    lambdas: np.ndarray,
    replicates: int = 400,
    rng: np.random.Generator | None = None,
    summary: str = "mean",
    c: float = 0.9081632653061225,
) -> np.ndarray:
    """MSE along the single identified parameter lambda = s * |ln c|.

    Because the win distribution depends on (s, c) only through this product,
    a 1-D sweep here carries all the information the 2-D grid does, at 1/50th
    the cost. `c` only fixes where on the ridge each lambda is evaluated.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    z, u = draw_noise(fixture, replicates, rng)
    real = np.sort(np.asarray(real_wins, dtype=float))[::-1]
    n = min(len(real), fixture.n_teams)

    out = np.empty(len(lambdas))
    for i, lam in enumerate(lambdas):
        s = lam / abs(np.log(c))
        sim = ranked_wins_batch(fixture, z, u, s, c)
        per_rep = np.mean((sim[:, :n] - real[:n]) ** 2, axis=1)
        out[i] = _summarise(per_rep, summary)
    return out
