"""Fit the simulation's (s, c) / (s, b) parameters to real season standings.

The procedure follows the notebook that produced the Sloan figures:

1. For every grid point, simulate the season `iterations` times and record the
   sorted win vector each time.
2. Score each replicate against the real sorted win vector by MSE.
3. Collapse the replicates to a single number per grid point using the *mode*
   of the MSE distribution (`mode_mse`), not the mean.
4. Pick one global `c` by averaging that score across all seasons, then pick a
   per-season `s` conditional on that `c`.

Step 3 is the paper's choice and is preserved here. `summary_stat="mean"`
switches to the more usual average, which `scripts/07_metric_robustness.py`
compares against.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from durable_dominance.simulation import (
    create_schedule,
    generate_skills,
    simulate_season,
    simulate_season_with_ties,
)

# The grids the original notebook searched.
S_GRID = np.linspace(0, 10, 50)
C_GRID = np.linspace(0.5, 1, 50)
B_GRID = np.linspace(0, 1, 50)


def mse(simulated, real) -> float:
    """Mean squared error between two equal-length ranked win vectors."""
    a = np.asarray(simulated, dtype=float)
    b = np.asarray(real, dtype=float)
    n = min(len(a), len(b))
    return float(np.mean((a[:n] - b[:n]) ** 2))


def mode_mse(values) -> float:
    """Most frequent MSE across replicates, as used in the original notebook.

    MSE values are continuous, so ties are common and `np.unique` picks the
    smallest of the tied modes. That makes this a rank-order statistic closer
    to a minimum than to a centre, which is why it is worth comparing against
    the mean.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan")
    vals, counts = np.unique(arr, return_counts=True)
    return float(vals[counts.argmax()])


def _summarise(values, how: str) -> float:
    return mode_mse(values) if how == "mode" else float(np.mean(values))


@dataclass(frozen=True)
class SeasonSpec:
    """One season to fit: its label, size, schedule length and real results."""

    season: object
    n_teams: int
    n_games: int
    real_wins: np.ndarray
    real_ties: np.ndarray | None = None


def score_grid(
    spec: SeasonSpec,
    s_grid=S_GRID,
    c_grid=C_GRID,
    iterations: int = 100,
    seed: int | None = 0,
    summary_stat: str = "mode",
) -> pd.DataFrame:
    """MSE surface over (s, c) for one season."""
    rng = np.random.default_rng(seed)
    schedule = create_schedule(spec.n_teams, spec.n_games)
    real = np.sort(np.asarray(spec.real_wins, dtype=float))[::-1]

    rows = []
    for s in s_grid:
        for c in c_grid:
            scores = []
            for _ in range(iterations):
                skills = generate_skills(spec.n_teams, s, rng=rng)
                wins = simulate_season(skills, schedule, c, rng=rng)
                scores.append(mse(np.sort(wins)[::-1], real))
            rows.append({"season": spec.season, "s": s, "c": c,
                         "mse": _summarise(scores, summary_stat)})
    return pd.DataFrame(rows)


def score_grid_with_ties(
    spec: SeasonSpec,
    c: float,
    s_grid=S_GRID,
    b_grid=B_GRID,
    iterations: int = 100,
    seed: int | None = 0,
    summary_stat: str = "mode",
) -> pd.DataFrame:
    """MSE surface over (s, b) at fixed `c`, scoring wins and ties separately.

    Teams are ranked by points (3-1-0) before comparison, matching the soccer
    notebook, so the tie and win vectors share one ordering.
    """
    rng = np.random.default_rng(seed)
    schedule = create_schedule(spec.n_teams, spec.n_games)
    real_wins = np.asarray(spec.real_wins, dtype=float)
    real_ties = (
        np.asarray(spec.real_ties, dtype=float)
        if spec.real_ties is not None
        else np.zeros_like(real_wins)
    )

    rows = []
    for s in s_grid:
        for b in b_grid:
            win_scores, tie_scores = [], []
            for _ in range(iterations):
                skills = generate_skills(spec.n_teams, s, rng=rng)
                res = simulate_season_with_ties(skills, schedule, c, b, rng=rng)
                order = np.argsort(-res["points"])
                win_scores.append(mse(res["wins"][order], real_wins))
                tie_scores.append(mse(res["ties"][order], real_ties))
            rows.append({
                "season": spec.season, "s": s, "b": b,
                "mse_wins": _summarise(win_scores, summary_stat),
                "mse_ties": _summarise(tie_scores, summary_stat),
            })
    return pd.DataFrame(rows)


def choose_global_c(grid: pd.DataFrame, value_col: str = "mse") -> float:
    """The `c` with the lowest mean MSE across every season."""
    return float(grid.groupby("c")[value_col].mean().idxmin())


def best_s_per_season(
    grid: pd.DataFrame, c: float, value_col: str = "mse"
) -> pd.DataFrame:
    """Lowest-MSE `s` for each season, holding `c` fixed."""
    at_c = grid[np.isclose(grid["c"], c)]
    idx = at_c.groupby("season")[value_col].idxmin()
    out = at_c.loc[idx, ["season", "s", value_col]]
    return out.rename(columns={value_col: "mse"}).reset_index(drop=True)


def fit_league(
    specs: list[SeasonSpec],
    s_grid=S_GRID,
    c_grid=C_GRID,
    iterations: int = 100,
    seed: int | None = 0,
    summary_stat: str = "mode",
) -> tuple[float, pd.DataFrame, pd.DataFrame]:
    """Run the full two-stage fit. Returns (best c, per-season fits, full grid)."""
    grid = pd.concat(
        [
            score_grid(spec, s_grid, c_grid, iterations, seed, summary_stat)
            for spec in specs
        ],
        ignore_index=True,
    )
    c = choose_global_c(grid)
    return c, best_s_per_season(grid, c), grid
