"""Fit the Monte Carlo season model to real NBA and NFL standings.

Reproduces the deck's two-stage fit:

1. Search the 50x50 (s, c) grid for every season, scoring each grid point by
   the MSE between simulated and real *ranked* win vectors.
2. Choose one global `c` (lowest mean MSE across seasons), then the best `s`
   per season conditional on that `c`.

Writes `reports/fit_<league>.csv` (per-season s, c, MSE) and
`reports/fit_<league>_surface.npz` (the full grid, for script 06).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from durable_dominance import datasets as D
from durable_dominance import engine, fitting

REPORTS = Path(__file__).resolve().parents[1] / "reports"
REPLICATES = 100


def season_specs(df: pd.DataFrame) -> list[dict]:
    """One entry per season: team count, games per team, real ranked wins."""
    specs = []
    for season, g in df.groupby("season", sort=False):
        games = int(round(g["games"].median()))
        if games < 2 or len(g) < 4:
            continue
        specs.append({
            "season": season,
            "start_year": D.season_key(season),
            "n_teams": len(g),
            "n_games": games,
            "real_wins": np.sort(g["wins"].to_numpy(dtype=float))[::-1],
            "win_pct_sd": float(g["win_pct"].std(ddof=1)),
        })
    return sorted(specs, key=lambda r: r["start_year"])


def fit_league(name: str, df: pd.DataFrame) -> pd.DataFrame:
    specs = season_specs(df)
    print(f"\n=== {name}: {len(specs)} seasons ===")
    surfaces = {}
    t0 = time.perf_counter()

    for i, spec in enumerate(specs, 1):
        fixture = engine.Fixture.build(spec["n_teams"], spec["n_games"])
        surfaces[spec["season"]] = engine.mse_surface(
            fixture, spec["real_wins"], fitting.S_GRID, fitting.C_GRID,
            replicates=REPLICATES, rng=np.random.default_rng(abs(hash(name)) % 2**31 + i),
        )
        if i % 10 == 0 or i == len(specs):
            print(f"  {i}/{len(specs)} seasons  ({time.perf_counter()-t0:.0f}s)")

    stack = np.stack([surfaces[s["season"]] for s in specs])  # (season, s, c)

    # Stage 1: one global c, lowest mean MSE across every season.
    mean_by_c = np.nanmean(stack, axis=(0, 1))
    c_idx = int(np.nanargmin(mean_by_c))
    best_c = float(fitting.C_GRID[c_idx])

    # Stage 2: best s per season, holding c fixed.
    rows = []
    for spec, surface in zip(specs, stack):
        col = surface[:, c_idx]
        s_idx = int(np.nanargmin(col))
        rows.append({
            "season": spec["season"],
            "start_year": spec["start_year"],
            "n_teams": spec["n_teams"],
            "n_games": spec["n_games"],
            "c": best_c,
            "s": float(fitting.S_GRID[s_idx]),
            "mse": float(col[s_idx]),
            "lambda": abs(float(fitting.S_GRID[s_idx]) * np.log(best_c)),
            "win_pct_sd": spec["win_pct_sd"],
        })

    out = pd.DataFrame(rows)
    out.to_csv(REPORTS / f"fit_{name}.csv", index=False)
    np.savez_compressed(
        REPORTS / f"fit_{name}_surface.npz",
        surface=stack,
        s_grid=fitting.S_GRID,
        c_grid=fitting.C_GRID,
        seasons=np.array([str(s["season"]) for s in specs]),
    )
    print(f"  global c = {best_c:.6f}   mean MSE = {mean_by_c[c_idx]:.3f}")
    print(f"  s ranges {out['s'].min():.2f} - {out['s'].max():.2f}")
    return out


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    for name, loader in (("nba", D.load_nba), ("nfl", D.load_nfl)):
        out = fit_league(name, loader())
        print(out.head(5).to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
