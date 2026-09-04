"""Diagnostics for the fitted simulation.

Three questions:
1. Does the fitted model reproduce a season's ranked win curve? (overlay plots)
2. Does the fitted `s` track real competitive balance? (the deck's headline
   "Simulated CB vs. Real Winning Percentage Spread" regression)
3. Does fitted `s` move with dominance the way the hypothesis predicts?
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from durable_dominance import datasets as D
from durable_dominance import engine
from durable_dominance import leagues as L
from durable_dominance import plotting as P

REPORTS = Path(__file__).resolve().parents[1] / "reports"
LOADERS = {"nba": D.load_nba, "nfl": D.load_nfl}


def overlay_grid(name: str, fits: pd.DataFrame, df: pd.DataFrame, n: int = 12) -> None:
    """Real vs simulated ranked wins for an evenly spaced sample of seasons."""
    picks = fits.iloc[np.linspace(0, len(fits) - 1, n).astype(int)]
    rows = (n + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(17, 3.5 * rows))
    axes = axes.flatten()
    rng = np.random.default_rng(7)

    for ax, (_, row) in zip(axes, picks.iterrows()):
        g = df[df["season"] == row["season"]]
        real = np.sort(g["wins"].to_numpy(dtype=float))[::-1]
        fixture = engine.Fixture.build(int(row["n_teams"]), int(row["n_games"]))
        z, u = engine.draw_noise(fixture, 400, rng)
        sim = engine.ranked_wins_batch(fixture, z, u, row["s"], row["c"])
        lo, mid, hi = np.percentile(sim, [10, 50, 90], axis=0)

        x = np.arange(len(real))
        ax.fill_between(x, lo[:len(real)], hi[:len(real)], color=P.ACCENT,
                        alpha=0.45, label="simulated 10-90%")
        ax.plot(x, mid[:len(real)], color=P.TRend, lw=1.6, label="simulated median")
        ax.plot(x, real, "o-", color=P.INK, ms=3.2, lw=1.4, label="actual")
        ax.set_title(f"{row['season']}  (s={row['s']:.2f})", fontsize=10)
        ax.set_xlabel("Team rank"); ax.set_ylabel("Wins")

    for ax in axes[len(picks):]:
        ax.axis("off")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{name.upper()}: fitted model vs. actual ranked wins",
                 fontweight="bold", y=1.005)
    print("wrote", P.save(fig, f"05_{name}_ranked_wins.png").name)


def main() -> None:
    P.apply_style()
    records = []

    for name, loader in LOADERS.items():
        fits = pd.read_csv(REPORTS / f"fit_{name}.csv")
        df = loader()
        # season labels round-trip through CSV as strings for the NBA
        df["season"] = df["season"].astype(str)
        fits["season"] = fits["season"].astype(str)
        overlay_grid(name, fits, df)

        panel = L.PANELS[name]().assign(season=lambda d: d["season"].astype(str))
        merged = fits.merge(panel[["season", "dominance", "noll_scully"]],
                            on="season", how="left")

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))

        # (1) fitted s over time, against the real win% spread
        ax = axes[0]
        ax.plot(merged["start_year"], merged["s"], color=P.INK, lw=1.5,
                label="fitted s")
        ax.set_xlabel("Season"); ax.set_ylabel("Fitted s", color=P.INK)
        ax2 = ax.twinx()
        ax2.plot(merged["start_year"], merged["win_pct_sd"], color=P.TRend,
                 lw=1.5, label="SD of win%")
        ax2.set_ylabel("SD of win%", color=P.TRend); ax2.grid(False)
        ax.set_title("Fitted skill spread vs. actual win% spread")

        # (2) the deck's headline regression
        fit = P.regression_scatter(
            axes[1], merged["s"], merged["win_pct_sd"],
            xlabel="Simulated s (fitted)", ylabel="SD of winning percentage",
            title="Simulated CB vs. real win% spread",
        )
        records.append({"league": name.upper(), "relationship": "s ~ win_pct_sd",
                        "n": len(merged), "slope": fit.slope,
                        "r_squared": fit.rvalue**2, "p_value": fit.pvalue})

        # (3) does the fitted spread relate to dominance?
        sub = merged.dropna(subset=["dominance"])
        fit2 = P.regression_scatter(
            axes[2], sub["s"], sub["dominance"],
            xlabel="Simulated s (fitted)", ylabel="Dominance index",
            title="Simulated CB vs. durable dominance",
        )
        records.append({"league": name.upper(), "relationship": "s ~ dominance",
                        "n": len(sub), "slope": fit2.slope,
                        "r_squared": fit2.rvalue**2, "p_value": fit2.pvalue})

        fig.suptitle(f"{name.upper()} simulation diagnostics", fontweight="bold",
                     y=1.03)
        print("wrote", P.save(fig, f"05_{name}_diagnostics.png").name)

    out = pd.DataFrame(records)
    out.to_csv(REPORTS / "simulation_diagnostics.csv", index=False)
    print("\n" + out.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
