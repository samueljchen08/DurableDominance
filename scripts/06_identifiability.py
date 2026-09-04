"""Is the (s, c) parameterisation identified? No -- and this shows why.

The match rule
    P(team 1 wins) = c**s1 / (c**s1 + c**s2)
can be rewritten as
    P = sigmoid(ln(c) * (s1 - s2)),
and with skills drawn from Normal(mean, s) the difference s1 - s2 is
Normal(0, s*sqrt(2)). So every outcome depends on (s, c) only through

    lambda = s * |ln c|.

Any two parameter pairs on the same lambda contour produce the *same*
distribution of results. The 50x50 grid search therefore has a ridge of
equally good optima rather than a unique one, which is why the fitted `c`
comes out as a single constant for every league and season while `s` absorbs
all of the variation: `c` is not being estimated, it is being pinned by the
grid's edge and the tie-breaking rule.

The one place the degeneracy breaks is the clip to [0, 100]: at large `s` the
Normal is truncated and the equivalence weakens. This script measures where
that happens.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from durable_dominance import engine, fitting, plotting as P
from durable_dominance.simulation import effective_spread

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def main() -> None:
    P.apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    # ---- (1) The MSE surface, with the analytic lambda contour on top -------
    data = np.load(REPORTS / "fit_nba_surface.npz", allow_pickle=True)
    surface = np.nanmean(data["surface"], axis=0)  # average over seasons
    s_grid, c_grid = data["s_grid"], data["c_grid"]

    ax = axes[0]
    mesh = ax.pcolormesh(c_grid, s_grid, np.log10(surface), cmap="magma_r",
                         shading="auto")
    fig.colorbar(mesh, ax=ax, label="log10 mean MSE")
    best = np.unravel_index(np.nanargmin(surface), surface.shape)
    lam_star = effective_spread(s_grid[best[0]], c_grid[best[1]])
    cc = np.linspace(c_grid[1], c_grid[-2], 200)
    ax.plot(cc, lam_star / np.abs(np.log(cc)), color="#2bb7a8", lw=2.2,
            label=f"$\\lambda = s\\,|\\ln c| = {lam_star:.2f}$")
    ax.scatter([c_grid[best[1]]], [s_grid[best[0]]], s=60, color="#2bb7a8",
               edgecolor="white", zorder=5)
    ax.set_xlabel("c"); ax.set_ylabel("s"); ax.set_ylim(s_grid[0], s_grid[-1])
    ax.set_title("NBA mean MSE surface\n(valley follows the $\\lambda$ contour)")
    ax.legend(loc="upper left", fontsize=8)

    # ---- (2) Collapse: MSE against lambda, coloured by c -------------------
    ax = axes[1]
    S, C = np.meshgrid(s_grid, c_grid, indexing="ij")
    lam = np.abs(S * np.log(np.where(C > 0, C, np.nan)))
    ok = np.isfinite(lam) & np.isfinite(surface) & (lam < 3)
    sc = ax.scatter(lam[ok], surface[ok], c=C[ok], s=7, cmap="viridis", alpha=0.7)
    fig.colorbar(sc, ax=ax, label="c")
    ax.set_yscale("log")
    ax.set_xlabel("$\\lambda = s\\,|\\ln c|$"); ax.set_ylabel("mean MSE")
    ax.set_title("All 2,500 grid points collapse\nonto one curve in $\\lambda$")

    # ---- (3) Where the clip breaks the equivalence --------------------------
    fixture = engine.Fixture.build(30, 82)
    rng = np.random.default_rng(11)
    z, u = engine.draw_noise(fixture, 3000, rng)
    c0 = 0.9081632653061225
    rows = []
    for s0 in [2, 4, 6, 8, 10, 14, 18, 22, 26, 30]:
        base = engine.ranked_wins_batch(fixture, z, u, s0, c0).mean(axis=0)
        for mult in (2.0, 4.0):
            s1 = s0 * mult
            c1 = float(np.exp(np.log(c0) / mult))
            alt = engine.ranked_wins_batch(fixture, z, u, s1, c1).mean(axis=0)
            rows.append({"s": s0, "multiplier": mult,
                         "max_abs_diff_wins": float(np.abs(base - alt).max())})
    dev = pd.DataFrame(rows)
    ax = axes[2]
    for mult, g in dev.groupby("multiplier"):
        ax.plot(g["s"], g["max_abs_diff_wins"], "o-",
                label=f"s scaled x{mult:.0f}")
    ax.axhline(0.5, color=P.TRend, ls=":", label="half a win")
    ax.set_xlabel("base s"); ax.set_ylabel("max |difference| in mean wins")
    ax.set_title("Equivalence holds until the\nskill clip at [0, 100] binds")
    ax.legend(fontsize=8)

    fig.suptitle(
        "The (s, c) grid search is not identified: only $\\lambda = s\\,|\\ln c|$ "
        "is estimable", fontweight="bold", y=1.04,
    )
    print("wrote", P.save(fig, "06_identifiability.png").name)
    dev.to_csv(REPORTS / "identifiability_clip.csv", index=False)

    # ---- 1-D lambda fit reproduces the 2-D fit at 1/50th the cost ----------
    fits = pd.read_csv(REPORTS / "fit_nba.csv")
    lam_grid = np.linspace(0.05, 2.0, 60)
    recovered = []
    for _, row in fits.iterrows():
        fx = engine.Fixture.build(int(row["n_teams"]), int(row["n_games"]))
        real = None
        curve = engine.lambda_curve(
            fx, _real_wins(row), lam_grid, replicates=300,
            rng=np.random.default_rng(int(row["start_year"])),
        )
        recovered.append(lam_grid[int(np.nanargmin(curve))])
    fits["lambda_1d"] = recovered
    corr = fits[["lambda", "lambda_1d"]].corr().iloc[0, 1]
    fits.to_csv(REPORTS / "fit_nba_lambda.csv", index=False)
    print(f"\n2-D grid lambda vs 1-D lambda sweep: r = {corr:.3f} "
          f"(n={len(fits)}); the 1-D sweep is 50x cheaper.")
    print(dev.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def _real_wins(row) -> np.ndarray:
    """Recover a season's ranked wins from the source table."""
    from durable_dominance import datasets as D
    df = D.load_nba()
    df["season"] = df["season"].astype(str)
    g = df[df["season"] == str(row["season"])]
    return np.sort(g["wins"].to_numpy(dtype=float))[::-1]


if __name__ == "__main__":
    main()
