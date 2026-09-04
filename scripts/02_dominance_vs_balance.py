"""Recreate the deck's "Competitive Balance vs. Durable Dominance" charts.

Covers the three leagues whose data is in this repo (NBA, NFL, NCAA football)
across the same era windows the deck used, and writes every regression to
`reports/dominance_vs_balance.csv`.

The deck's soccer, MLB and NHL panels cannot be reproduced here: those source
files were never committed (see reports/DATA_GAPS.md).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import pandas as pd

from durable_dominance import leagues as L
from durable_dominance import plotting as P

REPORTS = Path(__file__).resolve().parents[1] / "reports"

# (league, balance column, axis label, era windows the deck showed)
SPECS = [
    ("nba", "paper_cb_win", "Standard Deviation of Win%",
     [("1981-Present", 1981, 2100), ("1981-2000", 1981, 2000),
      ("2000-Present", 2000, 2100), ("2003-Present", 2003, 2100)]),
    ("nba", "paper_cb_rating", "Standard Deviation of MOV",
     [("1981-Present", 1981, 2100), ("1988-Present", 1988, 2100),
      ("2000-Present", 2000, 2100), ("2010-Present", 2010, 2100)]),
    ("nba_nrtg", "paper_cb_rating", "Standard Deviation of NRTG",
     [("1984-Present", 1984, 2100), ("1988-Present", 1988, 2100),
      ("2000-Present", 2000, 2100), ("2010-Present", 2010, 2100)]),
    ("nfl", "paper_cb_win", "Standard Deviation of Win%",
     [("1975-Present", 1975, 2100), ("1990-Present", 1990, 2100),
      ("2000-Present", 2000, 2100)]),
    ("nfl", "paper_cb_rating", "Standard Deviation of MOV",
     [("1975-Present", 1975, 2100), ("1990-Present", 1990, 2100),
      ("2000-Present", 2000, 2100)]),
    ("ncaaf", "win_pct_sd", "Std Dev of Win%", [("1970-Present", 1970, 2100)]),
    ("ncaaf", "rating_sd", "Std Dev of SRS", [("1970-Present", 1970, 2100)]),
]

TITLES = {"nba": "NBA", "nba_nrtg": "NBA", "nfl": "NFL", "ncaaf": "NCAA Football"}


def main() -> None:
    P.apply_style()
    panels = {name: fn() for name, fn in L.PANELS.items()}
    records = []

    for league, cb_col, xlabel, windows in SPECS:
        panel = panels[league].dropna(subset=["dominance", cb_col])
        fig, axes = plt.subplots(
            1, len(windows), figsize=(4.6 * len(windows), 4.1), squeeze=False
        )
        for ax, (label, lo, hi) in zip(axes[0], windows):
            sub = panel[(panel["start_year"] >= lo) & (panel["start_year"] <= hi)]
            fit = P.regression_scatter(
                ax, sub[cb_col], sub["dominance"],
                xlabel=xlabel, ylabel="Dominance Index",
                title=f"{TITLES[league]} {label}",
            )
            records.append({
                "league": TITLES[league], "balance_metric": cb_col, "window": label,
                "n_seasons": len(sub), "slope": fit.slope, "intercept": fit.intercept,
                "r_squared": fit.rvalue**2, "p_value": fit.pvalue,
                "std_err": fit.stderr,
            })
        fig.suptitle(
            f"{TITLES[league]} competitive balance vs. durable dominance "
            f"({xlabel})", y=1.04, fontweight="bold",
        )
        name = f"02_{league}_{cb_col}.png"
        print("wrote", P.save(fig, name).name)

    out = pd.DataFrame(records)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out.to_csv(REPORTS / "dominance_vs_balance.csv", index=False)
    print("\n" + out.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
