"""Recreate the deck's "Miscellaneous Sports" panels.

Each sport asks the same question in its own vocabulary: when the field opens
up (more new entrants, more distinct winners, closer scores), does the
established elite loosen its grip?

The deck did not state its exact definitions for these panels, so each one is
reconstructed here from the axis labels and value ranges and the definition is
written out in the function docstring. Where the reconstruction is uncertain
it is flagged in `reports/DATA_GAPS.md`. Rugby and the men's World Cup panels
are omitted: neither dataset is in this repo.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from durable_dominance import datasets as D
from durable_dominance import plotting as P

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def masters_panel() -> pd.DataFrame:
    """Masters: new players in the field vs. returning top-10 finishers.

    x = players appearing for the first time in this year's field.
    y = how many of this year's top 10 had already finished top 10 before.
    """
    df = D.load_masters()
    seen: set[str] = set()
    prior_top10: dict[str, int] = {}
    rows = []
    for year, g in df.groupby("season", sort=True):
        field = set(g["player"])
        top10 = g.nsmallest(10, "position")["player"]
        rows.append({
            "season": int(year),
            "new_players": len(field - seen),
            "returning_top10": sum(1 for p in top10 if prior_top10.get(p, 0) > 0),
        })
        seen |= field
        for p in top10:
            prior_top10[p] = prior_top10.get(p, 0) + 1
    return pd.DataFrame(rows)


def tour_de_france_panel() -> pd.DataFrame:
    """Tour de France: new entrants vs. accumulated top-10 pedigree.

    x = riders starting their first Tour in this edition.
    y = summed prior top-10 finishes of this year's top 10.
    """
    df = D.load_tour_de_france()
    seen: set[str] = set()
    prior_top10: dict[str, int] = {}
    rows = []
    for year, g in df.groupby("season", sort=True):
        field = set(g["rider"])
        top10 = g.nsmallest(10, "position")["rider"]
        rows.append({
            "season": int(year),
            "new_entrants": len(field - seen),
            "top10_pedigree": sum(prior_top10.get(r, 0) for r in top10),
        })
        seen |= field
        for r in top10:
            prior_top10[r] = prior_top10.get(r, 0) + 1
    return pd.DataFrame(rows)


def grand_slam_panel(window: int = 2) -> pd.DataFrame:
    """Men's tennis: distinct champions vs. their combined major haul.

    Over a rolling `window` of years (8 majors at the default of 2, which
    reproduces the deck's 3-8 range on the x-axis):
    x = number of distinct champions.
    y = their combined career major total.
    """
    df = D.load_grand_slams()
    career = df.groupby("winner").size()
    years = sorted(df["season"].dropna().unique())
    rows = []
    for i in range(window - 1, len(years)):
        span = years[i - window + 1: i + 1]
        champs = df[df["season"].isin(span)]["winner"].unique()
        rows.append({
            "season": int(years[i]),
            "unique_champions": len(champs),
            "combined_majors": int(career.reindex(champs).fillna(0).sum()),
        })
    return pd.DataFrame(rows)


RUNS = re.compile(r"(\d+)\s*runs", re.I)


def cricket_panel() -> pd.DataFrame:
    """Cricket World Cup: spread of runs margins vs. the champion's pedigree.

    x = standard deviation of the runs margin across that tournament's matches.
    y = the champion's cumulative titles up to and including that year.
    """
    df = D.load_cricket()
    df = df.assign(
        runs_margin=pd.to_numeric(
            df["margin"].astype(str).str.extract(RUNS)[0], errors="coerce"
        )
    )
    titles: dict[str, int] = {}
    rows = []
    for year, g in df.groupby("season", sort=True):
        margins = g["runs_margin"].dropna()
        if len(margins) < 2:
            continue
        # The champion wins the tournament's last match.
        champion = g.iloc[-1]["winner"]
        titles[champion] = titles.get(champion, 0) + 1
        rows.append({
            "season": int(year),
            "margin_sd": float(margins.std(ddof=1)),
            "champion_titles": titles[champion],
        })
    return pd.DataFrame(rows)


def womens_world_cup_panel() -> pd.DataFrame:
    """Women's World Cup: average goal margin vs. the champion's title count."""
    df = D.load_womens_world_cup()
    titles: dict[str, int] = {}
    rows = []
    for year, g in df.groupby("season", sort=True):
        final = g.iloc[-1]
        champ = (
            final["home"] if final["home_goals"] >= final["away_goals"]
            else final["away"]
        )
        titles[champ] = titles.get(champ, 0) + 1
        rows.append({
            "season": int(year),
            "score_difference": float(g["goal_diff"].mean()),
            "champion_titles": titles[champ],
        })
    return pd.DataFrame(rows)


PANELS = [
    ("Masters", masters_panel, "new_players", "returning_top10",
     "New players in field", "Returning top-10 finishers"),
    ("Tour de France", tour_de_france_panel, "new_entrants", "top10_pedigree",
     "New entrants", "Sum of prior top-10 finishes"),
    ("Grand Slams", grand_slam_panel, "unique_champions", "combined_majors",
     "Unique champions (2-yr window)", "Combined career majors"),
    ("Cricket World Cup", cricket_panel, "margin_sd", "champion_titles",
     "SD of runs margin", "Champion's cumulative titles"),
    ("Women's World Cup", womens_world_cup_panel, "score_difference",
     "champion_titles", "Mean goal margin", "Champion's cumulative titles"),
]


def main() -> None:
    P.apply_style()
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    axes = axes.flatten()
    records = []

    for ax, (name, builder, xcol, ycol, xlabel, ylabel) in zip(axes, PANELS):
        panel = builder()
        fit = P.regression_scatter(
            ax, panel[xcol], panel[ycol],
            xlabel=xlabel, ylabel=ylabel, title=name,
        )
        records.append({
            "sport": name, "x": xcol, "y": ycol, "n_years": len(panel),
            "slope": fit.slope, "r_squared": fit.rvalue**2, "p_value": fit.pvalue,
        })
        panel.to_csv(REPORTS / f"individual_{name.lower().replace(' ', '_')}.csv",
                     index=False)

    axes[-1].axis("off")
    axes[-1].text(
        0.0, 0.5,
        "Rugby Championship and men's World Cup\npanels from the deck are not "
        "reproducible:\nneither dataset is committed to this repo.\n\n"
        "See reports/DATA_GAPS.md",
        fontsize=9, va="center", color=P.INK,
    )
    fig.suptitle("Individual and knockout sports: openness vs. entrenchment",
                 fontweight="bold", y=1.01)
    print("wrote", P.save(fig, "03_individual_sports.png").name)

    out = pd.DataFrame(records)
    out.to_csv(REPORTS / "individual_sports.csv", index=False)
    print("\n" + out.to_string(index=False, float_format=lambda v: f"{v:.4g}"))


if __name__ == "__main__":
    main()
