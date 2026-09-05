"""Study E -- which sports are most dominated, on a common scale?

Comparing dominance across sports is awkward because the fields differ wildly:
30 NBA teams, ~130 college football programmes, ~90 Masters entrants, ~180
Tour de France starters. A sport with fewer contenders looks concentrated for
trivial reasons.

The fix is to score every sport against its own null. For each edition, draw a
champion uniformly from the field that actually competed that year, repeat
thousands of times, and measure how concentrated those random histories are.
The **dominance multiple** is the real concentration divided by the null's --
how many times more concentrated the sport is than blind chance among the
people who actually showed up.

That makes a 43-season basketball league and a 148-running horse race
comparable, and it is the same test Study B applies to the NBA, only with a
weaker (uniform) null instead of a strength-based one.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from durable_dominance import datasets as D
from durable_dominance import plotting as P

REPORTS = Path(__file__).resolve().parents[1] / "reports"
N_NULL = 20_000


def concentration_stats(labels: np.ndarray) -> dict:
    _, counts = np.unique(labels, return_counts=True)
    shares = counts / counts.sum()
    return {
        "unique": len(counts),
        "hhi": float((shares**2).sum()),
        "top_share": float(shares.max()),
        "effective_n": float(np.exp(-(shares * np.log(shares)).sum())),
    }


def null_distribution(fields: list[np.ndarray], n: int = N_NULL,
                      seed: int = 0) -> pd.DataFrame:
    """Champions drawn uniformly from each edition's actual field."""
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        picks = np.array([f[rng.integers(len(f))] for f in fields])
        rows.append(concentration_stats(picks))
    return pd.DataFrame(rows)


def analyse(name: str, fields: list[np.ndarray], winners: list,
            unit: str, seed: int = 0) -> dict:
    """Real concentration against the uniform-field null."""
    real = concentration_stats(np.array(winners))
    null = null_distribution(fields, seed=seed)
    below = float((null["hhi"] <= real["hhi"]).mean())
    return {
        "sport": name, "unit": unit, "editions": len(winners),
        "field_median": float(np.median([len(f) for f in fields])),
        "unique_real": real["unique"], "unique_null": float(null["unique"].median()),
        "hhi_real": real["hhi"], "hhi_null": float(null["hhi"].median()),
        "effective_n_real": real["effective_n"],
        "effective_n_null": float(null["effective_n"].median()),
        "top_share_real": real["top_share"],
        "dominance_multiple": real["hhi"] / float(null["hhi"].median()),
        "p_two_sided": 2 * min(below, 1 - below),
    }


def from_leaderboard(df, season_col, name_col, pos_col):
    """Fields and winners from a results table with finishing positions."""
    fields, winners, seasons = [], [], []
    for season, g in df.groupby(season_col, sort=True):
        g = g.dropna(subset=[pos_col, name_col])
        if g.empty:
            continue
        fields.append(g[name_col].to_numpy())
        winners.append(g.nsmallest(1, pos_col)[name_col].iloc[0])
        seasons.append(season)
    return fields, winners, seasons


def main() -> None:
    P.apply_style()
    results = []

    # --- NBA: champion known, field = the league -------------------------
    n, m = D.load_nba(), D.load_nba_matches()
    champs = m.groupby("season")["champion"].first()
    fields = [n[n["season"] == s]["team"].to_numpy() for s in champs.index]
    results.append(analyse("NBA", fields, list(champs), "franchise"))

    # --- NCAA football: AP #1 --------------------------------------------
    c = D.load_ncaaf()
    fields, winners, _ = [], [], []
    for season, g in c.groupby("season", sort=True):
        top = g[g["ap_rank"] == 1]
        if top.empty:
            continue
        fields.append(g["team"].to_numpy())
        winners.append(top["team"].iloc[0])
    results.append(analyse("NCAA football", fields, winners, "programme"))

    # --- Masters ----------------------------------------------------------
    f, w, _ = from_leaderboard(D.load_masters(), "season", "player", "position")
    results.append(analyse("Masters golf", f, w, "player"))

    # --- Tour de France ---------------------------------------------------
    f, w, _ = from_leaderboard(D.load_tour_de_france(), "season", "rider", "position")
    results.append(analyse("Tour de France", f, w, "rider"))

    # --- Tennis majors: field approximated by that year's finalists -------
    gs = D.load_grand_slams()
    fields, winners = [], []
    for season, g in gs.groupby("season", sort=True):
        pool = pd.concat([g["winner"], g["runner_up"]]).dropna().unique()
        if len(pool) < 2:
            continue
        for w_ in g["winner"].dropna():
            fields.append(pool)
            winners.append(w_)
    # The field here is only that year's finalists, the strongest possible
    # comparison set, so this multiple is a lower bound.
    results.append(analyse("Tennis majors", fields, winners, "player"))

    # --- Cricket World Cup ------------------------------------------------
    ck = D.load_cricket()
    fields, winners = [], []
    for season, g in ck.groupby("season", sort=True):
        pool = pd.concat([g["team1"], g["team2"]]).dropna().unique()
        champ = g.iloc[-1]["winner"]
        if pd.isna(champ) or len(pool) < 2:
            continue
        fields.append(pool)
        winners.append(champ)
    results.append(analyse("Cricket World Cup", fields, winners, "nation"))

    # --- Women's World Cup ------------------------------------------------
    wc = D.load_womens_world_cup()
    fields, winners = [], []
    for season, g in wc.groupby("season", sort=True):
        pool = pd.concat([g["home"], g["away"]]).dropna().unique()
        final = g.iloc[-1]
        champ = final["home"] if final["home_goals"] >= final["away_goals"] else final["away"]
        fields.append(pool)
        winners.append(champ)
    results.append(analyse("Women's World Cup", fields, winners, "nation"))

    # --- Kentucky Derby -------------------------------------------------
    # Reported descriptively only. A horse runs the Derby once, so the unit
    # has to be the sire -- but a sire is only eligible in the years he has
    # three-year-olds running, and that eligible field is not in the data. A
    # uniform-over-all-sires null would let every sire win every year, which
    # is why its "multiple" comes out below 1 and is not comparable.
    kd = D.load_kentucky_derby().dropna(subset=["sire"])
    sires = kd["sire"].astype(str).str.strip()
    derby = concentration_stats(sires.to_numpy())
    print(f"Kentucky Derby, {len(sires)} runnings: {derby['unique']} distinct "
          f"sires, top sire share {derby['top_share']:.3f}, "
          f"effective variety {derby['effective_n']:.1f}.")
    print("  No comparable null: the eligible field of sires is unobserved.\n")

    atlas = pd.DataFrame(results).sort_values("dominance_multiple", ascending=False)
    atlas.to_csv(REPORTS / "cross_sport_atlas.csv", index=False)
    print(atlas.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- figure ----------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    ax = axes[0]
    a = atlas.sort_values("dominance_multiple")
    cols = [P.TRend if p < 0.05 else "#9aa8b5" for p in a["p_two_sided"]]
    ax.barh(range(len(a)), a["dominance_multiple"], color=cols)
    ax.axvline(1, color=P.INK, lw=1.2, label="chance among entrants")
    ax.set_yticks(range(len(a))); ax.set_yticklabels(a["sport"], fontsize=9)
    ax.set_xlabel("dominance multiple  (real HHI ÷ null HHI)")
    ax.set_title("How concentrated is each sport,\nrelative to its own field?")
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.scatter(atlas["field_median"], atlas["dominance_multiple"], s=70,
               color=P.INK, zorder=3)
    for _, r in atlas.iterrows():
        ax.annotate(r["sport"], (r["field_median"], r["dominance_multiple"]),
                    fontsize=7.5, xytext=(5, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("median field size (log)")
    ax.set_ylabel("dominance multiple")
    ax.set_title("Bigger fields, more room\nfor a dynasty to stand out")

    ax = axes[2]
    x = np.arange(len(atlas))
    ax.bar(x - .2, atlas["effective_n_real"], .4, label="actual", color=P.TRend)
    ax.bar(x + .2, atlas["effective_n_null"], .4, label="if random", color=P.INK)
    ax.set_xticks(x)
    ax.set_xticklabels(atlas["sport"], rotation=35, ha="right", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("effective number of champions (log)")
    ax.set_title("Effective variety of winners")
    ax.legend(fontsize=8)

    fig.suptitle("A cross-sport atlas of durable dominance", fontweight="bold",
                 y=1.02)
    print("\nwrote", P.save(fig, "15_cross_sport_atlas.png").name)


if __name__ == "__main__":
    main()
