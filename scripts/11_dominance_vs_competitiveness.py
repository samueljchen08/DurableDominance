"""Study A -- how established are champions, and does league balance matter?

For every season we ask two things:

* **Pedigree.** At the moment it won, how much history did the champion
  already have? Six backward-looking measures: prior titles, titles in the
  last decade, prior top-4 and top-8 finishes, career winning percentage, and
  time since the last title.
* **Competitiveness.** How tight was the league that year? Eight measures,
  from the spread of records to the rate at which the weaker team actually
  won its games.

Then we correlate every pedigree measure against every competitiveness
measure. If balance entrenches elites, tighter seasons should crown more
decorated champions.

Covers the NBA (real championships, 1980-81 to 2022-23) and NCAA football
(AP #1 as champion, 1970-2023).
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
from durable_dominance import dominance as DOM
from durable_dominance import plotting as P
from durable_dominance import strength as S

REPORTS = Path(__file__).resolve().parents[1] / "reports"

PEDIGREE = ["prior_titles", "prior_titles_10y", "prior_top4", "prior_top8",
            "career_win_pct", "pedigree_index"]
BALANCE = ["win_pct_sd", "noll_scully", "mov_sd", "strength_sd",
           "top_bottom_gap", "hhi_wins", "gini_wins", "upset_rate"]

PRETTY = {
    "prior_titles": "prior titles", "prior_titles_10y": "titles, last 10 yrs",
    "prior_top4": "prior top-4 finishes", "prior_top8": "prior top-8 finishes",
    "career_win_pct": "career win %", "pedigree_index": "pedigree index",
    "win_pct_sd": "SD of win%", "noll_scully": "Noll-Scully",
    "mov_sd": "SD of margin", "strength_sd": "SD of fitted strength",
    "top_bottom_gap": "best-worst gap", "hhi_wins": "HHI of wins",
    "gini_wins": "Gini of wins", "upset_rate": "upset rate",
}


def build_nba():
    n, m = D.load_nba(), D.load_nba_matches()
    champions = m.groupby("season")["champion"].first()
    scale = S.calibrate_mov_scale(n, m)
    strengths = {s: S.season_strengths(n, m, s, scale)
                 for s in n["season"].unique()}
    ped = DOM.champion_pedigree(n, champions, order=D.season_key)
    comp = DOM.season_competitiveness(n, strengths, m)
    conc = DOM.rolling_title_concentration(champions, window=10)
    turn = DOM.elite_turnover(n, order=D.season_key)
    panel = (ped.merge(comp, on="season").merge(conc, on="season")
             .merge(turn, on="season"))
    panel["start_year"] = panel["season"].map(D.season_key)
    return panel.sort_values("start_year").reset_index(drop=True), strengths


def build_ncaaf():
    c = D.load_ncaaf()
    champions = c[c["ap_rank"] == 1].set_index("season")["team"]
    c = c.rename(columns={"srs": "mov"})
    ped = DOM.champion_pedigree(c, champions)
    comp = DOM.season_competitiveness(c)
    conc = DOM.rolling_title_concentration(champions, window=10)
    panel = ped.merge(comp, on="season").merge(conc, on="season")
    panel["start_year"] = panel["season"]
    return panel.sort_values("start_year").reset_index(drop=True)


def correlate(panel: pd.DataFrame, league: str) -> pd.DataFrame:
    """Every pedigree measure against every balance measure, raw and detrended.

    Both families drift over time -- pedigree counts accumulate, expansion
    changes the spread of records -- so a raw correlation between them can be
    two trends passing in the night. The detrended column removes a linear
    year effect from both sides first.
    """
    rows = []
    yr = panel["start_year"].to_numpy(dtype=float)
    for p in PEDIGREE:
        for b in BALANCE:
            if b not in panel or panel[b].notna().sum() < 10:
                continue
            d = panel[[p, b]].dropna()
            if len(d) < 10:
                continue
            x, y = d[b].to_numpy(), d[p].to_numpy()
            raw = stats.pearsonr(x, y)
            t = yr[d.index]
            rx = x - np.poly1d(np.polyfit(t, x, 1))(t)
            ry = y - np.poly1d(np.polyfit(t, y, 1))(t)
            det = stats.pearsonr(rx, ry)
            rows.append({
                "league": league, "pedigree": p, "balance": b, "n": len(d),
                "r": raw.statistic, "p": raw.pvalue,
                "r_detrended": det.statistic, "p_detrended": det.pvalue,
            })
    return pd.DataFrame(rows)


def fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values)."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def heatmap(ax, corr: pd.DataFrame, value: str, title: str):
    piv = corr.pivot(index="pedigree", columns="balance", values=value)
    piv = piv.reindex(index=[p for p in PEDIGREE if p in piv.index],
                      columns=[b for b in BALANCE if b in piv.columns])
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=-0.7, vmax=0.7, aspect="auto")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([PRETTY.get(c, c) for c in piv.columns], rotation=40,
                       ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([PRETTY.get(i, i) for i in piv.index], fontsize=8)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(v) > 0.45 else P.INK)
    ax.set_title(title, fontsize=11)
    ax.grid(False)
    return im


def main() -> None:
    P.apply_style()
    REPORTS.mkdir(parents=True, exist_ok=True)

    nba, strengths = build_nba()
    ncaaf = build_ncaaf()
    nba.to_csv(REPORTS / "champion_panel_nba.csv", index=False)
    ncaaf.to_csv(REPORTS / "champion_panel_ncaaf.csv", index=False)

    corr = pd.concat([correlate(nba, "NBA"), correlate(ncaaf, "NCAAF")],
                     ignore_index=True)

    # 84 correlations from one dataset: control the false discovery rate
    # (Benjamini-Hochberg) rather than reading raw p-values.
    corr["q_detrended"] = fdr(corr["p_detrended"].to_numpy())
    corr.to_csv(REPORTS / "pedigree_vs_balance.csv", index=False)
    sig = corr[corr["q_detrended"] < 0.10]

    print(f"Pedigree x balance pairs tested: {len(corr)}")
    print(f"  significant raw (p<.05):            {(corr['p'] < .05).sum()}")
    print(f"  significant detrended (p<.05):      {(corr['p_detrended'] < .05).sum()}"
          f"   (~{0.05*len(corr):.0f} expected by chance)")
    print(f"  surviving FDR correction (q<.10):   {len(sig)}")
    if len(sig):
        print("\nSurviving Benjamini-Hochberg:")
        print(sig[["league", "pedigree", "balance", "n", "r_detrended",
                   "p_detrended", "q_detrended"]].sort_values("q_detrended")
              .to_string(index=False, float_format=lambda v: f"{v:.4g}"))
        print("\nAll surviving measures are orientated so that a HIGHER value "
              "means a LESS balanced season,\nso a positive r means: less "
              "competitive seasons crown more decorated champions.")

    # ---------------- figure 1: correlation heatmaps ----------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 9.5))
    for row, (lg, panel) in enumerate([("NBA", nba), ("NCAAF", ncaaf)]):
        sub = corr[corr["league"] == lg]
        heatmap(axes[row, 0], sub, "r", f"{lg}: raw correlation")
        im = heatmap(axes[row, 1], sub, "r_detrended", f"{lg}: year-detrended")
    fig.colorbar(im, ax=axes, label="Pearson r", shrink=0.6, pad=0.02)
    fig.suptitle("Champion pedigree vs. league competitiveness",
                 fontweight="bold", y=0.98)
    print("\nwrote", P.save(fig, "11_pedigree_balance_heatmap.png").name)

    # ---------------- figure 2: the story over time ----------------
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 8.4))

    ax = axes[0, 0]
    ax.bar(nba["start_year"], nba["prior_titles"], color=P.INK, width=0.75)
    ax.set_xlabel("Season"); ax.set_ylabel("Champion's prior titles")
    ax.set_title("NBA champions arrive already decorated")
    for _, r in nba.nlargest(3, "prior_titles").iterrows():
        ax.annotate(r["champion"].split()[-1], (r["start_year"], r["prior_titles"]),
                    fontsize=8, ha="center", va="bottom", color=P.TRend)

    ax = axes[0, 1]
    ax.plot(nba["start_year"], nba["strength_sd"], color=P.INK, lw=1.6,
            label="SD of fitted strength")
    ax2 = ax.twinx()
    ax2.plot(nba["start_year"], nba["title_hhi"], color=P.TRend, lw=1.6,
             label="title HHI, trailing 10 yrs")
    ax2.grid(False)
    ax.set_xlabel("Season"); ax.set_ylabel("SD of strength", color=P.INK)
    ax2.set_ylabel("Title concentration", color=P.TRend)
    ax.set_title("Balance and title concentration over time")

    P.regression_scatter(
        axes[1, 0], nba["strength_sd"], nba["pedigree_index"],
        xlabel="SD of fitted strength (less balanced →)",
        ylabel="Champion pedigree index",
        title="NBA: does imbalance crown veterans?")

    P.regression_scatter(
        axes[1, 1], nba["upset_rate"], nba["pedigree_index"],
        xlabel="Upset rate (more balanced →)",
        ylabel="Champion pedigree index",
        title="NBA: pedigree vs. how often the weaker team won")

    fig.suptitle("NBA champions and the seasons that produced them",
                 fontweight="bold", y=1.01)
    print("wrote", P.save(fig, "11_nba_timeline.png").name)

    print("\nNBA champion pedigree summary")
    print(nba[PEDIGREE].describe().loc[["mean", "50%", "max"]]
          .to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"\nChampions with zero prior titles: "
          f"{(nba['prior_titles'] == 0).sum()} of {len(nba)} NBA seasons")


if __name__ == "__main__":
    main()
