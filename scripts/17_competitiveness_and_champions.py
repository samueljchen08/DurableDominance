"""Study F -- do tight regular seasons produce first-time champions?

Study A asked this as a correlation. This asks it the way you would actually
phrase it: **in the most competitive third of seasons, how often is the
champion a historical powerhouse, and how often is it somebody new?**

Two halves to define.

*Powerhouse* is measured three ways, all strictly backward-looking:

  has_won_before   the champion had at least one prior title
  top3_pedigree    it was among the three most decorated teams in the league
                   that season, by prior titles
  top_career       its career winning percentage before this season sits in the
                   top quartile of the teams active that season
  dynasty          it already had three or more titles

The middle two are *relative to contemporaries*, which matters: raw title
counts grow mechanically as history accumulates, so a fixed threshold would
call later champions powerhouses for no reason but the calendar. The four
definitions escalate, from "has won once" to "was already a dynasty".

*Competitive* is measured seven ways, including the one the question names --
the average team's absolute distance from the median win% -- plus the spread of
records, of scoring margin, and the rate at which the weaker side actually won.

The seven balance measures are near-duplicates of one another, so running all
7 x 4 combinations and correcting for 52 tests is both underpowered and
misleading. The **primary analysis** therefore collapses them into a single
tightness index -- the first principal component of the standardised measures,
oriented so higher means tighter -- and runs one logistic regression per
league and powerhouse definition, controlling for year. That is eight tests.

The full 7 x 4 tertile grid is kept as a robustness display: with measures this
correlated, the consistency of the *sign* across cells is more informative than
any individual p-value.

Leagues: NBA (real championships) and NCAA football (AP #1). The NFL is
excluded -- this repo has its standings but no record of who won the title.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from durable_dominance import datasets as D
from durable_dominance import dominance as DOM
from durable_dominance import metrics as M
from durable_dominance import plotting as P
from durable_dominance import strength as S

REPORTS = Path(__file__).resolve().parents[1] / "reports"

BALANCE = {
    "mad_from_median": "mean |win% − median|",
    "win_pct_sd": "SD of win%",
    "noll_scully": "Noll-Scully",
    "mov_sd": "SD of scoring margin",
    "gini_wins": "Gini of wins",
    "top_bottom_gap": "best − worst win%",
    "upset_rate": "upset rate",
}
# Every measure above rises with IMBALANCE except the upset rate, which rises
# with balance. Flipping it keeps "higher = tighter season" true throughout.
INVERTED = {"upset_rate"}

POWERHOUSE = {
    "has_won_before": "had won before",
    "top3_pedigree": "top-3 most decorated",
    "top_career": "top-quartile career win%",
    "dynasty": "already had 3+ titles",
}


def build_panel(league: str) -> pd.DataFrame:
    """One row per season: competitiveness measures plus champion pedigree."""
    if league == "NBA":
        st = D.load_nba()
        matches = D.load_nba_matches()
        champions = matches.groupby("season")["champion"].first()
        rank_col, rating = "rank", "mov"
        scale = S.calibrate_mov_scale(st, matches)
        strengths = {s: S.season_strengths(st, matches, s, scale)
                     for s in st["season"].unique()}
    else:
        st = D.load_ncaaf().rename(columns={"srs": "mov"})
        matches, strengths = None, None
        champions = st[st["ap_rank"] == 1].set_index("season")["team"]
        rank_col, rating = "rank", "mov"

    seasons = sorted(st["season"].dropna().unique(), key=D.season_key)

    titles: dict[str, int] = {}
    wins: dict[str, float] = {}
    games: dict[str, float] = {}
    rows = []

    for season in seasons:
        g = st[st["season"] == season]
        champ = champions.get(season)
        w = g["win_pct"].dropna()
        n_games = int(round((g["wins"] + g["losses"]).median()))

        row = {
            "league": league, "season": season,
            "start_year": D.season_key(season), "n_teams": len(g),
            "champion": champ,
            # the measure the question names: average distance from the median
            "mad_from_median": float((w - w.median()).abs().mean()),
            "win_pct_sd": M.win_pct_sd(w),
            "noll_scully": M.noll_scully(w, n_games) if n_games else np.nan,
            "mov_sd": float(g[rating].std(ddof=1)),
            "gini_wins": DOM.gini(g["wins"].to_numpy()),
            "top_bottom_gap": float(w.max() - w.min()),
        }
        if strengths is not None and season in strengths:
            row["upset_rate"] = DOM.upset_rate_from_matches(
                matches[matches["season"] == season], strengths[season]["bt"])

        if champ is not None and champ in set(g["team"]):
            active = list(g["team"])
            prior_titles = pd.Series({t: titles.get(t, 0) for t in active})
            prior_wp = pd.Series({
                t: (wins.get(t, 0) / games[t]) if games.get(t) else np.nan
                for t in active})
            row["prior_titles"] = int(prior_titles[champ])
            row["career_win_pct"] = float(prior_wp[champ]) if pd.notna(prior_wp[champ]) else np.nan
            row["has_won_before"] = int(prior_titles[champ] >= 1)
            # Rank among contemporaries rather than a fixed cut, so the bar
            # rises as the league's history deepens.
            rank = prior_titles.rank(ascending=False, method="min")[champ]
            row["top3_pedigree"] = int(rank <= 3 and prior_titles[champ] > 0)
            row["dynasty"] = int(prior_titles[champ] >= 3)
            valid = prior_wp.dropna()
            row["top_career"] = int(
                len(valid) > 4 and pd.notna(prior_wp[champ])
                and prior_wp[champ] >= valid.quantile(0.75))
            rows.append(row)

        # advance history only after the season is recorded
        for r in g.itertuples():
            wins[r.team] = wins.get(r.team, 0) + r.wins
            games[r.team] = games.get(r.team, 0) + r.wins + r.losses
        if champ is not None:
            titles[champ] = titles.get(champ, 0) + 1

    panel = pd.DataFrame(rows)
    # Drop the earliest seasons, which have no history to judge a champion by.
    return panel[panel["start_year"] > panel["start_year"].min() + 4].reset_index(drop=True)


INDEX_MEASURES = ["mad_from_median", "win_pct_sd", "noll_scully", "mov_sd",
                  "gini_wins", "top_bottom_gap"]


def add_tightness_index(panel: pd.DataFrame) -> pd.DataFrame:
    """First principal component of the balance measures, per league.

    The measures correlate at 0.8-0.99 with each other, so one component
    carries nearly all of the shared variation and gives a single,
    better-powered test instead of seven overlapping ones.
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    out = []
    for league, g in panel.groupby("league"):
        g = g.copy()
        X = StandardScaler().fit_transform(g[INDEX_MEASURES].to_numpy())
        pca = PCA(n_components=1).fit(X)
        pc = pca.transform(X)[:, 0]
        # every input rises with imbalance, so flip to make high = tighter
        if np.corrcoef(pc, g["win_pct_sd"])[0, 1] > 0:
            pc = -pc
        g["tightness_index"] = pc
        g.attrs["explained"] = float(pca.explained_variance_ratio_[0])
        print(f"  {league}: tightness index explains "
              f"{pca.explained_variance_ratio_[0]*100:.0f}% of the variance "
              f"in {len(INDEX_MEASURES)} balance measures")
        out.append(g)
    return pd.concat(out, ignore_index=True)


def primary_test(panel: pd.DataFrame) -> pd.DataFrame:
    """One logistic regression per league x powerhouse definition."""
    rows = []
    for league, g in panel.groupby("league"):
        d = g.dropna(subset=["tightness_index"]).copy()
        d["tight_z"] = ((d["tightness_index"] - d["tightness_index"].mean())
                        / d["tightness_index"].std(ddof=0))
        d["year_z"] = ((d["start_year"] - d["start_year"].mean())
                       / d["start_year"].std(ddof=0))
        for ph, label in POWERHOUSE.items():
            y = d[ph].astype(float)
            if y.nunique() < 2:
                continue
            fit = sm.Logit(y, sm.add_constant(d[["tight_z", "year_z"]])).fit(disp=0)
            coef = float(fit.params["tight_z"])
            lo, hi = fit.conf_int().loc["tight_z"]
            rows.append({
                "league": league, "powerhouse": label, "n": len(d),
                "base_rate": float(y.mean()),
                "odds_ratio": float(np.exp(coef)),
                "or_lo": float(np.exp(lo)), "or_hi": float(np.exp(hi)),
                "coef": coef, "p": float(fit.pvalues["tight_z"]),
            })
    out = pd.DataFrame(rows)
    out["q"] = fdr(out["p"].to_numpy())
    return out


def tertile_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Powerhouse rate in the tightest vs. loosest third of seasons."""
    rows = []
    for league, g in panel.groupby("league"):
        for bal, bal_label in BALANCE.items():
            if bal not in g or g[bal].notna().sum() < 12:
                continue
            d = g.dropna(subset=[bal])
            # orient so a HIGH value always means a TIGHTER season
            x = -d[bal] if bal not in INVERTED else d[bal]
            lo, hi = x.quantile([1 / 3, 2 / 3])
            tight, loose = x >= hi, x <= lo
            for ph, ph_label in POWERHOUSE.items():
                a, b = int(d.loc[tight, ph].sum()), int((~d.loc[tight, ph].astype(bool)).sum())
                c, e = int(d.loc[loose, ph].sum()), int((~d.loc[loose, ph].astype(bool)).sum())
                odds, p = stats.fisher_exact([[a, b], [c, e]])
                rows.append({
                    "league": league, "balance": bal_label, "powerhouse": ph_label,
                    "n_tight": a + b, "n_loose": c + e,
                    "rate_tight": a / (a + b) if a + b else np.nan,
                    "rate_loose": c / (c + e) if c + e else np.nan,
                    "difference": (a / (a + b) - c / (c + e)) if (a + b) and (c + e) else np.nan,
                    "odds_ratio": odds, "p_fisher": p,
                })
    return pd.DataFrame(rows)


def logistic_table(panel: pd.DataFrame) -> pd.DataFrame:
    """Logistic regression of powerhouse status on tightness, controlling for year."""
    rows = []
    for league, g in panel.groupby("league"):
        for bal, bal_label in BALANCE.items():
            if bal not in g or g[bal].notna().sum() < 12:
                continue
            d = g.dropna(subset=[bal]).copy()
            tight = -d[bal] if bal not in INVERTED else d[bal]
            d["tight_z"] = (tight - tight.mean()) / tight.std(ddof=0)
            d["year_z"] = ((d["start_year"] - d["start_year"].mean())
                           / d["start_year"].std(ddof=0))
            for ph, ph_label in POWERHOUSE.items():
                y = d[ph].astype(float)
                if y.nunique() < 2:
                    continue
                try:
                    fit = sm.Logit(y, sm.add_constant(d[["tight_z", "year_z"]])).fit(disp=0)
                    rows.append({
                        "league": league, "balance": bal_label,
                        "powerhouse": ph_label, "n": len(d),
                        "coef_tight": float(fit.params["tight_z"]),
                        "p_tight": float(fit.pvalues["tight_z"]),
                    })
                except Exception:
                    continue
    return pd.DataFrame(rows)


def fdr(p: np.ndarray) -> np.ndarray:
    n = len(p)
    o = np.argsort(p)
    r = p[o] * n / np.arange(1, n + 1)
    r = np.minimum.accumulate(r[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.clip(r, 0, 1)
    return out


def main() -> None:
    P.apply_style()
    panel = pd.concat([build_panel("NBA"), build_panel("NCAAF")],
                      ignore_index=True)
    print("Building the combined tightness index:")
    panel = add_tightness_index(panel)
    panel.to_csv(REPORTS / "competitiveness_champion_panel.csv", index=False)
    print("Season-champion pairs:")
    print(panel.groupby("league").agg(
        n=("season", "size"),
        first=("start_year", "min"), last=("start_year", "max"),
        pct_won_before=("has_won_before", "mean"),
        pct_top3=("top3_pedigree", "mean"),
        pct_top_career=("top_career", "mean"),
        pct_dynasty=("dynasty", "mean"),
    ).to_string(float_format=lambda v: f"{v:.3f}"))

    primary = primary_test(panel)
    primary.to_csv(REPORTS / "competitiveness_primary.csv", index=False)
    print("\nPRIMARY TEST — odds of a powerhouse champion per 1 SD tighter season")
    print("(odds ratio below 1 = tighter seasons crown FEWER powerhouses)")
    print(primary[["league", "powerhouse", "n", "base_rate", "odds_ratio",
                   "or_lo", "or_hi", "p", "q"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n  {(primary['p'] < .05).sum()}/{len(primary)} significant raw, "
          f"{(primary['q'] < .10).sum()} surviving FDR; "
          f"{(primary['odds_ratio'] < 1).sum()}/{len(primary)} point the same way "
          f"(fewer powerhouses in tighter seasons)")

    tert = tertile_table(panel)
    tert["q_fisher"] = fdr(tert["p_fisher"].to_numpy())
    tert.to_csv(REPORTS / "competitiveness_tertiles.csv", index=False)

    logit = logistic_table(panel)
    logit["q_tight"] = fdr(logit["p_tight"].to_numpy())
    logit.to_csv(REPORTS / "competitiveness_logit.csv", index=False)

    print("\nPowerhouse rate: tightest third vs. loosest third of seasons")
    show = tert[["league", "balance", "powerhouse", "rate_tight", "rate_loose",
                 "difference", "p_fisher"]]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    print(f"\nFisher tests: {len(tert)} run, "
          f"{(tert['p_fisher'] < .05).sum()} significant raw, "
          f"{(tert['q_fisher'] < .10).sum()} surviving FDR")
    print(f"Logistic (year-controlled): {len(logit)} run, "
          f"{(logit['p_tight'] < .05).sum()} significant raw, "
          f"{(logit['q_tight'] < .10).sum()} surviving FDR")

    n_neg = int((tert["difference"] < 0).sum())
    print(f"\nSign consistency across the 7x4 robustness grid: "
          f"{n_neg}/{len(tert)} cells show a LOWER powerhouse rate in tighter "
          f"seasons.\n  (The measures are near-duplicates, so treat this as one "
          f"consistent pattern, not {len(tert)} independent confirmations.)")

    sig = tert[tert["q_fisher"] < 0.10]
    if len(sig):
        print("\nSurviving FDR:")
        print(sig.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    else:
        mean_diff = tert["difference"].mean()
        print(f"\nNothing survives. Mean difference in powerhouse rate between "
              f"the tightest and loosest third: {mean_diff:+.3f} "
              f"({tert['difference'].gt(0).sum()}/{len(tert)} positive).")

    # ---------------- figures ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4),
                             gridspec_kw={"width_ratios": [1.15, 1, 1.25]})

    # (1) the primary test as a forest plot
    ax = axes[0]
    pr = primary.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(pr))
    for i, r in pr.iterrows():
        col = P.TRend if r["q"] < 0.10 else "#7C8A96"
        ax.plot([r["or_lo"], r["or_hi"]], [i, i], color=col, lw=2.4,
                solid_capstyle="round")
        ax.plot(r["odds_ratio"], i, "o", color=col, ms=8, zorder=3)
    ax.axvline(1, color=P.INK, lw=1.3)
    ax.set_xscale("log")
    ax.set_xticks([0.1, 0.25, 0.5, 1, 2])
    ax.set_xticklabels(["0.1", "0.25", "0.5", "1", "2"])
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['league']}  ·  {r['powerhouse']}"
                        for _, r in pr.iterrows()], fontsize=8.5)
    ax.set_xlabel("odds ratio per 1 SD tighter season  (log scale)")
    ax.set_title("Primary test: tighter seasons crown\nless-established champions")
    from matplotlib.lines import Line2D
    ax.legend(handles=[
        Line2D([], [], color=P.TRend, marker="o", lw=2.4, ms=7,
               label="survives FDR (q < .10)"),
        Line2D([], [], color="#7C8A96", marker="o", lw=2.4, ms=7,
               label="not significant")],
        fontsize=7.5, loc="upper left", framealpha=.9)

    # (2) powerhouse rate by tertile of the combined index
    ax = axes[1]
    width = 0.2
    xs = np.arange(len(POWERHOUSE))
    for k, (league, mark) in enumerate([("NBA", 0), ("NCAAF", 1)]):
        g = panel[panel["league"] == league]
        lo, hi = g["tightness_index"].quantile([1 / 3, 2 / 3])
        tight = g[g["tightness_index"] >= hi]
        loose = g[g["tightness_index"] <= lo]
        for j, ph in enumerate(POWERHOUSE):
            ax.bar(j + (k * 2 - 1) * width * 1.05 - width / 2,
                   tight[ph].mean(), width, color=["#A0701F", "#8A5E19"][k],
                   label=f"{league} tightest third" if j == 0 else None)
            ax.bar(j + (k * 2 - 1) * width * 1.05 + width / 2,
                   loose[ph].mean(), width, color=["#C9B48A", "#B5A075"][k],
                   label=f"{league} loosest third" if j == 0 else None)
    ax.set_xticks(xs)
    ax.set_xticklabels(["had won\nbefore", "top-3 most\ndecorated",
                        "top-quartile\ncareer win%", "already had\n3+ titles"],
                       fontsize=7.5)
    ax.set_ylabel("share of champions")
    ax.set_ylim(0, 1.14)
    ax.set_title("How often the champion was\nalready established")
    ax.legend(fontsize=7)

    # (3) the robustness grid
    ax = axes[2]
    piv = tert.assign(row=tert["league"] + " · " + tert["balance"]).pivot(
        index="row", columns="powerhouse", values="difference")
    piv = piv[[v for v in POWERHOUSE.values() if v in piv.columns]]
    im = ax.imshow(piv.values, cmap="RdBu", vmin=-0.55, vmax=0.55, aspect="auto")
    fig.colorbar(im, ax=ax, label="tight − loose powerhouse rate", shrink=0.85)
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([c.replace(" ", "\n", 1) for c in piv.columns], fontsize=7)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=6.8)
    ax.set_title("Robustness: every balance measure\n× every powerhouse definition")
    ax.grid(False)

    fig.suptitle("Does a tight regular season crown a new champion?",
                 fontweight="bold", y=1.02)
    print("\nwrote", P.save(fig, "17_competitiveness_champions.png").name)

    # second figure: the question's own measure, season by season
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    for ax, league in zip(axes, ["NBA", "NCAAF"]):
        g = panel[panel["league"] == league]
        colors = [P.TRend if v else "#3D6B8C" for v in g["has_won_before"]]
        ax.scatter(g["start_year"], g["mad_from_median"], c=colors, s=54,
                   zorder=3, edgecolor="white", linewidth=.6)
        ax.set_xlabel("Season")
        ax.set_ylabel("mean |win% − median|  (lower = tighter)")
        ax.set_title(f"{league}: season tightness, coloured by champion type")
    axes[0].scatter([], [], c=P.TRend, label="champion had won before")
    axes[0].scatter([], [], c="#3D6B8C", label="first-time champion")
    axes[0].legend(fontsize=8)
    fig.suptitle("The question's own measure, season by season",
                 fontweight="bold", y=1.02)
    print("wrote", P.save(fig, "17_tightness_timeline.png").name)


if __name__ == "__main__":
    main()
