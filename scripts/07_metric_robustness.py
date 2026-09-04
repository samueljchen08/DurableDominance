"""Do the deck's conclusions survive fixing the metrics and the alignment?

Three audits:

A. **Row-order alignment.** The original NFL script builds its dominance list
   from `df['Year'].unique()` (file order, descending) but its balance list
   from `groupby('Year')` (ascending). Slicing both with `[:48]` pairs each
   season's balance with a different season's dominance. This re-runs the NFL
   result both ways.

B. **Spurious trend.** NCAA football's dominance measure is a cumulative
   count, so it rises with time by construction. If balance also trends, the
   two correlate for reasons that have nothing to do with dominance. This
   re-runs it on year-detrended residuals.

C. **Metric choice.** The deck's "standard deviation of win%" is
   `sqrt(sum |w - .5|)`, not a standard deviation. This re-runs every headline
   regression under six balance measures.

D. **League-size confounding.** The two metrics that come out significant are
   the two that are not scale-free. `paper_cb_win` sums `|w - .5|` without
   dividing by n, so it grows with the number of teams; `hhi_wins` has a 1/n
   floor, so it shrinks with it. The NBA expanded from 23 teams to 30 across
   the sample. This re-runs every regression with team count as a covariate.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from durable_dominance import datasets as D
from durable_dominance import leagues as L
from durable_dominance import metrics as M
from durable_dominance import plotting as P

REPORTS = Path(__file__).resolve().parents[1] / "reports"

BALANCE_METRICS = [
    ("paper_cb_win", "deck's win% spread"),
    ("win_pct_sd", "SD of win%"),
    ("noll_scully", "Noll-Scully"),
    ("hhi_wins", "HHI of wins"),
    ("rsd_wins", "RSD of wins"),
    ("closeness", "mean |win% - .5|"),
]


def audit_nfl_alignment() -> pd.DataFrame:
    """Reproduce the original NFL pairing, then the label-aligned one."""
    r = pd.read_excel(D.PATHS["nfl"])
    r.columns = ["Team", "Wins", "Losses", "Ties", "Win%", "PF", "PA", "MOV", "Year"]

    def hist(df, cur):
        h = df[df["Year"] < cur]
        if h.empty:
            return pd.Series(dtype=float)
        elite = h[h["Wins"] >= 12]
        slots = elite.groupby("Year")["Team"].nunique().sum()
        return (elite.groupby("Team").size() / slots) * 100 if slots else pd.Series(dtype=float)

    seasons = r["Year"].unique()          # file order: descending
    dom = {}
    for cur in seasons[1:]:
        top4 = r[r["Year"] == cur].nlargest(4, "Wins")["Team"]
        dom[cur] = hist(r, cur).reindex(top4, fill_value=0).sum()

    cb = {y: math.sqrt(sum(abs(w - 0.5) for w in g["Win%"]))
          for y, g in r[r["Wins"] > 0].groupby("Year")}   # groupby: ascending

    dom_list, cb_list = list(dom.values()), list(cb.values())
    rows = []
    f = stats.linregress(cb_list[:48], dom_list[:48])
    rows.append({"pairing": "original script (positional [:48])",
                 "n": 48, "slope": f.slope, "r_squared": f.rvalue**2,
                 "p_value": f.pvalue})

    aligned = pd.DataFrame({"season": list(dom), "dom": dom_list})
    aligned["cb"] = aligned["season"].map(cb)
    a = aligned[aligned["season"] >= 1975]
    f2 = stats.linregress(a["cb"], a["dom"])
    rows.append({"pairing": "aligned on season label (1975+)",
                 "n": len(a), "slope": f2.slope, "r_squared": f2.rvalue**2,
                 "p_value": f2.pvalue})
    return pd.DataFrame(rows)


def audit_detrending() -> pd.DataFrame:
    """Raw vs year-detrended dominance/balance correlations."""
    rows = []
    for league, cb_col in [("nba", "paper_cb_win"), ("nfl", "paper_cb_win"),
                           ("ncaaf", "rating_sd"), ("ncaaf", "win_pct_sd")]:
        panel = L.PANELS[league]().dropna(subset=["dominance", cb_col])
        x, y, t = panel[cb_col].values, panel["dominance"].values, panel["start_year"].values
        raw = stats.linregress(x, y)
        rx = x - np.poly1d(np.polyfit(t, x, 1))(t)
        ry = y - np.poly1d(np.polyfit(t, y, 1))(t)
        det = stats.linregress(rx, ry)
        dom_trend = stats.linregress(t, y)
        rows.append({
            "league": league, "balance_metric": cb_col, "n": len(panel),
            "raw_r2": raw.rvalue**2, "raw_p": raw.pvalue,
            "dominance_vs_year_r2": dom_trend.rvalue**2,
            "detrended_r2": det.rvalue**2, "detrended_p": det.pvalue,
        })
    return pd.DataFrame(rows)


def audit_metric_choice() -> pd.DataFrame:
    """Every headline regression, under six different balance measures."""
    rows = []
    windows = {"nba": [("1981-2000", 1981, 2000), ("2000-Present", 2000, 2100),
                       ("Full", 1981, 2100)],
               "nfl": [("1975-Present", 1975, 2100), ("Full", 1970, 2100)],
               "ncaaf": [("Full", 1970, 2100)]}
    for league, wins in windows.items():
        panel = L.PANELS[league]()
        for label, lo, hi in wins:
            sub = panel[(panel["start_year"] >= lo) & (panel["start_year"] <= hi)]
            for col, pretty in BALANCE_METRICS:
                s = sub.dropna(subset=["dominance", col])
                if len(s) < 6:
                    continue
                f = stats.linregress(s[col], s["dominance"])
                rows.append({"league": league, "window": label, "metric": pretty,
                             "n": len(s), "slope": f.slope,
                             "r_squared": f.rvalue**2, "p_value": f.pvalue,
                             "significant": f.pvalue < 0.05})
    return pd.DataFrame(rows)


def audit_league_size() -> pd.DataFrame:
    """Add team count (and year) as covariates and see what survives."""
    import statsmodels.api as sm

    rows = []
    windows = {"nba": [("1981-2000", 1981, 2000), ("Full", 1981, 2100)],
               "nfl": [("Full", 1970, 2100)],
               "ncaaf": [("Full", 1970, 2100)]}
    for league, wins in windows.items():
        panel = L.PANELS[league]().dropna(subset=["dominance"])
        for label, lo, hi in wins:
            sub = panel[(panel["start_year"] >= lo) & (panel["start_year"] <= hi)]
            for col, pretty in BALANCE_METRICS:
                d = sub.dropna(subset=[col])
                if len(d) < 8 or d["n_teams"].nunique() < 2:
                    continue
                out = {"league": league, "window": label, "metric": pretty,
                       "n": len(d),
                       "size_r2": stats.linregress(d["n_teams"], d[col]).rvalue ** 2}
                for tag, controls in (("raw", []), ("size", ["n_teams"]),
                                      ("year", ["start_year"]),
                                      ("size_year", ["n_teams", "start_year"])):
                    X = sm.add_constant(d[[col] + controls])
                    fit = sm.OLS(d["dominance"], X).fit()
                    out[f"p_{tag}"] = float(fit.pvalues[col])
                rows.append(out)
    return pd.DataFrame(rows)


def main() -> None:
    P.apply_style()
    REPORTS.mkdir(parents=True, exist_ok=True)

    nfl = audit_nfl_alignment()
    detr = audit_detrending()
    metric = audit_metric_choice()
    size = audit_league_size()

    nfl.to_csv(REPORTS / "audit_nfl_alignment.csv", index=False)
    detr.to_csv(REPORTS / "audit_detrending.csv", index=False)
    metric.to_csv(REPORTS / "audit_metric_choice.csv", index=False)
    size.to_csv(REPORTS / "audit_league_size.csv", index=False)

    print("A. NFL row-order alignment\n" + nfl.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\nB. Time-trend confounding\n" + detr.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    print("\nC. Metric sensitivity (significant at p<.05)")
    print(metric.groupby(["league", "window"])["significant"].agg(["sum", "count"])
          .rename(columns={"sum": "n_significant", "count": "n_metrics"}).to_string())

    print("\nD. Controlling for league size (p-value of the balance coefficient)")
    print(size[["league", "window", "metric", "size_r2", "p_raw", "p_size",
                "p_year", "p_size_year"]]
          .to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n   metrics significant raw: {(size['p_raw'] < .05).sum()}/{len(size)}"
          f"   |  after controlling for team count: "
          f"{(size['p_size'] < .05).sum()}/{len(size)}")

    # ---- figure ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))

    ax = axes[0]
    ax.barh(nfl["pairing"], nfl["slope"], color=[P.TRend, P.INK], height=0.5)
    ax.axvline(0, color=P.INK, lw=1)
    for i, row in nfl.reset_index().iterrows():
        ax.text(row["slope"], i, f"  p={row['p_value']:.3f}", va="center",
                fontsize=9, ha="left" if row["slope"] > 0 else "right")
    ax.set_xlabel("slope of dominance on balance")
    ax.set_title("A. NFL: the sign flips when\nseasons are aligned correctly")
    ax.set_yticks(range(len(nfl)))
    ax.set_yticklabels(["original\n(positional slice)", "aligned\non season"], fontsize=9)

    ax = axes[1]
    idx = np.arange(len(detr))
    ax.bar(idx - 0.2, detr["raw_r2"], 0.4, label="raw $R^2$", color=P.TRend)
    ax.bar(idx + 0.2, detr["detrended_r2"], 0.4, label="detrended $R^2$", color=P.INK)
    ax.set_xticks(idx)
    ax.set_xticklabels([f"{r.league}\n{r.balance_metric}" for r in detr.itertuples()],
                       fontsize=8)
    ax.set_ylabel("$R^2$"); ax.legend(fontsize=8)
    ax.set_title("B. Removing the shared time trend\nerases most of the fit")

    ax = axes[2]
    nba = size[(size["league"] == "nba") & (size["window"] == "Full")]
    idx = np.arange(len(nba))
    ax.barh(idx + 0.2, nba["p_raw"], 0.4, label="raw", color=P.TRend)
    ax.barh(idx - 0.2, nba["p_size"], 0.4, label="controlling for team count",
            color=P.INK)
    ax.axvline(0.05, color="#2bb7a8", ls="--", lw=1.5, label="p = .05")
    ax.set_yticks(idx)
    ax.set_yticklabels(nba["metric"], fontsize=8)
    ax.set_xlabel("p-value of the balance coefficient")
    ax.set_xlim(0, 1)
    ax.set_title("C. NBA: every metric goes null once\nleague size is controlled")
    ax.legend(fontsize=7, loc="lower right")

    fig.suptitle("Robustness audit of the published relationships",
                 fontweight="bold", y=1.04)
    print("\nwrote", P.save(fig, "07_robustness.png").name)


if __name__ == "__main__":
    main()
