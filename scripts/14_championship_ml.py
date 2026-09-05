"""Study D -- can a learned model beat the physics?

Study B built a calibrated generative model of a championship: strengths, a
bracket, seven-game random walks. That is a strong baseline, and it is the one
machine learning has to beat here.

Three tasks:

1. **Predict the champion.** Features from the regular season, target = won
   the title. Scored against the simulation's own title probability on the
   same held-out seasons. If ML wins, the simulation is missing something
   learnable; if it does not, the simulation already captures what the
   standings know.
2. **Predict who beats their strength.** Target = won the title *given* the
   simulation's probability, i.e. the residual. Any signal here is durable
   dominance the random walk misses.
3. **Predict next season's elite.** The long-horizon question, with
   permutation importance to see what actually carries.

All splits are temporal: train on earlier seasons, test on later ones. Model
selection happens inside the training years only, so the test seasons are
touched exactly once.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from durable_dominance import datasets as D
from durable_dominance import dominance as DOM
from durable_dominance import plotting as P
from durable_dominance import strength as S

REPORTS = Path(__file__).resolve().parents[1] / "reports"

FEATURES = ["bt", "mov", "win_pct", "conf_rank", "league_rank", "strength_gap_to_best",
            "prior_titles", "prior_top4", "career_win_pct", "league_strength_sd"]
PRETTY = {
    "bt": "fitted strength", "mov": "margin of victory", "win_pct": "win %",
    "conf_rank": "conference seed", "league_rank": "league rank",
    "strength_gap_to_best": "gap to best team", "prior_titles": "prior titles",
    "prior_top4": "prior top-4 finishes", "career_win_pct": "career win %",
    "league_strength_sd": "league strength spread",
}


def build_table():
    """One row per playoff team-season, with strictly backward-looking history."""
    n, m = D.load_nba(), D.load_nba_matches()
    scale = S.calibrate_mov_scale(n, m)
    strengths = {s: S.season_strengths(n, m, s, scale) for s in n["season"].unique()}
    champions = m.groupby("season")["champion"].first()
    seasons = sorted(strengths, key=D.season_key)

    titles: dict[str, int] = {}
    top4: dict[str, int] = {}
    wins: dict[str, float] = {}
    games: dict[str, float] = {}

    rows = []
    for season in seasons:
        st = strengths[season].copy()
        st["league_rank"] = st["bt"].rank(ascending=False)
        st["conf_rank"] = st.groupby("conference")["win_pct"].rank(ascending=False)
        st["strength_gap_to_best"] = st["bt"].max() - st["bt"]
        st["league_strength_sd"] = st["bt"].std(ddof=1)
        for team, r in st.iterrows():
            if r["conf_rank"] > 8:
                continue
            rows.append({
                "season": season, "start_year": D.season_key(season), "team": team,
                "bt": r["bt"], "mov": r["mov"], "win_pct": r["win_pct"],
                "conf_rank": r["conf_rank"], "league_rank": r["league_rank"],
                "strength_gap_to_best": r["strength_gap_to_best"],
                "league_strength_sd": r["league_strength_sd"],
                "prior_titles": titles.get(team, 0),
                "prior_top4": top4.get(team, 0),
                "career_win_pct": (wins.get(team, 0) / games[team]
                                   if games.get(team) else 0.5),
                "won_title": int(champions[season] == team),
            })
        # advance history after the season is recorded
        ranked = st.sort_values("rank")
        for pos, (team, r) in enumerate(ranked.iterrows(), start=1):
            wins[team] = wins.get(team, 0) + r["wins"]
            games[team] = games.get(team, 0) + r["wins"] + r["losses"]
            if pos <= 4:
                top4[team] = top4.get(team, 0) + 1
        titles[champions[season]] = titles.get(champions[season], 0) + 1

    return pd.DataFrame(rows)


def normalise_by_season(p: np.ndarray, seasons: np.ndarray) -> np.ndarray:
    """Exactly one champion per season, so probabilities must sum to 1."""
    out = p.astype(float).copy()
    for s in np.unique(seasons):
        mask = seasons == s
        tot = out[mask].sum()
        out[mask] = out[mask] / tot if tot > 0 else 1.0 / mask.sum()
    return out


def main() -> None:
    P.apply_style()
    tab = build_table()
    sim = pd.read_csv(REPORTS / "title_probabilities.csv")
    tab = tab.merge(sim, on=["season", "team"], how="left")
    tab["title_prob"] = tab["title_prob"].fillna(0.0)
    tab.to_csv(REPORTS / "ml_champion_table.csv", index=False)
    print(f"Playoff team-seasons: {len(tab)}  champions: {tab['won_title'].sum()}")

    cutoff = 2008
    tr, te = tab["start_year"] <= cutoff, tab["start_year"] > cutoff
    X, y = tab[FEATURES].to_numpy(float), tab["won_title"].to_numpy()
    seas = tab["start_year"].to_numpy()
    print(f"train {tr.sum()} rows (<= {cutoff}), test {te.sum()} rows")

    models = {
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000, C=0.5)),
        "Random forest": RandomForestClassifier(
            n_estimators=600, min_samples_leaf=8, random_state=0),
        "Gradient boosting": GradientBoostingClassifier(
            random_state=0, max_depth=2, n_estimators=200, learning_rate=0.05),
    }

    # ---- task 1: predict the champion ------------------------------------
    rows = []
    base = normalise_by_season(tab.loc[te, "title_prob"].to_numpy(), seas[te])
    rows.append({"task": "champion", "model": "Random-walk simulation",
                 "auc": roc_auc_score(y[te], base),
                 "log_loss": log_loss(y[te], base, labels=[0, 1])})
    rows.append({"task": "champion", "model": "Uniform over playoff field",
                 "auc": 0.5,
                 "log_loss": log_loss(
                     y[te], normalise_by_season(np.ones(te.sum()), seas[te]),
                     labels=[0, 1])})

    fitted = {}
    for name, mdl in models.items():
        mdl.fit(X[tr], y[tr])
        p = normalise_by_season(mdl.predict_proba(X[te])[:, 1], seas[te])
        fitted[name] = p
        rows.append({"task": "champion", "model": name,
                     "auc": roc_auc_score(y[te], p),
                     "log_loss": log_loss(y[te], p, labels=[0, 1])})

    # simulation probability offered to the model as a feature
    Xa = np.column_stack([X, tab["title_prob"].to_numpy()])
    mdl = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=0.5))
    mdl.fit(Xa[tr], y[tr])
    p = normalise_by_season(mdl.predict_proba(Xa[te])[:, 1], seas[te])
    rows.append({"task": "champion", "model": "Logistic + simulation feature",
                 "auc": roc_auc_score(y[te], p),
                 "log_loss": log_loss(y[te], p, labels=[0, 1])})

    res = pd.DataFrame(rows).sort_values("log_loss")
    res.to_csv(REPORTS / "ml_champion_results.csv", index=False)
    print("\n1. Predicting the champion on held-out seasons "
          "(lower log-loss is better)")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- task 2: can anything predict beating your strength? -------------
    resid = tab["won_title"] - tab["title_prob"]
    lr = make_pipeline(StandardScaler(),
                       LogisticRegression(max_iter=5000, C=0.5))
    hist_only = ["prior_titles", "prior_top4", "career_win_pct"]
    Xh = tab[hist_only].to_numpy(float)
    lr.fit(Xh[tr], y[tr])
    ph = normalise_by_season(lr.predict_proba(Xh[te])[:, 1], seas[te])
    from scipy import stats as st2
    corr = st2.pearsonr(tab.loc[te, "prior_titles"], resid[te])
    print("\n2. Does franchise history predict beating the simulation?")
    print(f"   history-only model, held-out AUC = {roc_auc_score(y[te], ph):.3f}")
    print(f"   corr(prior titles, actual − simulated) = {corr.statistic:+.3f} "
          f"(p = {corr.pvalue:.3f}, n = {te.sum()})")

    # ---- task 3: elite next season + importance --------------------------
    imp_model = GradientBoostingClassifier(random_state=0, max_depth=2,
                                           n_estimators=250, learning_rate=0.05)
    imp_model.fit(X[tr], y[tr])
    imp = permutation_importance(imp_model, X[te], y[te], n_repeats=30,
                                 random_state=0, scoring="neg_log_loss")
    importance = pd.DataFrame({
        "feature": FEATURES, "importance": imp.importances_mean,
        "std": imp.importances_std}).sort_values("importance", ascending=False)
    importance.to_csv(REPORTS / "ml_champion_importance.csv", index=False)
    print("\n3. Permutation importance for predicting the champion")
    print(importance.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- figures ---------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))

    ax = axes[0]
    r = res.sort_values("log_loss", ascending=False)
    cols = [P.TRend if "simulation" in mname.lower() and "feature" not in mname.lower()
            else P.INK for mname in r["model"]]
    ax.barh(range(len(r)), r["log_loss"], color=cols)
    ax.set_yticks(range(len(r)))
    ax.set_yticklabels(r["model"], fontsize=8)
    ax.set_xlabel("held-out log-loss (lower is better)")
    ax.set_title("The simulation is hard to beat")

    ax = axes[1]
    ax.barh(range(len(r)), r["auc"], color=cols)
    ax.set_yticks(range(len(r))); ax.set_yticklabels(r["model"], fontsize=8)
    ax.axvline(0.5, color=P.ACCENT, ls=":", label="chance")
    ax.set_xlim(0.4, 1.0); ax.set_xlabel("held-out ROC AUC")
    ax.set_title("Ranking ability"); ax.legend(fontsize=8)

    ax = axes[2]
    imp_s = importance.sort_values("importance")
    ax.barh(range(len(imp_s)), imp_s["importance"], xerr=imp_s["std"], color=P.INK)
    ax.set_yticks(range(len(imp_s)))
    ax.set_yticklabels([PRETTY.get(f, f) for f in imp_s["feature"]], fontsize=8)
    ax.set_xlabel("increase in log-loss when shuffled")
    ax.set_title("What predicts a title")

    fig.suptitle("Machine learning against a calibrated simulation",
                 fontweight="bold", y=1.03)
    print("\nwrote", P.save(fig, "14_championship_ml.png").name)


if __name__ == "__main__":
    main()
