"""Machine learning on the dominance data.

Four experiments, each chosen because it answers something the linear
regressions in the deck cannot:

1. **Regularised season-level regression.** The six balance metrics are highly
   collinear, so ordinary least squares cannot say which one matters. Lasso /
   ElasticNet / Random Forest are scored with *forward-chaining* cross
   validation (train on the past, test on the future) so the reported skill is
   honest out-of-sample skill, not in-sample fit.

2. **Data-driven eras.** The deck splits history at 1981 / 1990 / 2000 / 2003
   by hand. PCA + k-means on the balance metrics lets the data choose its own
   regimes, and a change-point scan asks where dominance actually shifts.

3. **Team-level persistence.** With ~9k team-seasons instead of ~50 league
   seasons there is enough data to learn something. Task: given a team's
   season, will it be elite *next* season? Gradient boosting vs. logistic
   regression, split strictly by time.

4. **Permutation importance.** Which features carry the signal, measured on
   held-out data rather than by reading model coefficients.

Every model is compared against a baseline that predicts the training mean (or
base rate). A model that cannot beat that baseline is reported as no signal.
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
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestRegressor,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import ElasticNetCV, LassoCV, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from durable_dominance import datasets as D
from durable_dominance import leagues as L
from durable_dominance import plotting as P

REPORTS = Path(__file__).resolve().parents[1] / "reports"
FEATURES = ["paper_cb_win", "win_pct_sd", "noll_scully", "hhi_wins",
            "rsd_wins", "closeness", "rating_sd"]


# ---------------------------------------------------------------- experiment 1
def season_level_models() -> pd.DataFrame:
    """Forward-chaining CV: can balance metrics predict dominance out of sample?"""
    rows = []
    models = {
        "Baseline (train mean)": lambda: DummyRegressor(strategy="mean"),
        "Lasso": lambda: make_pipeline(StandardScaler(), LassoCV(cv=3, max_iter=20000)),
        "ElasticNet": lambda: make_pipeline(StandardScaler(),
                                            ElasticNetCV(cv=3, max_iter=20000)),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=0),
    }

    for league in ("nba", "nfl", "ncaaf"):
        panel = L.PANELS[league]().dropna(subset=["dominance"])
        feats = [f for f in FEATURES if panel[f].notna().sum() > len(panel) * 0.8]
        panel = panel.dropna(subset=feats).sort_values("start_year")
        X, y = panel[feats].to_numpy(), panel["dominance"].to_numpy()
        splitter = TimeSeriesSplit(n_splits=5)

        for label, make in models.items():
            preds, actual = [], []
            for tr, te in splitter.split(X):
                m = make().fit(X[tr], y[tr])
                preds.append(m.predict(X[te]))
                actual.append(y[te])
            preds, actual = np.concatenate(preds), np.concatenate(actual)
            rows.append({
                "league": league, "model": label, "n_seasons": len(panel),
                "cv_r2": r2_score(actual, preds),
                "cv_mae": float(np.mean(np.abs(actual - preds))),
            })
    return pd.DataFrame(rows)


def lasso_coefficients() -> pd.DataFrame:
    """Which metrics does Lasso keep when fit on the whole record?"""
    rows = []
    for league in ("nba", "nfl", "ncaaf"):
        panel = L.PANELS[league]().dropna(subset=["dominance"])
        feats = [f for f in FEATURES if panel[f].notna().sum() > len(panel) * 0.8]
        panel = panel.dropna(subset=feats)
        pipe = make_pipeline(StandardScaler(), LassoCV(cv=5, max_iter=50000))
        pipe.fit(panel[feats], panel["dominance"])
        coefs = pipe[-1].coef_
        for f, c in zip(feats, coefs):
            rows.append({"league": league, "feature": f,
                         "standardised_coef": float(c), "kept": bool(abs(c) > 1e-8)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- experiment 2
def discover_eras(league: str = "nba", k: int = 3):
    """PCA + k-means on balance metrics: where does the data split history?"""
    panel = L.PANELS[league]().dropna(subset=["dominance"])
    feats = [f for f in FEATURES if panel[f].notna().sum() > len(panel) * 0.8]
    panel = panel.dropna(subset=feats).sort_values("start_year").reset_index(drop=True)
    X = StandardScaler().fit_transform(panel[feats])
    pca = PCA(n_components=2).fit(X)
    comps = pca.transform(X)
    labels = KMeans(n_clusters=k, n_init=25, random_state=0).fit_predict(X)
    panel["cluster"] = labels
    panel["pc1"], panel["pc2"] = comps[:, 0], comps[:, 1]
    return panel, pca, feats


def change_points(series: np.ndarray, min_size: int = 6) -> list[int]:
    """Binary segmentation on the mean: the split that most reduces SSE."""
    def best_split(a: np.ndarray) -> tuple[int | None, float]:
        n = len(a)
        if n < 2 * min_size:
            return None, 0.0
        total = float(((a - a.mean()) ** 2).sum())
        best, gain = None, 0.0
        for i in range(min_size, n - min_size):
            sse = float(((a[:i] - a[:i].mean()) ** 2).sum()
                        + ((a[i:] - a[i:].mean()) ** 2).sum())
            if total - sse > gain:
                best, gain = i, total - sse
        return best, gain

    idx, _ = best_split(series)
    if idx is None:
        return []
    return sorted([idx] + [idx + j for j in change_points(series[idx:], min_size)]
                  + change_points(series[:idx], min_size))


# ---------------------------------------------------------------- experiment 3
def team_persistence(league: str = "nba") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict whether a team is elite next season, splitting strictly by time."""
    loaders = {"nba": D.load_nba, "nfl": D.load_nfl, "ncaaf": D.load_ncaaf}
    df = loaders[league]().copy()
    df["start_year"] = df["season"].map(D.season_key)

    if league == "nba":
        df["elite"] = df["rank"] <= 4
        rating = "mov"
    elif league == "nfl":
        df["elite"] = df["wins"] >= 12
        rating = "mov"
    else:
        df["elite"] = df["ap_rank"] <= 10
        rating = "srs"

    df = df.sort_values(["team", "start_year"])
    df["elite_next"] = df.groupby("team")["elite"].shift(-1)
    df["next_year"] = df.groupby("team")["start_year"].shift(-1)
    df = df[df["next_year"] == df["start_year"] + 1]          # consecutive only

    # History features: how entrenched was this team before this season?
    df["prior_elite"] = df.groupby("team")["elite"].transform(
        lambda s: s.shift(1).expanding().sum()).fillna(0)
    df["prior_seasons"] = df.groupby("team").cumcount()
    df["elite_rate"] = df["prior_elite"] / df["prior_seasons"].clip(lower=1)

    feats = ["win_pct", rating, "elite", "prior_elite", "prior_seasons", "elite_rate"]
    data = df.dropna(subset=feats + ["elite_next"]).copy()
    data["elite"] = data["elite"].astype(int)
    y = data["elite_next"].astype(int).to_numpy()
    X = data[feats].to_numpy(dtype=float)
    years = data["start_year"].to_numpy()

    cutoff = int(np.quantile(years, 0.7))
    tr, te = years <= cutoff, years > cutoff

    models = {
        "Baseline (base rate)": DummyClassifier(strategy="prior"),
        "Logistic regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=5000)),
        "Gradient boosting": GradientBoostingClassifier(random_state=0),
    }
    rows, importances = [], pd.DataFrame()
    for label, model in models.items():
        model.fit(X[tr], y[tr])
        prob = (model.predict_proba(X[te])[:, 1]
                if hasattr(model, "predict_proba") else model.predict(X[te]))
        auc = roc_auc_score(y[te], prob) if len(np.unique(y[te])) > 1 else np.nan
        rows.append({"league": league, "model": label,
                     "n_train": int(tr.sum()), "n_test": int(te.sum()),
                     "test_auc": auc, "base_rate": float(y[te].mean()),
                     "train_cutoff_year": cutoff})
        if label == "Gradient boosting":
            imp = permutation_importance(model, X[te], y[te], n_repeats=20,
                                         random_state=0, scoring="roc_auc")
            importances = pd.DataFrame({
                "league": league, "feature": feats,
                "importance": imp.importances_mean,
                "std": imp.importances_std,
            }).sort_values("importance", ascending=False)
    return pd.DataFrame(rows), importances


def main() -> None:
    P.apply_style()
    REPORTS.mkdir(parents=True, exist_ok=True)

    season_cv = season_level_models()
    coefs = lasso_coefficients()
    season_cv.to_csv(REPORTS / "ml_season_cv.csv", index=False)
    coefs.to_csv(REPORTS / "ml_lasso_coefficients.csv", index=False)
    print("1. Season-level forward-chaining CV (negative R2 = worse than the mean)")
    print(season_cv.pivot(index="league", columns="model", values="cv_r2")
          .to_string(float_format=lambda v: f"{v:.3f}"))

    persist, imps = [], []
    for lg in ("nba", "nfl", "ncaaf"):
        r, i = team_persistence(lg)
        persist.append(r); imps.append(i)
    persist = pd.concat(persist, ignore_index=True)
    imps = pd.concat(imps, ignore_index=True)
    persist.to_csv(REPORTS / "ml_team_persistence.csv", index=False)
    imps.to_csv(REPORTS / "ml_feature_importance.csv", index=False)
    print("\n3. Team-level 'elite next season?' (held-out AUC)")
    print(persist.pivot(index="league", columns="model", values="test_auc")
          .to_string(float_format=lambda v: f"{v:.3f}"))

    panel, pca, feats = discover_eras("nba")
    cps = change_points(panel["dominance"].to_numpy())
    era_years = [int(panel["start_year"].iloc[i]) for i in cps]
    print(f"\n2. NBA change points in dominance: {era_years}")
    print(f"   PCA explained variance: {pca.explained_variance_ratio_[:2].round(3)}")
    panel.to_csv(REPORTS / "ml_era_clusters.csv", index=False)

    # ------------------------------- figure -------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.5))

    ax = axes[0, 0]
    piv = season_cv.pivot(index="league", columns="model", values="cv_r2")
    piv = piv[["Baseline (train mean)", "Lasso", "ElasticNet", "Random Forest"]]
    piv.plot.bar(ax=ax, width=0.78, color=["#9aa8b5", P.ACCENT, "#6fb3a8", P.INK])
    ax.axhline(0, color=P.INK, lw=1)
    ax.set_ylabel("out-of-sample $R^2$"); ax.set_xlabel("")
    ax.set_title("1. Season-level models do not beat\npredicting the mean")
    ax.legend(fontsize=7, ncol=2)
    ax.tick_params(axis="x", rotation=0)

    ax = axes[0, 1]
    colors = ["#2f6f8f", "#e0a33a", "#b5495b"]
    for cl, g in panel.groupby("cluster"):
        ax.scatter(g["start_year"], g["dominance"], s=42,
                   color=colors[cl % len(colors)], label=f"regime {cl+1}")
    for yr in era_years:
        ax.axvline(yr, color=P.TRend, ls="--", lw=1.4)
        ax.text(yr, ax.get_ylim()[1], f" {yr}", color=P.TRend, fontsize=8, va="top")
    ax.set_xlabel("Season"); ax.set_ylabel("Dominance index")
    ax.set_title("2. Data-chosen regimes and change points\n(NBA)")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    piv2 = persist.pivot(index="league", columns="model", values="test_auc")
    piv2 = piv2[["Baseline (base rate)", "Logistic regression", "Gradient boosting"]]
    piv2.plot.bar(ax=ax, width=0.78, color=["#9aa8b5", P.ACCENT, P.INK])
    ax.axhline(0.5, color=P.TRend, ls=":", label="chance")
    ax.set_ylim(0.4, 1.0); ax.set_ylabel("held-out ROC AUC"); ax.set_xlabel("")
    ax.set_title("3. Team-level persistence is highly\npredictable")
    ax.legend(fontsize=7); ax.tick_params(axis="x", rotation=0)

    ax = axes[1, 1]
    nba_imp = imps[imps["league"] == "nba"].sort_values("importance")
    ax.barh(nba_imp["feature"], nba_imp["importance"],
            xerr=nba_imp["std"], color=P.INK)
    ax.set_xlabel("drop in AUC when shuffled")
    ax.set_title("4. What predicts staying elite (NBA)\npermutation importance")

    fig.suptitle("Machine learning on durable dominance", fontweight="bold", y=1.02)
    print("\nwrote", P.save(fig, "09_machine_learning.png").name)
    print("\nNBA permutation importance\n" +
          imps[imps["league"] == "nba"].to_string(index=False,
                                                  float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main()
