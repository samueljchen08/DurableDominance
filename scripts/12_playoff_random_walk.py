"""Study B -- how much of championship dominance is a random walk?

A postseason is a short random walk. Each series is seven coin flips weighted
by the two teams' strength gap and home court; win four and you advance. Feed
each season's *real* measured strengths through that process and you get the
championship distribution a league should produce if nothing carried across
seasons except quality you can already see in the standings.

Three steps:

1. **Calibrate.** Scale every strength gap by a temperature `gamma` and pick
   the value that maximises the likelihood of the 43 champions that actually
   happened. `gamma = 1` means the raw Bradley-Terry gaps are already right.
2. **Check calibration properly.** A reliability curve: among teams the model
   gives a 20% title chance, do about 20% win?
3. **Test concentration.** Draw thousands of alternate 43-season leagues from
   the model and compare their concentration of titles to the real one.

Step 3 is the durable-dominance test. Because the simulation already uses each
season's real strengths, a dynasty's *quality* is fully accounted for. Excess
concentration beyond that means the same franchises convert quality into
titles at a rate the random walk does not explain.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from durable_dominance import datasets as D
from durable_dominance import playoffs as PO
from durable_dominance import plotting as P
from durable_dominance import strength as S

REPORTS = Path(__file__).resolve().parents[1] / "reports"
N_SIMS = 20_000
N_HISTORIES = 20_000
STATS = ["unique_champions", "hhi", "top_share", "entropy", "repeat_rate"]
STAT_LABEL = {
    "unique_champions": "unique champions", "hhi": "title HHI",
    "top_share": "top team's share", "entropy": "title entropy",
    "repeat_rate": "back-to-back rate",
}


def setup():
    n, m = D.load_nba(), D.load_nba_matches()
    scale = S.calibrate_mov_scale(n, m)
    strengths = {s: S.season_strengths(n, m, s, scale)
                 for s in n["season"].unique()}
    champions = m.groupby("season")["champion"].first()
    return n, m, strengths, champions, scale


def calibrate(strengths, champions, col="bt", gammas=None) -> pd.DataFrame:
    """Maximum-likelihood temperature on the observed championship record."""
    gammas = gammas if gammas is not None else np.linspace(0.3, 1.8, 16)
    rows = []
    for g in gammas:
        ll, ps = 0.0, []
        for s, st in strengths.items():
            scaled = st.copy()
            scaled[col] = scaled[col] * g
            pr = PO.simulate_season(scaled, col, 6000,
                                    rng=np.random.default_rng(abs(hash(s)) % 2**31))
            p = max(float(pr.get(champions[s], 0.0)), 1e-4)
            ll += np.log(p)
            ps.append(p)
        rows.append({"gamma": g, "loglik": ll, "mean_p_champion": np.mean(ps)})
    return pd.DataFrame(rows)


def reliability(probs, champions, bins=(0, .02, .05, .10, .20, .35, 1.01)):
    """Predicted title probability vs. how often those teams actually won."""
    df = probs.copy()
    df["won"] = [champions[s] == t for s, t in zip(df["season"], df["team"])]
    df = df[df["title_prob"] > 0]
    df["bin"] = pd.cut(df["title_prob"], bins=list(bins))
    g = df.groupby("bin", observed=True).agg(
        predicted=("title_prob", "mean"), actual=("won", "mean"),
        n=("won", "size"))
    return g.reset_index()


def concentration_test(probs, champions, n_histories=N_HISTORIES, seed=1):
    teams = sorted(probs["team"].unique())
    idx = {t: i for i, t in enumerate(teams)}
    seasons = list(probs["season"].unique())
    real = np.array([idx[champions[s]] for s in seasons])

    hist = PO.sample_title_histories(probs, n_histories, seed)
    real_stats = PO.concentration(real, len(teams))
    real_stats["repeat_rate"] = PO.repeat_rate(real)

    sim = pd.DataFrame([PO.concentration(h, len(teams)) for h in hist])
    sim["repeat_rate"] = [PO.repeat_rate(h) for h in hist]

    rows = []
    for k in STATS:
        v = real_stats[k]
        lo, med, hi = np.percentile(sim[k], [2.5, 50, 97.5])
        below = float((sim[k] <= v).mean())
        rows.append({
            "statistic": k, "real": v, "sim_median": med,
            "ci_lo": lo, "ci_hi": hi,
            "p_two_sided": 2 * min(below, 1 - below),
            "outside_95": bool(v < lo or v > hi),
        })
    return pd.DataFrame(rows), sim, real_stats


def main() -> None:
    P.apply_style()
    n, m, strengths, champions, scale = setup()
    print(f"MOV->strength scale: {scale:.4f} log-odds per point\n")

    # ---- 1. temperature ---------------------------------------------------
    cal = calibrate(strengths, champions)
    cal.to_csv(REPORTS / "playoff_calibration.csv", index=False)
    best = cal.loc[cal["loglik"].idxmax()]
    print("Temperature calibration (gamma scales every strength gap)")
    print(cal.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"\n  MLE gamma = {best['gamma']:.2f}  "
          f"(1.00 would mean the raw fitted gaps are already correct)")

    # ---- 2. main simulation + reliability ---------------------------------
    probs = PO.simulate_all_seasons(strengths, "bt", N_SIMS, seed=0)
    probs.to_csv(REPORTS / "title_probabilities.csv", index=False)
    rel = reliability(probs, champions)
    rel.to_csv(REPORTS / "playoff_reliability.csv", index=False)
    print("\nReliability of the simulated title probabilities")
    print(rel.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- 3. concentration test -------------------------------------------
    test, sim, real_stats = concentration_test(probs, champions)
    test.to_csv(REPORTS / "concentration_test.csv", index=False)
    print("\nReal vs. simulated concentration of 43 championships")
    print(test.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ---- 4. who beats their strength -------------------------------------
    exp = probs.groupby("team")["title_prob"].sum().rename("expected_titles")
    var = probs.groupby("team")["title_prob"].apply(
        lambda p: float((p * (1 - p)).sum())).rename("variance")
    act = champions.value_counts().rename("actual_titles")
    conv = pd.concat([exp, var, act], axis=1).fillna({"actual_titles": 0})
    conv["excess"] = conv["actual_titles"] - conv["expected_titles"]
    conv["z"] = conv["excess"] / np.sqrt(conv["variance"].clip(lower=1e-9))
    conv = conv.sort_values("excess", ascending=False)
    conv.to_csv(REPORTS / "title_conversion.csv")
    print("\nConverting strength into titles (expected from the random walk)")
    print(conv.head(6).to_string(float_format=lambda v: f"{v:.2f}"))
    print("  ...")
    print(conv.tail(4).to_string(float_format=lambda v: f"{v:.2f}"))
    print(f"\n  largest |z| = {conv['z'].abs().max():.2f} over "
          f"{len(conv)} franchises; with that many tests a |z| near 2.5-3 is "
          f"expected by chance, so read the aggregate test above, not this one.")

    # ---- 5. sensitivity ---------------------------------------------------
    sens = []
    for col in ("bt", "mov", "record"):
        for hca in (0.0, PO.DEFAULT_HCA):
            pr = PO.simulate_all_seasons(strengths, col, 8000, hca=hca, seed=3)
            t, _, rs = concentration_test(pr, champions, n_histories=4000, seed=4)
            row = {"strength": col, "hca": hca}
            for st in ("unique_champions", "hhi"):
                r = t[t["statistic"] == st].iloc[0]
                row[f"{st}_real"] = r["real"]
                row[f"{st}_sim"] = r["sim_median"]
                row[f"{st}_p"] = r["p_two_sided"]
            sens.append(row)
    sens = pd.DataFrame(sens)
    sens.to_csv(REPORTS / "concentration_sensitivity.csv", index=False)
    print("\nSensitivity to the strength estimator and home-court assumption")
    print(sens.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- figures ----------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    ax = axes[0, 0]
    ax.plot(cal["gamma"], cal["loglik"], "o-", color=P.INK, ms=4)
    ax.axvline(best["gamma"], color=P.TRend, ls="--",
               label=f"MLE γ = {best['gamma']:.2f}")
    ax.axvline(1.0, color=P.ACCENT, ls=":", label="γ = 1 (raw gaps)")
    ax.set_xlabel("temperature γ"); ax.set_ylabel("log-likelihood of real champions")
    ax.set_title("1. The raw strength gaps are\nalready calibrated")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    ax.plot([0, rel["predicted"].max() * 1.1], [0, rel["predicted"].max() * 1.1],
            color=P.ACCENT, ls=":", label="perfect calibration")
    ax.scatter(rel["predicted"], rel["actual"], s=rel["n"] / 3, color=P.INK,
               zorder=3, label="probability bin")
    ax.set_xlabel("predicted title probability")
    ax.set_ylabel("observed title rate")
    ax.set_title("2. Reliability: predictions match\nwhat happened")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.hist(sim["unique_champions"], bins=np.arange(12, 30) - .5, color=P.ACCENT,
            edgecolor="white", label="simulated leagues")
    ax.axvline(real_stats["unique_champions"], color=P.TRend, lw=2.4,
               label=f"real = {real_stats['unique_champions']}")
    ax.set_xlabel("unique champions in 43 seasons"); ax.set_ylabel("count")
    ax.set_title("3. Reality crowns far fewer\ndifferent champions")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.hist(sim["hhi"], bins=40, color=P.ACCENT, edgecolor="white")
    ax.axvline(real_stats["hhi"], color=P.TRend, lw=2.4,
               label=f"real = {real_stats['hhi']:.3f}")
    ax.set_xlabel("title HHI (concentration)"); ax.set_ylabel("count")
    ax.set_title("Title concentration")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.hist(sim["repeat_rate"], bins=30, color=P.ACCENT, edgecolor="white")
    ax.axvline(real_stats["repeat_rate"], color=P.TRend, lw=2.4,
               label=f"real = {real_stats['repeat_rate']:.3f}")
    ax.set_xlabel("share of back-to-back titles"); ax.set_ylabel("count")
    ax.set_title("Repeat champions")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    top = pd.concat([conv.head(6), conv.tail(5)])
    colors = [P.TRend if v > 0 else "#3d6f8e" for v in top["excess"]]
    ax.barh(range(len(top)), top["excess"], color=colors)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels([t.replace(" ", "\n", 0)[:22] for t in top.index], fontsize=8)
    ax.axvline(0, color=P.INK, lw=1)
    ax.invert_yaxis()
    ax.set_xlabel("actual − expected titles")
    ax.set_title("Who converts strength\ninto rings")

    fig.suptitle("Championships vs. a strength-driven random walk, NBA 1980-2023",
                 fontweight="bold", y=1.005)
    print("\nwrote", P.save(fig, "12_playoff_random_walk.png").name)


if __name__ == "__main__":
    main()
