"""Study C -- why is reality more concentrated than the random walk?

Study B leaves a puzzle. The simulation uses each season's *real* strengths,
so a dynasty's quality is already priced in, and the per-team-season
predictions are well calibrated. Yet across 43 seasons the real title record
is far more concentrated than the model produces.

Both facts can hold at once. Marginal calibration says each season's
probability is right on average. Concentration depends on something else:
whether the same franchise's outcomes are *correlated across seasons* once
strength is accounted for. This script tests exactly that.

1. **How persistent is quality?** Lag-1 autocorrelation of fitted strength,
   and the implied AR(1) half-life.
2. **Is over-performance a stable franchise trait?** Split the seasons in
   half and correlate each franchise's excess titles across the two halves.
   A real conversion skill shows up here; a run of luck does not.
3. **Is it one era?** The 1980s Lakers and Celtics could carry the whole
   result, so the test is repeated era by era.
4. **How big is the effect?** Add a per-franchise conversion bonus to the
   simulation and find the size that reproduces the observed concentration.
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
from durable_dominance import playoffs as PO
from durable_dominance import plotting as P
from durable_dominance import strength as S

REPORTS = Path(__file__).resolve().parents[1] / "reports"


def setup():
    n, m = D.load_nba(), D.load_nba_matches()
    scale = S.calibrate_mov_scale(n, m)
    strengths = {s: S.season_strengths(n, m, s, scale)
                 for s in n["season"].unique()}
    champions = m.groupby("season")["champion"].first()
    return n, m, strengths, champions


def strength_persistence(strengths, n) -> pd.DataFrame:
    """AR(1) fit to each franchise's strength path."""
    seasons = sorted(strengths, key=D.season_key)
    rows = []
    for team in sorted(set(n["team"])):
        path = [(s, strengths[s].loc[team, "bt"]) for s in seasons
                if team in strengths[s].index]
        if len(path) < 12:
            continue
        v = np.array([p[1] for p in path])
        rows.append({"team": team, "n_seasons": len(v),
                     "autocorr": float(pd.Series(v).autocorr(1)),
                     "mean_strength": float(v.mean()),
                     "sd_strength": float(v.std(ddof=1))})
    return pd.DataFrame(rows)


def excess_by_half(probs, champions, seasons):
    """Each franchise's actual-minus-expected titles, in each half of history."""
    mid = len(seasons) // 2
    halves = {"first": seasons[:mid], "second": seasons[mid:]}
    out = {}
    for name, ss in halves.items():
        sub = probs[probs["season"].isin(ss)]
        exp = sub.groupby("team")["title_prob"].sum()
        act = champions[champions.index.isin(ss)].value_counts()
        out[name] = (act.reindex(exp.index).fillna(0) - exp)
    return pd.DataFrame(out).dropna()


def era_test(probs, champions, seasons, edges):
    rows = []
    for lo, hi in edges:
        ss = [s for s in seasons if lo <= D.season_key(s) <= hi]
        sub = probs[probs["season"].isin(ss)]
        if len(ss) < 8:
            continue
        teams = sorted(sub["team"].unique())
        idx = {t: i for i, t in enumerate(teams)}
        real = np.array([idx[champions[s]] for s in ss])
        hist = PO.sample_title_histories(sub, 8000, seed=7)
        rs = PO.concentration(real, len(teams))
        sim = pd.DataFrame([PO.concentration(h, len(teams)) for h in hist])
        below = float((sim["hhi"] <= rs["hhi"]).mean())
        rows.append({
            "era": f"{lo}-{hi}", "n_seasons": len(ss),
            "unique_real": rs["unique_champions"],
            "unique_sim": float(sim["unique_champions"].median()),
            "hhi_real": rs["hhi"], "hhi_sim": float(sim["hhi"].median()),
            "p_two_sided": 2 * min(below, 1 - below),
        })
    return pd.DataFrame(rows)


def conversion_bonus_sweep(strengths, champions, bonuses, seed=11):
    """How big a persistent franchise bonus reproduces the real concentration?

    Each franchise gets a fixed strength bonus drawn once and applied to every
    season, standing in for whatever a dynasty has that the standings miss.
    The bonus is drawn from Normal(0, `b`), so `b` is the effect size in the
    same log-odds units as the strength estimates.
    """
    seasons = list(strengths)
    teams = sorted({t for st in strengths.values() for t in st.index})
    rng = np.random.default_rng(seed)
    rows = []
    for b in bonuses:
        uniq, hhis = [], []
        for _ in range(30):
            bonus = pd.Series(rng.normal(0, b, len(teams)), index=teams)
            adj = {}
            for s, st in strengths.items():
                c = st.copy()
                c["bt"] = c["bt"] + bonus.reindex(c.index).fillna(0.0)
                adj[s] = c
            pr = PO.simulate_all_seasons(adj, "bt", 1500, seed=int(rng.integers(1e6)))
            hist = PO.sample_title_histories(pr, 300, seed=int(rng.integers(1e6)))
            idx = {t: i for i, t in enumerate(sorted(pr["team"].unique()))}
            for h in hist[:150]:
                c2 = PO.concentration(h, len(idx))
                uniq.append(c2["unique_champions"])
                hhis.append(c2["hhi"])
        rows.append({"bonus_sd": b, "unique_champions": float(np.mean(uniq)),
                     "hhi": float(np.mean(hhis))})
        print(f"  bonus sd={b:.2f} -> unique {np.mean(uniq):5.1f}, "
              f"hhi {np.mean(hhis):.4f}")
    return pd.DataFrame(rows)


def main() -> None:
    P.apply_style()
    n, m, strengths, champions = setup()
    seasons = sorted(strengths, key=D.season_key)
    probs = pd.read_csv(REPORTS / "title_probabilities.csv")

    # ---- 1. persistence ---------------------------------------------------
    pers = strength_persistence(strengths, n)
    pers.to_csv(REPORTS / "strength_persistence.csv", index=False)
    rho = pers["autocorr"].mean()
    print(f"1. Strength persistence across {len(pers)} franchises")
    print(f"   mean lag-1 autocorrelation rho = {rho:.3f}")
    print(f"   AR(1) half-life = {np.log(0.5)/np.log(rho):.1f} seasons")
    print(f"   franchise strength SD within: {pers['sd_strength'].mean():.3f}, "
          f"between: {pers['mean_strength'].std(ddof=1):.3f}")

    # ---- 2. is over-performance a stable trait? ---------------------------
    halves = excess_by_half(probs, champions, seasons)
    r = stats.pearsonr(halves["first"], halves["second"])
    print(f"\n2. Split-half reliability of title over-performance")
    print(f"   r = {r.statistic:+.3f}, p = {r.pvalue:.3f}, n = {len(halves)} franchises")
    print("   A franchise that beat its strength in 1980-2001 is "
          + ("MORE" if r.statistic > 0 else "NOT more")
          + " likely to have done so in 2002-2023.")
    halves.to_csv(REPORTS / "overperformance_split_half.csv")

    # ---- 3. era by era ----------------------------------------------------
    eras = era_test(probs, champions, seasons,
                    [(1980, 1994), (1995, 2009), (2010, 2023), (1980, 2023)])
    eras.to_csv(REPORTS / "concentration_by_era.csv", index=False)
    print("\n3. Concentration test, era by era")
    print(eras.to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    # ---- 4. effect size ---------------------------------------------------
    print("\n4. How large a persistent franchise bonus reproduces reality?")
    sweep = conversion_bonus_sweep(strengths, champions,
                                   [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    sweep.to_csv(REPORTS / "conversion_bonus_sweep.csv", index=False)
    real_uniq, real_hhi = 14, 0.1195
    need = np.interp(real_hhi, sweep["hhi"], sweep["bonus_sd"])
    print(f"   real HHI {real_hhi:.4f} is matched at bonus sd ~ {need:.2f} "
          f"log-odds\n   (for scale, the within-season SD of strength itself is "
          f"{np.mean([st['bt'].std() for st in strengths.values()]):.2f})")

    # ---- figures ----------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    ax = axes[0, 0]
    ax.hist(pers["autocorr"], bins=14, color=P.ACCENT, edgecolor="white")
    ax.axvline(rho, color=P.TRend, lw=2.2, label=f"mean ρ = {rho:.2f}")
    ax.set_xlabel("lag-1 autocorrelation of team strength")
    ax.set_ylabel("franchises")
    ax.set_title("Quality is persistent — but this is\nalready inside the simulation")
    ax.legend(fontsize=8)

    P.regression_scatter(
        axes[0, 1], halves["first"], halves["second"],
        xlabel="excess titles, 1980-2001", ylabel="excess titles, 2002-2023",
        title="Is beating your strength a stable trait?")

    ax = axes[1, 0]
    e = eras[eras["era"] != "1980-2023"]
    x = np.arange(len(e))
    ax.bar(x - .2, e["unique_real"], .4, label="real", color=P.TRend)
    ax.bar(x + .2, e["unique_sim"], .4, label="simulated median", color=P.INK)
    ax.set_xticks(x); ax.set_xticklabels(e["era"])
    ax.set_ylabel("unique champions"); ax.legend(fontsize=8)
    ax.set_title("Every era is more concentrated\nthan the random walk")
    for i, row in e.reset_index().iterrows():
        ax.text(i, max(row["unique_real"], row["unique_sim"]) + .3,
                f"p={row['p_two_sided']:.3f}", ha="center", fontsize=8,
                color=P.ink if hasattr(P, "ink") else P.INK)

    ax = axes[1, 1]
    ax.plot(sweep["bonus_sd"], sweep["hhi"], "o-", color=P.INK, ms=5,
            label="simulated")
    ax.axhline(real_hhi, color=P.TRend, ls="--", label=f"real HHI = {real_hhi:.3f}")
    ax.axvline(need, color=P.ACCENT, ls=":", label=f"needs σ ≈ {need:.2f}")
    ax.set_xlabel("SD of persistent franchise bonus (log-odds)")
    ax.set_ylabel("title HHI")
    ax.set_title("Effect size needed to explain\nthe gap")
    ax.legend(fontsize=8)

    fig.suptitle("Where durable dominance comes from", fontweight="bold", y=1.005)
    print("\nwrote", P.save(fig, "13_dynasty_dynamics.png").name)


if __name__ == "__main__":
    main()
