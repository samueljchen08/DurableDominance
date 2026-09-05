"""Random-walk playoff simulation.

A postseason is a short random walk: each series is seven Bernoulli draws
whose success probability is set by the two teams' strength gap plus home
court. Feeding real strength estimates through this process gives the
championship distribution a league *should* produce if nothing carried over
between seasons except measured quality. Comparing that to the real title
record is the test for durable dominance.

Two modelling notes:

* A best-of-7 is simulated as the majority of all seven games rather than
  first-to-four. The two are distributionally identical when per-game
  probabilities are fixed, and simulating all seven vectorises cleanly.
* Every season uses the same 16-team, 2-2-1-1-1 bracket. The real NBA seeded
  12 teams before 1984 and reseeded by division for much of the 1980s. Holding
  the format constant is deliberate: it means an era-to-era difference in the
  results reflects the teams, not a rule change.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Higher seed hosts games 1, 2, 5 and 7 under the 2-2-1-1-1 format.
HOME_PATTERN = np.array([1, 1, -1, -1, 1, -1, 1], dtype=float)
# NBA home teams win about 60% of games, which is ~0.4 in log-odds.
DEFAULT_HCA = 0.40
# 1v8 / 4v5 in the top half, 2v7 / 3v6 in the bottom, so seeds 1 and 2 can
# only meet in the conference final.
BRACKET = [(0, 7), (3, 4), (1, 6), (2, 5)]


def simulate_series(
    beta_high: np.ndarray, beta_low: np.ndarray, n_sims: int,
    rng: np.random.Generator, hca: float = DEFAULT_HCA,
) -> np.ndarray:
    """True where the higher seed wins. Inputs broadcast over `n_sims`."""
    gap = (beta_high - beta_low)[..., None] + hca * HOME_PATTERN
    p = 1.0 / (1.0 + np.exp(-gap))
    wins = (rng.random(p.shape) < p).sum(axis=-1)
    return wins >= 4


def simulate_conference(
    seeds: np.ndarray, n_sims: int, rng: np.random.Generator, hca: float
) -> np.ndarray:
    """Run one conference's three rounds. `seeds` is 8 strengths, best first.

    Returns the winning *seed index* for each simulation.
    """
    alive = np.tile(np.arange(8), (n_sims, 1))          # (n_sims, 8)
    order = np.array([s for pair in BRACKET for s in pair])
    alive = alive[:, order]                              # bracket order

    while alive.shape[1] > 1:
        high, low = alive[:, 0::2], alive[:, 1::2]
        # The lower seed number is the higher seed, so it gets home court.
        top = np.minimum(high, low)
        bot = np.maximum(high, low)
        won = simulate_series(seeds[top], seeds[bot], n_sims, rng, hca)
        alive = np.where(won, top, bot)
    return alive[:, 0]


def simulate_season(
    strengths: pd.DataFrame,
    strength_col: str = "bt",
    n_sims: int = 10_000,
    hca: float = DEFAULT_HCA,
    rng: np.random.Generator | None = None,
) -> pd.Series:
    """Championship probability for every team in one season.

    `strengths` needs `conference` and the chosen strength column. The top
    eight by record in each conference make the field; everyone else gets 0.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    probs = pd.Series(0.0, index=strengths.index, name="title_prob")

    finalists, finalist_beta = [], []
    for conf in ("E", "W"):
        pool = strengths[strengths["conference"] == conf]
        pool = pool.sort_values("win_pct", ascending=False).head(8)
        if len(pool) < 8:
            return probs                                  # season not usable
        names = pool.index.to_numpy()
        beta = pool[strength_col].to_numpy()
        winner_seed = simulate_conference(beta, n_sims, rng, hca)
        finalists.append(names[winner_seed])
        finalist_beta.append(beta[winner_seed])

    east, west = finalists
    be, bw = finalist_beta
    # Home court in the Finals goes to the better regular-season record.
    rank = strengths["win_pct"]
    east_hosts = rank.reindex(east).to_numpy() >= rank.reindex(west).to_numpy()
    high = np.where(east_hosts, be, bw)
    low = np.where(east_hosts, bw, be)
    high_wins = simulate_series(high, low, n_sims, rng, hca)
    champs = np.where(east_hosts == high_wins, east, west)

    counts = pd.Series(champs).value_counts() / n_sims
    probs.update(counts)
    return probs


def simulate_all_seasons(
    strengths_by_season: dict, strength_col: str = "bt",
    n_sims: int = 10_000, hca: float = DEFAULT_HCA, seed: int = 0,
) -> pd.DataFrame:
    """Title probabilities for every team-season. Long format."""
    rng = np.random.default_rng(seed)
    frames = []
    for season, st in strengths_by_season.items():
        p = simulate_season(st, strength_col, n_sims, hca, rng)
        if p.sum() == 0:
            continue
        frames.append(pd.DataFrame({"season": season, "team": p.index,
                                    "title_prob": p.to_numpy()}))
    return pd.concat(frames, ignore_index=True)


def sample_title_histories(
    probs: pd.DataFrame, n_histories: int = 4000, seed: int = 1
) -> np.ndarray:
    """Draw whole alternate leagues: one champion per season, many times.

    Each history is an independent draw of a champion for every season from
    that season's simulated distribution. Comparing the *concentration* of
    these histories against the real one is the durable-dominance test: real
    basketball being more concentrated than every simulated league means
    something persists across seasons that single-season strength misses.
    """
    rng = np.random.default_rng(seed)
    seasons = probs["season"].unique()
    teams = sorted(probs["team"].unique())
    idx = {t: i for i, t in enumerate(teams)}
    out = np.empty((n_histories, len(seasons)), dtype=np.int32)

    for k, season in enumerate(seasons):
        g = probs[probs["season"] == season]
        p = g["title_prob"].to_numpy()
        p = p / p.sum()
        picks = rng.choice(len(g), size=n_histories, p=p)
        out[:, k] = np.array([idx[t] for t in g["team"].to_numpy()])[picks]
    return out


def concentration(labels: np.ndarray, n_teams: int) -> dict:
    """Concentration of a set of champions: HHI, unique count, top share."""
    counts = np.bincount(labels, minlength=n_teams)
    n = counts.sum()
    shares = counts / n
    nz = shares[shares > 0]
    return {
        "unique_champions": int((counts > 0).sum()),
        "hhi": float((shares**2).sum()),
        "top_share": float(shares.max()),
        "entropy": float(-(nz * np.log(nz)).sum()),
    }


def repeat_rate(labels: np.ndarray) -> float:
    """Share of consecutive season pairs won by the same team."""
    return float((labels[..., 1:] == labels[..., :-1]).mean())
