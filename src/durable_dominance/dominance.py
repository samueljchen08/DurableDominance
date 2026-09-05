"""Dominance and competitiveness measures built around league champions.

The organising question: when a league crowns a champion, how *established*
was that champion already, and does that depend on how competitive the league
was that year?

Dominance is measured from a champion's history at the moment it won, so every
measure is strictly backward-looking and no season uses information from its
own future. Competitiveness is measured from the same season's standings and
head-to-head results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from durable_dominance import metrics as M


# --------------------------------------------------------------------------
# Champion pedigree: how established was the winner?
# --------------------------------------------------------------------------

def champion_pedigree(
    standings: pd.DataFrame, champions: pd.Series, *,
    season_col: str = "season", team_col: str = "team",
    rank_col: str = "rank", order=None,
) -> pd.DataFrame:
    """One row per season describing the eventual champion's prior record.

    Columns
    -------
    prior_titles       championships won before this season
    prior_titles_10y   championships in the ten preceding seasons
    prior_top4         prior seasons finishing in the league's top four
    prior_top8         prior seasons finishing in the league's top eight
    career_win_pct     cumulative winning percentage before this season
    years_since_title  seasons since the last title (NaN if never)
    seasons_of_history seasons the franchise has existed in the data
    """
    order = order or (lambda s: s)
    seasons = sorted(standings[season_col].dropna().unique(), key=order)

    titles: dict[str, list[int]] = {}
    top4: dict[str, int] = {}
    top8: dict[str, int] = {}
    wins: dict[str, float] = {}
    games: dict[str, float] = {}
    appearances: dict[str, int] = {}

    rows = []
    for i, season in enumerate(seasons):
        g = standings[standings[season_col] == season]
        champ = champions.get(season)
        if champ is not None and champ in set(g[team_col]):
            hist = titles.get(champ, [])
            rows.append({
                "season": season,
                "champion": champ,
                "prior_titles": len(hist),
                "prior_titles_10y": sum(1 for t in hist if i - t <= 10),
                "prior_top4": top4.get(champ, 0),
                "prior_top8": top8.get(champ, 0),
                "career_win_pct": (wins.get(champ, 0) / games[champ]
                                   if games.get(champ) else np.nan),
                "years_since_title": (i - hist[-1]) if hist else np.nan,
                "seasons_of_history": appearances.get(champ, 0),
            })

        # advance history *after* recording, so nothing leaks from this season
        ranked = g.sort_values(rank_col)
        for pos, r in enumerate(ranked.itertuples(), start=1):
            t = getattr(r, team_col)
            appearances[t] = appearances.get(t, 0) + 1
            wins[t] = wins.get(t, 0) + r.wins
            games[t] = games.get(t, 0) + r.wins + r.losses
            if pos <= 4:
                top4[t] = top4.get(t, 0) + 1
            if pos <= 8:
                top8[t] = top8.get(t, 0) + 1
        if champ is not None:
            titles.setdefault(champ, []).append(i)

    out = pd.DataFrame(rows)
    # A single index combining the four count-based measures on a common scale.
    if len(out):
        z = out[["prior_titles", "prior_titles_10y", "prior_top4", "prior_top8"]]
        out["pedigree_index"] = ((z - z.mean()) / z.std(ddof=0)).mean(axis=1)
    return out


# --------------------------------------------------------------------------
# League-level dominance: how concentrated has success been lately?
# --------------------------------------------------------------------------

def rolling_title_concentration(
    champions: pd.Series, window: int = 10
) -> pd.DataFrame:
    """Concentration of titles in each trailing window of seasons."""
    seasons = list(champions.index)
    rows = []
    for i, season in enumerate(seasons):
        past = champions.iloc[max(0, i - window):i]
        if len(past) < 3:
            rows.append({"season": season, "title_hhi": np.nan,
                         "unique_champions": np.nan, "title_entropy": np.nan})
            continue
        share = past.value_counts(normalize=True)
        rows.append({
            "season": season,
            "title_hhi": float((share**2).sum()),
            "unique_champions": int(share.size),
            "title_entropy": float(-(share * np.log(share)).sum()),
        })
    return pd.DataFrame(rows)


def elite_turnover(
    standings: pd.DataFrame, *, season_col="season", team_col="team",
    rank_col="rank", top_n=4, order=None,
) -> pd.DataFrame:
    """How much of the league's elite is new each season."""
    order = order or (lambda s: s)
    seasons = sorted(standings[season_col].dropna().unique(), key=order)
    rows, prev = [], None
    for season in seasons:
        g = standings[standings[season_col] == season]
        elite = set(g.sort_values(rank_col).head(top_n)[team_col])
        rows.append({
            "season": season,
            "elite_holdover": (len(elite & prev) / top_n) if prev else np.nan,
        })
        prev = elite
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Competitiveness of a single season
# --------------------------------------------------------------------------

def season_competitiveness(
    standings: pd.DataFrame, strengths: dict | None = None,
    matches: pd.DataFrame | None = None, *, season_col="season",
) -> pd.DataFrame:
    """A battery of within-season balance measures, one row per season.

    `win_pct_sd` and `noll_scully` describe the record distribution;
    `mov_sd` and `strength_sd` describe underlying quality, which is less
    noisy; `top_bottom_gap` and `gini_wins` describe the shape of the tail;
    `upset_rate` counts how often the weaker side actually won.
    """
    rows = []
    for season, g in standings.groupby(season_col, sort=False):
        games = int(round((g["wins"] + g["losses"]).median()))
        w = g["win_pct"].dropna()
        row = {
            "season": season,
            "n_teams": len(g),
            "win_pct_sd": M.win_pct_sd(w),
            "noll_scully": M.noll_scully(w, games) if games else np.nan,
            "mov_sd": float(g["mov"].std(ddof=1)) if "mov" in g else np.nan,
            "top_bottom_gap": float(w.max() - w.min()),
            "hhi_wins": M.hhi(g["wins"]),
            "gini_wins": gini(g["wins"].to_numpy()),
        }
        if strengths and season in strengths:
            row["strength_sd"] = float(strengths[season]["bt"].std(ddof=1))
        if matches is not None and strengths and season in strengths:
            row["upset_rate"] = upset_rate_from_matches(
                matches[matches[season_col] == season], strengths[season]["bt"])
        rows.append(row)
    return pd.DataFrame(rows)


def gini(values: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector."""
    x = np.sort(np.asarray(values, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return np.nan
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def upset_rate_from_matches(season_matches: pd.DataFrame, beta: pd.Series) -> float:
    """Share of games won by the team with the lower fitted strength."""
    wins = lower = 0.0
    for r in season_matches.itertuples():
        if r.team not in beta.index or r.opponent not in beta.index:
            continue
        if beta[r.team] < beta[r.opponent]:
            lower += r.series_wins + r.series_losses
            wins += r.series_wins
    return float(wins / lower) if lower else np.nan
