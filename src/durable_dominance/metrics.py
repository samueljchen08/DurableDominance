"""Dominance and competitive-balance metrics.

Two families of functions live here:

* ``paper_*``  reproduce the exact arithmetic used in the original scripts and
  in the Sloan deck, quirks included, so published figures can be regenerated.
* everything else is the textbook version of the same idea.

Where the two disagree the difference is documented on the ``paper_*``
function. ``scripts/07_metric_robustness.py`` re-runs every headline
correlation under both and reports which conclusions survive.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Competitive balance
# --------------------------------------------------------------------------


def paper_win_pct_spread(win_pcts) -> float:
    """The deck's "Standard Deviation of Win%".

    Reproduces ``competitiveness_index_win_pct`` from the original NBA/NFL
    scripts: ``sqrt(sum(|w - 0.5|))``.

    This is *not* a standard deviation. It omits the division by n, so it
    grows with league size, and it sums absolute deviations rather than
    squares before taking the root. It is retained because the x-axes of the
    published NFL (1.9-2.4) and NBA (1.5-2.15) scatter plots are on this
    scale. Use `win_pct_sd` or `noll_scully` for an interpretable measure.
    """
    return math.sqrt(sum(abs(w - 0.5) for w in win_pcts))


def paper_rating_spread(ratings, n_total_rows: int, skip_top: int = 3) -> float:
    """The deck's "Standard Deviation of MOV / NRTG / SRS".

    Reproduces ``competitiveness_index_net_rtg``: drop the top ``skip_top``
    teams, sum ``|rating|``, divide by ``n_total_rows``, take the root.

    Two quirks are preserved deliberately. The divisor is the row count of the
    *entire* multi-season table rather than the number of teams in the season,
    so the result is scaled by an arbitrary constant; and dropping the top 3
    teams removes exactly the teams whose dominance is being measured. The
    constant divisor cancels in any correlation, but the truncation does not.
    """
    tail = list(ratings)[skip_top:]
    return math.sqrt(sum(abs(r) for r in tail) / n_total_rows)


def win_pct_sd(win_pcts) -> float:
    """Sample standard deviation of winning percentage (the usual measure)."""
    return float(np.std(np.asarray(win_pcts, dtype=float), ddof=1))


def idealised_sd(n_games: int) -> float:
    """SD of win% for a league of evenly matched coin-flip teams."""
    return 0.5 / math.sqrt(n_games)


def noll_scully(win_pcts, n_games: int) -> float:
    """Noll-Scully ratio: actual SD of win% over the idealised SD.

    1.0 means perfect balance; higher means more spread than chance alone
    would produce. This is the standard cross-league comparable, because it
    removes the dependence on season length that `win_pct_sd` retains.
    """
    return win_pct_sd(win_pcts) / idealised_sd(n_games)


def hhi(shares) -> float:
    """Herfindahl-Hirschman index of a set of shares (e.g. season points).

    Shares are normalised internally, so raw point totals can be passed. The
    floor is 1/n (perfectly equal) and the ceiling is 1 (one team takes all).
    """
    arr = np.asarray(shares, dtype=float)
    total = arr.sum()
    if total <= 0:
        return float("nan")
    p = arr / total
    return float(np.sum(p**2))


def relative_sd(values) -> float:
    """Coefficient of variation, reported as "RSD" in the deck."""
    arr = np.asarray(values, dtype=float)
    mean = arr.mean()
    if mean == 0:
        return float("nan")
    return float(np.std(arr, ddof=1) / mean)


def closeness(win_pcts) -> float:
    """Mean absolute distance from .500 - the deck's "Closeness of Win Pct"."""
    return float(np.mean(np.abs(np.asarray(win_pcts, dtype=float) - 0.5)))


def upset_rate(favourite_won) -> float:
    """Share of matches won by the weaker side."""
    arr = np.asarray(favourite_won, dtype=bool)
    return float(1.0 - arr.mean()) if arr.size else float("nan")


# --------------------------------------------------------------------------
# Dominance
# --------------------------------------------------------------------------


def historical_share(
    df: pd.DataFrame,
    season,
    *,
    season_col: str,
    team_col: str,
    elite_mask: pd.Series,
    order,
) -> pd.Series:
    """Percentage of all prior elite slots taken by each team.

    ``elite_mask`` marks the rows that count as an elite finish (top 4, 12+
    wins, top-10 AP, ...). ``order`` maps a season label to a sortable key so
    "prior" is well defined for both ``2001`` and ``"2000-2001"``.
    """
    key = order(season)
    prior = df[df[season_col].map(order) < key]
    prior_elite = prior[elite_mask.reindex(prior.index, fill_value=False)]
    if prior_elite.empty:
        return pd.Series(dtype=float)

    slots = prior_elite.groupby(season_col)[team_col].nunique().sum()
    if slots == 0:
        return pd.Series(dtype=float)
    return prior_elite.groupby(team_col).size() / slots * 100.0


def dominance_index(
    df: pd.DataFrame,
    *,
    season_col: str,
    team_col: str,
    elite_mask: pd.Series,
    top_n_current,
    order,
) -> pd.Series:
    """Durable dominance: how established this season's best teams already were.

    For each season, take the teams that finished on top (``top_n_current``
    returns them) and sum the share of *previous* elite finishes they own. A
    high value means the current elite is the same elite as before.

    The first season is skipped: it has no history to draw on.
    """
    seasons = sorted(df[season_col].dropna().unique(), key=order)
    out: dict = {}
    for season in seasons[1:]:
        share = historical_share(
            df,
            season,
            season_col=season_col,
            team_col=team_col,
            elite_mask=elite_mask,
            order=order,
        )
        current = top_n_current(df[df[season_col] == season])
        out[season] = float(share.reindex(current, fill_value=0.0).sum())
    return pd.Series(out, name="dominance_index")


def dominance_count(
    df: pd.DataFrame,
    *,
    season_col: str,
    team_col: str,
    elite_mask: pd.Series,
    order,
) -> pd.Series:
    """Raw count version used for NCAA football.

    Sums how many times each of this season's elite teams had previously been
    elite. Unlike `dominance_index` this is not normalised by the number of
    slots awarded, so it trends upward simply because history accumulates.
    """
    seasons = sorted(df[season_col].dropna().unique(), key=order)
    counts: dict = {}
    out: dict = {}
    for season in seasons:
        current = df.loc[(df[season_col] == season) & elite_mask, team_col]
        out[season] = sum(counts.get(team, 0) for team in current)
        for team in current:
            counts[team] = counts.get(team, 0) + 1
    return pd.Series(out, name="dominance_count")
