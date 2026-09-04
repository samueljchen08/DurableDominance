"""Per-league season panels: one row per season, dominance beside balance.

Each builder returns a frame with

    season, start_year, n_teams, n_games,
    dominance            -- durable dominance (definition varies by league)
    paper_cb_win         -- the deck's win% spread (see metrics.paper_*)
    paper_cb_rating      -- the deck's MOV/NRTG/SRS spread
    win_pct_sd, noll_scully, hhi_wins, rsd_wins, closeness, rating_sd

so the published charts and the corrected ones can be drawn from one table.
"""

from __future__ import annotations

import pandas as pd

from durable_dominance import datasets as D
from durable_dominance import metrics as M


def _balance_row(group: pd.DataFrame, rating_col: str | None, n_total_rows: int) -> dict:
    win_pcts = group["win_pct"].dropna().tolist()
    games = int(round(group["games"].median())) if "games" in group else 0
    row = {
        "n_teams": len(group),
        "n_games": games,
        "paper_cb_win": M.paper_win_pct_spread(win_pcts),
        "win_pct_sd": M.win_pct_sd(win_pcts) if len(win_pcts) > 1 else float("nan"),
        "noll_scully": (
            M.noll_scully(win_pcts, games) if games > 0 and len(win_pcts) > 1
            else float("nan")
        ),
        "hhi_wins": M.hhi(group["wins"]),
        "rsd_wins": M.relative_sd(group["wins"]),
        "closeness": M.closeness(win_pcts),
    }
    if rating_col and rating_col in group:
        ratings = group[rating_col].dropna().tolist()
        row["paper_cb_rating"] = (
            M.paper_rating_spread(ratings, n_total_rows) if ratings else float("nan")
        )
        row["rating_sd"] = (
            float(pd.Series(ratings).std(ddof=1)) if len(ratings) > 1 else float("nan")
        )
    return row


def _panel(df, *, season_col, rating_col, dominance) -> pd.DataFrame:
    n_total_rows = len(df)
    rows = []
    for season, group in df.groupby(season_col, sort=False):
        row = {"season": season, "start_year": D.season_key(season)}
        row.update(_balance_row(group, rating_col, n_total_rows))
        rows.append(row)
    panel = pd.DataFrame(rows).sort_values("start_year")
    panel["dominance"] = panel["season"].map(dominance)
    return panel.reset_index(drop=True)


def nba_panel() -> pd.DataFrame:
    """NBA. Elite = a top-4 league finish; dominance = share of prior top-4 slots."""
    df = D.load_nba()
    dominance = M.dominance_index(
        df,
        season_col="season",
        team_col="team",
        elite_mask=df["rank"] <= 4,
        top_n_current=lambda g: g.nsmallest(4, "rank")["team"],
        order=D.season_key,
    )
    return _panel(df, season_col="season", rating_col="mov", dominance=dominance)


def nba_nrtg_panel() -> pd.DataFrame:
    """NBA using net rating as the balance signal (available from 1983-84)."""
    df = D.load_nba()
    dominance = M.dominance_index(
        df,
        season_col="season",
        team_col="team",
        elite_mask=df["rank"] <= 4,
        top_n_current=lambda g: g.nsmallest(4, "rank")["team"],
        order=D.season_key,
    )
    return _panel(df, season_col="season", rating_col="nrtg", dominance=dominance)


def nfl_panel() -> pd.DataFrame:
    """NFL. Elite = a 12-win season, the deck's stand-in for a top-4 finish.

    A fixed win threshold is not scale-free: 12 wins out of 14 (pre-1978) is a
    far rarer feat than 12 of 17 (2021 on), so the elite pool widens over time
    for reasons unrelated to dominance. `07_metric_robustness.py` re-runs this
    with a rank-based elite definition instead.
    """
    df = D.load_nfl()
    dominance = M.dominance_index(
        df,
        season_col="season",
        team_col="team",
        elite_mask=df["wins"] >= 12,
        top_n_current=lambda g: g.nlargest(4, "wins")["team"],
        order=D.season_key,
    )
    return _panel(df, season_col="season", rating_col="mov", dominance=dominance)


def ncaaf_panel() -> pd.DataFrame:
    """NCAA football. Elite = a top-10 AP finish; dominance is a raw count."""
    df = D.load_ncaaf()
    dominance = M.dominance_count(
        df,
        season_col="season",
        team_col="team",
        elite_mask=df["ap_rank"] <= 10,
        order=D.season_key,
    )
    return _panel(df, season_col="season", rating_col="srs", dominance=dominance)


PANELS = {
    "nba": nba_panel,
    "nba_nrtg": nba_nrtg_panel,
    "nfl": nfl_panel,
    "ncaaf": ncaaf_panel,
}
