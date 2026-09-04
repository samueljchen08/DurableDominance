"""Loaders that turn the raw spreadsheets in this repo into tidy frames.

Every loader returns a DataFrame with normalised column names and a `season`
column that sorts chronologically via `season_key`. Raw files are never
modified; cleaned copies are written to `data/processed/` by
`scripts/01_build_clean_datasets.py`.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

PATHS = {
    "nba_rankings": ROOT / "Individual Sports" / "Combined_Team_Rankings.xlsx",
    "nba_matches": ROOT / "NBA Coding" / "Combined_Team_Matches.xlsx",
    "nfl": ROOT / "NBA Coding" / "NFL.xlsx",
    "ncaaf": ROOT / "NBA Coding" / "NCAAF.xlsx",
    "masters": ROOT / "Individual Sports" / "Golf_Masters.xlsx",
    "tour_de_france": ROOT / "Individual Sports" / "Tour De France.xlsx",
    "grand_slams": ROOT / "Individual Sports" / "Mens_Tennis_Grand_Slam_Winner.csv",
    "cricket": ROOT / "Harvested Data" / "cricket.csv",
    "womens_world_cup": ROOT / "Harvested Data" / "FIFA_Women's_Team_Matches.csv",
    "kentucky_derby": ROOT
    / "Harvested Data"
    / "Kentucky Derby Winners 1875 - current - Sheet1.csv",
}


def season_key(season) -> int:
    """Sortable start year for either ``2001`` or ``"2000-2001"``."""
    if isinstance(season, (int, float)):
        return int(season)
    return int(str(season).split("-")[0])


# --------------------------------------------------------------------------
# Team leagues
# --------------------------------------------------------------------------


def load_nba() -> pd.DataFrame:
    """NBA end-of-season standings, 1980-81 to 2022-23.

    Net rating is only populated from 1983-84 onward; margin of victory covers
    every season.
    """
    df = pd.read_excel(PATHS["nba_rankings"])
    df = df.rename(
        columns={
            "Unnamed: 0_level_0_Rk": "rank",
            "Unnamed: 1_level_0_Team": "team",
            "Unnamed: 2_level_0_Conf": "conference",
            "Unnamed: 3_level_0_Div": "division",
            "Unnamed: 4_level_0_W": "wins",
            "Unnamed: 5_level_0_L": "losses",
            "Unnamed: 6_level_0_W/L%": "win_pct",
            "Adjusted_MOV/A": "mov",
            "Adjusted_NRtg/A": "nrtg",
            "Year_": "season",
        }
    )
    keep = [
        "rank", "team", "conference", "division", "wins", "losses",
        "win_pct", "mov", "nrtg", "season",
    ]
    df = df[keep].dropna(subset=["season", "team"])
    df = df[df["wins"] > 0].copy()
    df["games"] = df["wins"] + df["losses"]
    df["start_year"] = df["season"].map(season_key)
    return df.sort_values(["start_year", "rank"]).reset_index(drop=True)


def load_nfl() -> pd.DataFrame:
    """NFL standings, 1970-2023. Ties are counted as half a win in `win_pct`."""
    df = pd.read_excel(PATHS["nfl"]).rename(
        columns={
            "NFL Team": "team",
            "W": "wins",
            "L": "losses",
            "T": "ties",
            "PCT": "win_pct",
            "PF": "points_for",
            "PA": "points_against",
            "Net Pts": "net_points",
            "Year": "season",
        }
    )
    df = df.dropna(subset=["season", "team"]).copy()
    df["season"] = df["season"].astype(int)
    df["games"] = df["wins"] + df["losses"] + df["ties"].fillna(0)
    # Net points is a season total; per-game margin is the comparable quantity.
    df["mov"] = df["net_points"] / df["games"]
    df["start_year"] = df["season"]
    return df.sort_values(["season", "wins"], ascending=[True, False]).reset_index(
        drop=True
    )


def load_ncaaf() -> pd.DataFrame:
    """NCAA football team-seasons, 1970-2023, with AP poll finish and SRS."""
    df = pd.read_excel(PATHS["ncaaf"], header=2).rename(
        columns={
            "Rk": "rank",
            "School": "team",
            "Conf": "conference",
            "W": "wins",
            "L": "losses",
            "Pct": "win_pct",
            "SRS": "srs",
            "SOS": "sos",
            "AP Rank": "ap_rank",
            "AP Pre": "ap_pre",
            "AP High": "ap_high",
            "Year": "season",
        }
    )
    df = df.dropna(subset=["season", "team"]).copy()
    df["season"] = df["season"].astype(int)
    for col in ("wins", "losses", "win_pct", "srs", "ap_rank"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["games"] = df["wins"] + df["losses"]
    df["start_year"] = df["season"]
    return df.sort_values(["season", "rank"]).reset_index(drop=True)


def load_nba_matches() -> pd.DataFrame:
    """NBA head-to-head season series, 1980-81 to 2022-23.

    `Result` holds a season series record such as ``"4-2"``, not a single game.
    """
    df = pd.read_excel(PATHS["nba_matches"]).rename(
        columns={
            "Rk": "rank",
            "Team": "team",
            "Season": "season",
            "Opponent": "opponent",
            "Result": "result",
            "Tournament Winner": "champion",
        }
    )
    df = df.dropna(subset=["result", "season"]).copy()
    parts = df["result"].astype(str).str.extract(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
    df["series_wins"] = pd.to_numeric(parts[0], errors="coerce")
    df["series_losses"] = pd.to_numeric(parts[1], errors="coerce")
    df = df.dropna(subset=["series_wins", "series_losses"])
    df["start_year"] = df["season"].map(season_key)
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Individual sports
# --------------------------------------------------------------------------


def load_masters() -> pd.DataFrame:
    """Masters leaderboards, 1970-2023."""
    df = pd.read_excel(PATHS["masters"], header=1).rename(
        columns={"Position": "position", "Player": "player", "Final": "final",
                 "R4": "round4", "Year": "season"}
    )
    df = df.dropna(subset=["player", "season"]).copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df["player"] = df["player"].astype(str).str.strip()
    return df.dropna(subset=["season"]).reset_index(drop=True)


def load_tour_de_france() -> pd.DataFrame:
    """Tour de France general classification finishers, 1980-2023."""
    df = pd.read_excel(PATHS["tour_de_france"], header=1).rename(
        columns={"RANK": "position", "RIDER": "rider", "TEAM": "team",
                 "GAP": "gap", "YEAR": "season"}
    )
    df = df.dropna(subset=["rider", "season"]).copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["position"] = pd.to_numeric(df["position"], errors="coerce")
    df["rider"] = df["rider"].astype(str).str.strip().str.upper()
    return df.dropna(subset=["season"]).reset_index(drop=True)


def load_grand_slams() -> pd.DataFrame:
    """Men's tennis grand slam finals."""
    df = pd.read_csv(PATHS["grand_slams"]).rename(
        columns={"YEAR": "season", "TOURNAMENT": "tournament", "WINNER": "winner",
                 "RUNNER-UP": "runner_up", "WINNER_ATP_RANKING": "winner_rank",
                 "RUNNER-UP_ATP_RANKING": "runner_up_rank",
                 "TOURNAMENT_SURFACE": "surface"}
    )
    df = df.dropna(subset=["season", "winner"]).copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df["winner"] = df["winner"].astype(str).str.strip()
    return df.dropna(subset=["season"]).reset_index(drop=True)


def load_cricket() -> pd.DataFrame:
    """Cricket World Cup / ODI results. The sheet carries a blank leading column."""
    df = pd.read_csv(PATHS["cricket"], skiprows=1)
    df = df.rename(columns={c: str(c).strip() for c in df.columns})
    df = df.rename(
        columns={"Team 1": "team1", "Team 2": "team2", "Winner": "winner",
                 "Margin": "margin", "Ground": "ground", "Match Date": "date",
                 "Year": "season"}
    )
    cols = [c for c in ["team1", "team2", "winner", "margin", "ground", "date",
                        "season"] if c in df.columns]
    df = df[cols].dropna(subset=["team1", "team2"]).copy()
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    return df.dropna(subset=["season"]).reset_index(drop=True)


TEAM_CODE = re.compile(r"^\s*[a-z]{2,3}\s+|\s+[a-z]{2,3}\s*$")


def _clean_country(name: str) -> str:
    """FIFA rows carry a two-letter flag code glued to the country name."""
    return TEAM_CODE.sub("", str(name)).strip()


def load_womens_world_cup() -> pd.DataFrame:
    """FIFA Women's World Cup matches, 1991-2023, with goals parsed out."""
    df = pd.read_csv(PATHS["womens_world_cup"]).rename(
        columns={"Round": "round", "Home": "home", "Away": "away",
                 "Score": "score", "Venue": "venue", "Year": "season"}
    )
    df = df.dropna(subset=["score", "home", "away"]).copy()
    # Scores use an en dash and may carry a penalty-shootout suffix in brackets.
    goals = df["score"].astype(str).str.replace("–", "-", regex=False)
    parts = goals.str.extract(r"(\d+)\s*(?:\(\d+\))?\s*-\s*(?:\(\d+\)\s*)?(\d+)")
    df["home_goals"] = pd.to_numeric(parts[0], errors="coerce")
    df["away_goals"] = pd.to_numeric(parts[1], errors="coerce")
    df["home"] = df["home"].map(_clean_country)
    df["away"] = df["away"].map(_clean_country)
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["home_goals", "away_goals", "season"])
    df["goal_diff"] = (df["home_goals"] - df["away_goals"]).abs()
    return df.reset_index(drop=True)


def load_kentucky_derby() -> pd.DataFrame:
    """Kentucky Derby winners, 1875-present."""
    df = pd.read_csv(PATHS["kentucky_derby"]).rename(
        columns={"Year": "season", "Winner": "winner", "Sire": "sire",
                 "Dam": "dam", "Time": "time"}
    )
    df["season"] = pd.to_numeric(df["season"], errors="coerce").astype("Int64")
    return df.dropna(subset=["season", "winner"]).reset_index(drop=True)


LOADERS = {
    "nba": load_nba,
    "nfl": load_nfl,
    "ncaaf": load_ncaaf,
    "nba_matches": load_nba_matches,
    "masters": load_masters,
    "tour_de_france": load_tour_de_france,
    "grand_slams": load_grand_slams,
    "cricket": load_cricket,
    "womens_world_cup": load_womens_world_cup,
    "kentucky_derby": load_kentucky_derby,
}
