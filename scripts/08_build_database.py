"""Load every dataset into a normalised SQLite database and benchmark it.

Why a database at all, for 50k rows? Three practical reasons, each measured
below:

1. **One join key.** The raw files disagree about what a season is -- the NBA
   uses ``"2000-2001"``, everyone else uses ``2001`` -- so every cross-sport
   question currently needs bespoke glue. A `season` dimension table fixes
   that once.
2. **Query cost.** The analysis scripts re-parse ~8 MB of Excel on every run;
   openpyxl is slow enough that this dominates their runtime.
3. **Constraints.** Foreign keys and NOT NULL catch the class of error found
   in `07_metric_robustness.py`, where a positional slice silently paired
   seasons that did not belong together.

Run with `--benchmark` to reproduce the timing table.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from durable_dominance import datasets as D

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "durable_dominance.sqlite"

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE season (
    season_id   INTEGER PRIMARY KEY,
    label       TEXT    NOT NULL UNIQUE,   -- '2000-2001' or '2001'
    start_year  INTEGER NOT NULL,          -- the single sortable key
    sport       TEXT    NOT NULL
);

CREATE TABLE team_season (
    id          INTEGER PRIMARY KEY,
    season_id   INTEGER NOT NULL REFERENCES season(season_id),
    league      TEXT    NOT NULL,
    team        TEXT    NOT NULL,
    wins        REAL,
    losses      REAL,
    games       REAL,
    win_pct     REAL,
    rating      REAL,                      -- MOV / NRTG / SRS, league-specific
    finish_rank REAL,
    UNIQUE (season_id, league, team)
);

CREATE TABLE competitor_result (
    id          INTEGER PRIMARY KEY,
    season_id   INTEGER NOT NULL REFERENCES season(season_id),
    sport       TEXT    NOT NULL,
    competitor  TEXT    NOT NULL,
    position    REAL
);

CREATE TABLE season_metric (
    season_id   INTEGER NOT NULL REFERENCES season(season_id),
    league      TEXT    NOT NULL,
    metric      TEXT    NOT NULL,
    value       REAL,
    PRIMARY KEY (season_id, league, metric)
);

-- Indices chosen from the access patterns the analysis scripts actually use:
-- always "one league, a range of years", never a scan over team names.
CREATE INDEX idx_team_season_league_year ON team_season(league, season_id);
CREATE INDEX idx_team_season_team        ON team_season(team);
CREATE INDEX idx_competitor_sport_year   ON competitor_result(sport, season_id);
CREATE INDEX idx_season_start_year       ON season(start_year, sport);
"""

TEAM_LEAGUES = {
    "nba": ("nba", D.load_nba, "mov", "rank"),
    "nfl": ("nfl", D.load_nfl, "mov", None),
    "ncaaf": ("ncaaf", D.load_ncaaf, "srs", "ap_rank"),
}
COMPETITOR_SPORTS = {
    "masters": (D.load_masters, "player", "position"),
    "tour_de_france": (D.load_tour_de_france, "rider", "position"),
    "grand_slams": (D.load_grand_slams, "winner", None),
    "kentucky_derby": (D.load_kentucky_derby, "winner", None),
}


def build(db_path: Path = DB_PATH) -> Path:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.unlink(missing_ok=True)
    con = sqlite3.connect(db_path)
    con.executescript(SCHEMA)

    seasons: dict[tuple[str, str], int] = {}

    def season_id(label, sport) -> int:
        key = (str(label), sport)
        if key not in seasons:
            sid = len(seasons) + 1
            seasons[key] = sid
            con.execute(
                "INSERT INTO season (season_id, label, start_year, sport) "
                "VALUES (?,?,?,?)",
                (sid, f"{sport}:{label}", D.season_key(label), sport),
            )
        return seasons[key]

    for league, (sport, loader, rating_col, rank_col) in TEAM_LEAGUES.items():
        df = loader()
        rows = [
            (
                season_id(r["season"], sport), league, str(r["team"]),
                _num(r.get("wins")), _num(r.get("losses")), _num(r.get("games")),
                _num(r.get("win_pct")), _num(r.get(rating_col)),
                _num(r.get(rank_col)) if rank_col else None,
            )
            for _, r in df.iterrows()
        ]
        con.executemany(
            "INSERT OR IGNORE INTO team_season "
            "(season_id, league, team, wins, losses, games, win_pct, rating, finish_rank) "
            "VALUES (?,?,?,?,?,?,?,?,?)", rows,
        )
        print(f"  team_season   {league:8s} {len(rows):6d} rows")

    for sport, (loader, name_col, pos_col) in COMPETITOR_SPORTS.items():
        df = loader()
        rows = [
            (season_id(r["season"], sport), sport, str(r[name_col]),
             _num(r.get(pos_col)) if pos_col else None)
            for _, r in df.iterrows()
        ]
        con.executemany(
            "INSERT INTO competitor_result (season_id, sport, competitor, position) "
            "VALUES (?,?,?,?)", rows,
        )
        print(f"  competitor    {sport:14s} {len(rows):6d} rows")

    _load_metrics(con, seasons)
    con.commit()
    con.execute("ANALYZE")
    con.commit()
    con.close()
    print(f"\nbuilt {db_path.relative_to(ROOT)} "
          f"({db_path.stat().st_size/1024:.0f} KB)")
    return db_path


def _load_metrics(con, seasons) -> None:
    from durable_dominance import leagues as L
    cols = ["dominance", "paper_cb_win", "paper_cb_rating", "win_pct_sd",
            "noll_scully", "hhi_wins", "rsd_wins", "closeness", "rating_sd"]
    sport_of = {"nba": "nba", "nba_nrtg": "nba", "nfl": "nfl", "ncaaf": "ncaaf"}
    rows = []
    for league, fn in L.PANELS.items():
        if league == "nba_nrtg":
            continue
        panel = fn()
        for _, r in panel.iterrows():
            sid = seasons.get((str(r["season"]), sport_of[league]))
            if sid is None:
                continue
            for col in cols:
                if col in panel.columns and pd.notna(r[col]):
                    rows.append((sid, league, col, float(r[col])))
    con.executemany(
        "INSERT OR REPLACE INTO season_metric (season_id, league, metric, value) "
        "VALUES (?,?,?,?)", rows)
    print(f"  season_metric          {len(rows):6d} rows")


def _num(v):
    return None if v is None or pd.isna(v) else float(v)


def benchmark(db_path: Path = DB_PATH) -> pd.DataFrame:
    """Time the pipeline's real access patterns: Excel vs CSV vs SQLite."""
    results = []

    t0 = time.perf_counter()
    D.load_nba(); D.load_nfl(); D.load_ncaaf()
    results.append({"source": "Excel (openpyxl)", "operation": "load 3 leagues",
                    "seconds": time.perf_counter() - t0})

    proc = ROOT / "data" / "processed"
    t0 = time.perf_counter()
    for n in ("nba", "nfl", "ncaaf"):
        pd.read_csv(proc / f"{n}.csv")
    results.append({"source": "CSV", "operation": "load 3 leagues",
                    "seconds": time.perf_counter() - t0})

    con = sqlite3.connect(db_path)
    t0 = time.perf_counter()
    pd.read_sql("SELECT * FROM team_season", con)
    results.append({"source": "SQLite", "operation": "load 3 leagues",
                    "seconds": time.perf_counter() - t0})

    q = """SELECT s.start_year, t.team, t.wins
           FROM team_season t JOIN season s USING (season_id)
           WHERE t.league='nba' AND s.start_year BETWEEN 1990 AND 2000"""
    t0 = time.perf_counter()
    for _ in range(50):
        pd.read_sql(q, con)
    results.append({"source": "SQLite (indexed)", "operation": "50x era slice",
                    "seconds": time.perf_counter() - t0})

    t0 = time.perf_counter()
    for _ in range(50):
        df = pd.read_csv(proc / "nba.csv")
        df[(df["start_year"] >= 1990) & (df["start_year"] <= 2000)]
    results.append({"source": "CSV (rescan)", "operation": "50x era slice",
                    "seconds": time.perf_counter() - t0})
    con.close()

    out = pd.DataFrame(results)
    out["relative"] = out["seconds"] / out["seconds"].min()
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", action="store_true")
    args = ap.parse_args()
    build()
    if args.benchmark:
        bm = benchmark()
        bm.to_csv(ROOT / "reports" / "database_benchmark.csv", index=False)
        print("\n" + bm.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
