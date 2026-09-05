"""Team strength estimates for a season.

Three estimators, in increasing order of how much they use:

* ``win_pct_strength``  -- log-odds of the raw record. Ignores schedule.
* ``mov_strength``      -- point margin per game, scaled to log-odds. Uses
  scoring information, which is a less noisy signal of quality than wins.
* ``bradley_terry``     -- maximum-likelihood fit to the full head-to-head
  matrix, so a team is credited for *who* it beat, not just how often it won.

Bradley-Terry defines ``P(i beats j) = sigmoid(beta_i - beta_j)``. The fit is
by L-BFGS on the penalised log-likelihood; the ridge term keeps `beta` finite
when a team sweeps or is swept by an opponent, and pins the otherwise
unidentified additive constant. Strengths are returned centred at zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, log_expit

# One point of NBA margin is worth roughly this much in log-odds of winning a
# game. Calibrated in `calibrate_mov_scale` from the seasons in this repo.
DEFAULT_MOV_SCALE = 0.20


def _pairs(season_matches: pd.DataFrame) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Collapse the mirrored head-to-head table to one row per unordered pair."""
    teams = sorted(set(season_matches["team"]) | set(season_matches["opponent"]))
    idx = {t: i for i, t in enumerate(teams)}
    seen: dict[tuple[int, int], tuple[float, float]] = {}
    for r in season_matches.itertuples():
        i, j = idx[r.team], idx[r.opponent]
        if i < j:
            seen[(i, j)] = (r.series_wins, r.series_losses)
        elif (j, i) not in seen:
            seen[(j, i)] = (r.series_losses, r.series_wins)
    keys = list(seen)
    ii = np.array([k[0] for k in keys], dtype=int)
    jj = np.array([k[1] for k in keys], dtype=int)
    wij = np.array([seen[k][0] for k in keys], dtype=float)
    wji = np.array([seen[k][1] for k in keys], dtype=float)
    return teams, ii, jj, np.vstack([wij, wji])


def bradley_terry(
    season_matches: pd.DataFrame, ridge: float = 1.0
) -> pd.Series:
    """MLE team strengths in log-odds units, centred at zero."""
    teams, ii, jj, w = _pairs(season_matches)
    wij, wji = w
    n = len(teams)

    def objective(beta: np.ndarray) -> tuple[float, np.ndarray]:
        d = beta[ii] - beta[jj]
        ll = wij @ log_expit(d) + wji @ log_expit(-d)
        p = expit(d)
        resid = wij * (1 - p) - wji * p          # d(ll)/d(d)
        grad = np.zeros(n)
        np.add.at(grad, ii, resid)
        np.add.at(grad, jj, -resid)
        return -(ll - 0.5 * ridge * beta @ beta), -(grad - ridge * beta)

    res = minimize(objective, np.zeros(n), jac=True, method="L-BFGS-B")
    beta = res.x - res.x.mean()
    return pd.Series(beta, index=teams, name="bt_strength").sort_values(ascending=False)


def win_pct_strength(standings: pd.DataFrame, shrink: float = 1.0) -> pd.Series:
    """Log-odds of the record, with `shrink` pseudo-wins and pseudo-losses."""
    w = standings["wins"] + shrink
    l = standings["losses"] + shrink
    return pd.Series(np.log(w / l).to_numpy(), index=standings["team"],
                     name="record_strength")


def mov_strength(standings: pd.DataFrame, scale: float = DEFAULT_MOV_SCALE) -> pd.Series:
    """Margin of victory converted to log-odds units."""
    return pd.Series((standings["mov"] * scale).to_numpy(),
                     index=standings["team"], name="mov_strength")


def calibrate_mov_scale(standings: pd.DataFrame, matches: pd.DataFrame) -> float:
    """Least-squares slope mapping margin of victory onto Bradley-Terry units.

    Fitting the two on the same seasons is what makes them comparable; without
    it the MOV and BT playoff simulations would differ only by an arbitrary
    units choice.
    """
    xs, ys = [], []
    for season, g in matches.groupby("season"):
        st = standings[standings["season"] == season]
        if len(st) < 8:
            continue
        bt = bradley_terry(g)
        common = st[st["team"].isin(bt.index)]
        xs.append(common["mov"].to_numpy())
        ys.append(bt.reindex(common["team"]).to_numpy())
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    return float((x @ y) / (x @ x))


def season_strengths(
    standings: pd.DataFrame, matches: pd.DataFrame, season, mov_scale: float | None = None
) -> pd.DataFrame:
    """All three estimates for one season, indexed by team."""
    st = standings[standings["season"] == season]
    mt = matches[matches["season"] == season]
    out = pd.DataFrame({
        "bt": bradley_terry(mt),
        "record": win_pct_strength(st),
        "mov": mov_strength(st, mov_scale if mov_scale else DEFAULT_MOV_SCALE),
    })
    meta = st.set_index("team")[["conference", "wins", "losses", "win_pct", "rank"]]
    return out.join(meta, how="inner")
