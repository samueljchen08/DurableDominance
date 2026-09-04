"""Monte Carlo season simulation.

Implements the model described in *Durable Dominance in Sports*:

    Team skill:   s_i ~ Normal(mean, s), clipped to [lower_bound, upper_bound]
    Match:        P(team 1 wins) = c**s1 / (c**s1 + c**s2)
    With ties:    P(team 1 wins) = b * c**s1 / (b * c**s1 + c**s2)
                  P(team 2 wins) = b * c**s2 / (b * c**s2 + c**s1)
                  P(tie)         = 1 - P1 - P2

`s` controls the spread of team strength (the simulated inverse of competitive
balance); `c` controls how reliably the stronger team wins; `b` (<= 1) pulls
probability mass out of both win outcomes and into ties.

Note on the sign of `c`: the paper's grid search runs over c in (0.5, 1], so
c**s is *decreasing* in skill and the lower-skill team is favoured. Because
every downstream comparison is made on the *sorted* win vector, the sorted
distribution is identical under c -> 1/c, so this does not affect any published
result. See `docs/METHODS.md` and `scripts/06_identifiability.py`.
"""

from __future__ import annotations

import numpy as np

DEFAULT_SKILL_MEAN = 50.0
DEFAULT_LOWER_BOUND = 0.0
DEFAULT_UPPER_BOUND = 100.0

WIN, TIE, LOSS = 1, 0, -1


def generate_skills(
    t: int,
    s: float,
    mean: float = DEFAULT_SKILL_MEAN,
    lower_bound: float = DEFAULT_LOWER_BOUND,
    upper_bound: float = DEFAULT_UPPER_BOUND,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Draw `t` team skills from Normal(mean, s), clipped to the bounds."""
    rng = rng if rng is not None else np.random.default_rng()
    return np.clip(rng.normal(mean, s, t), lower_bound, upper_bound)


def win_probability(s1: float, s2: float, c: float, b: float = 1.0) -> float:
    """P(team 1 beats team 2). `b` < 1 reserves probability mass for ties."""
    a1 = b * c**s1
    return a1 / (a1 + c**s2)


def create_schedule(t: int, m: int) -> list[tuple[int, int]]:
    """Greedy schedule giving every team ~`m` games (the paper's construction)."""
    matches: list[tuple[int, int]] = []
    match_counts = dict.fromkeys(range(t), 0)
    target = t * m // 2
    while len(matches) < target:
        added = False
        for i in range(t):
            for j in range(i + 1, t):
                if match_counts[i] < m and match_counts[j] < m:
                    matches.append((i, j))
                    match_counts[i] += 1
                    match_counts[j] += 1
                    added = True
                    if len(matches) == target:
                        return matches
        if not added:
            # No pair can take another game; the schedule is as full as it gets.
            break
    return matches


def simulate_season(
    skills: np.ndarray,
    schedule: list[tuple[int, int]],
    c: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Play every scheduled match; return each team's win count."""
    rng = rng if rng is not None else np.random.default_rng()
    wins = np.zeros(len(skills), dtype=int)
    draws = rng.random(len(schedule))
    for (i, j), u in zip(schedule, draws):
        if u < win_probability(skills[i], skills[j], c):
            wins[i] += 1
        else:
            wins[j] += 1
    return wins


def simulate_season_with_ties(
    skills: np.ndarray,
    schedule: list[tuple[int, int]],
    c: float,
    b: float,
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Play every match allowing draws; return wins, ties, losses and points.

    Points follow the 3-1-0 convention used throughout the soccer analysis.
    """
    rng = rng if rng is not None else np.random.default_rng()
    n = len(skills)
    wins = np.zeros(n, dtype=int)
    ties = np.zeros(n, dtype=int)
    losses = np.zeros(n, dtype=int)
    draws = rng.random(len(schedule))

    for (i, j), u in zip(schedule, draws):
        p1 = win_probability(skills[i], skills[j], c, b)
        p2 = win_probability(skills[j], skills[i], c, b)
        p_tie = max(0.0, 1.0 - p1 - p2)
        if u < p1:
            wins[i] += 1
            losses[j] += 1
        elif u < p1 + p_tie:
            ties[i] += 1
            ties[j] += 1
        else:
            wins[j] += 1
            losses[i] += 1

    return {
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "points": 3 * wins + ties,
    }


def ranked_wins(
    t: int,
    m: int,
    s: float,
    c: float,
    schedule: list[tuple[int, int]] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """One season's win totals, sorted high to low."""
    rng = rng if rng is not None else np.random.default_rng()
    schedule = schedule if schedule is not None else create_schedule(t, m)
    skills = generate_skills(t, s, rng=rng)
    return np.sort(simulate_season(skills, schedule, c, rng=rng))[::-1]


def effective_spread(s: float, c: float) -> float:
    """The single parameter the win distribution actually depends on.

    P(win) is a function of (s1 - s2) * ln(c) alone, and s1 - s2 ~ Normal(0,
    s * sqrt(2)) before clipping. So (s, c) only enter through s * |ln c|.
    """
    return abs(s * np.log(c))
