# Appendix — audit of the earlier analysis

*This is a methodological appendix, not the project's findings. The research
itself is in [`DOMINANCE_FINDINGS.md`](DOMINANCE_FINDINGS.md).*

Before building the current study, the earlier slide-deck analysis in this
repo was reconstructed from the raw data to see which of its measures were
safe to reuse. Every published figure reproduces exactly; the problems are in
the metrics and the inference, not the arithmetic. Three findings shaped the
main study's design:

* **Use scale-free balance metrics.** The deck's "SD of win%" is
  `sqrt(sum |w − .5|)`, which grows with league size; HHI of wins has a `1/n`
  floor. Both are largely proxies for team count. The current study uses
  Noll-Scully, Gini and the spread of fitted strength instead.
* **Align on season labels, never on position.** A row-ordering bug paired
  each NFL season's balance with a different season's dominance.
* **Detrend before correlating.** Cumulative dominance counts rise with time
  by construction, so they correlate with anything else that trends.

## 1. What reproduces

| Deck figure | Reproduced | Evidence |
|---|---|---|
| NBA dominance index, 1981-82 = 75.0 | exact | `leagues.nba_panel()` |
| NBA win% spread axis, 1.5-2.15 | exact | 1.65-1.89 observed |
| NFL win% spread axis, 1.9-2.4 | exact | 2.04-2.19 observed |
| NBA MOV spread axis, 0.20-0.31 | exact | 0.21-0.28 observed |
| NCAAF SRS spread, 9.5-13 | exact | 9.28-12.87 observed |
| NBA 1988-present MOV: flat line | exact | R² = 1.3e-5 |
| Fitted global `c` = 0.908 | adjacent grid point | fitted 0.898 |
| Fitted NBA `s` range 4-10.2 | close | fitted 2.86-8.57 |
| "Simulated CB vs. real win% spread", R² 0.61 | close | R² 0.57 (NBA), 0.69 (NFL) |

The simulator works. Fitted `s` tracks real competitive balance closely, which
is the deck's own validation test, and it passes.

## 2. The model has one parameter, not two

`P(win) = c^s1/(c^s1 + c^s2) = sigmoid(ln(c)·(s1 − s2))`, and skill
differences are `Normal(0, s√2)`. So results depend on `(s, c)` only through

    λ = s · |ln c|

Every `(s, c)` on a λ contour gives an **identical** distribution. Verified to
floating-point precision: at `s = 5.31, c = 0.908` and `s = 2.65, c = 0.825`
(matched λ) the mean ranked-win curves agree exactly; changing λ alone moves
the top team from 62.0 wins to 73.1.

This explains a puzzle in the deck: the fitted `c` is the *same constant* for
every league and every season. It is not being estimated. The 50×50 grid has a
ridge of equally good optima, and `c` is pinned by the grid edge while `s`
absorbs all variation. A 1-D sweep over λ recovers the same fit at 1/50th the
cost.

The degeneracy is exact until the `[0, 100]` skill clip binds, around `s > 10`.
Every fitted value here (NBA 2.9-8.6, NFL 1.8-10.0) is inside the exact region.

## 3. Three threats to the published conclusions

### A. The NFL result is a row-ordering artifact

`NFL dominance.py` builds its dominance list from `df['Year'].unique()` —
file order, **descending** from 2023 — and its balance list from
`groupby('Year')`, which sorts **ascending**. Slicing both with `[:48]` pairs
1970's balance with 2022's dominance, 1971's with 2021's, and so on.

| Pairing | slope | R² | p |
|---|---|---|---|
| Original positional slice | **+23.3** | 0.109 | **0.022** |
| Aligned on season label | −9.8 | 0.020 | 0.334 |

The published positive NFL relationship is an artifact of the misalignment.

### B. The NCAAF result is a shared time trend

NCAA football's dominance measure is a *cumulative count* of prior top-10
finishes, so it rises mechanically with time: **R² = 0.85 against year alone**.
SRS spread also declines over time. Correlating them measures the trends.

| | raw R² | detrended R² | detrended p |
|---|---|---|---|
| NCAAF, SRS spread | 0.488 | **0.014** | 0.392 |
| NCAAF, win% SD | 0.159 | 0.080 | 0.039 |
| NBA, deck's win% spread | 0.336 | 0.199 | 0.003 |

### C. The NBA result is league expansion

This is the decisive one. The deck's two strongest metrics are the two that
are **not scale-free**:

* `paper_cb_win` sums `|w − .5|` across teams *without dividing by n*, so it
  grows with league size — R² **0.45** against team count alone.
* `hhi_wins` has a 1/n floor, so it shrinks with league size — R² **0.98**
  against team count alone.

The NBA went from 23 teams to 30 across the sample. Adding team count as a
covariate:

| Metric | scale-free | raw p | controlling for team count |
|---|---|---|---|
| deck's win% spread | no | 0.000 | **0.492** |
| HHI of wins | no | 0.000 | **0.396** |
| SD of win% | yes | 0.510 | 0.607 |
| Noll-Scully | yes | 0.792 | 0.719 |
| RSD of wins | yes | 0.429 | 0.573 |
| mean \|win% − .5\| | yes | 0.271 | 0.481 |

**All 12 NBA tests are null once league size is controlled** (p ≥ 0.19), in
both the full sample and the 1981-2000 window where the effect looked
strongest. Across all three leagues, 9 of 24 tests are significant raw; 3 of
24 survive.

The four properly normalised metrics show nothing even *before* controlling.
The relationship the deck reports is a league-size effect.

## 4. The simulation contradicts the hypothesis

The deck's own simulation output can be tested against the hypothesis
directly. Fitted `s` is the simulated inverse of competitive balance, so if
balance entrenches elites, `s` should move with dominance:

| League | relationship | R² | p |
|---|---|---|---|
| NBA | `s` vs. real win% spread | 0.568 | 5.3e-09 |
| NFL | `s` vs. real win% spread | 0.695 | 5.2e-15 |
| NBA | **`s` vs. dominance** | 0.018 | **0.399** |
| NFL | **`s` vs. dominance** | 0.025 | **0.255** |

The simulation is well calibrated to competitive balance and shows no relation
to dominance.

## 5. Machine learning

### Season level: no out-of-sample signal

Forward-chaining cross validation (train on the past, predict the future), six
balance metrics as features:

| League | Baseline | Lasso | ElasticNet | Random Forest |
|---|---|---|---|---|
| NBA | −9.69 | −7.62 | −7.74 | −6.49 |
| NFL | −7.29 | −2.22 | −3.26 | −2.45 |
| NCAAF | −0.87 | −1.20 | −1.04 | −0.97 |

All out-of-sample R² are negative — worse than predicting a constant. The
models beat the naive baseline, so there is *some* information, but nowhere
near enough to predict. With 42-54 seasons and 6 collinear predictors this is
what should be expected, and it means the deck's in-sample R² values are not
evidence of a predictive relationship.

### Team level: strongly predictable

Reframing to ~9,000 team-seasons — *given this team's season, will it be elite
next season?* — with a strict temporal split:

| League | base rate | Logistic | Gradient boosting |
|---|---|---|---|
| NBA | 13.5% | **0.854** | 0.742 |
| NFL | 15.9% | **0.763** | 0.691 |
| NCAAF | 7.9% | **0.895** | 0.882 |

Permutation importance on held-out NBA data:

| Feature | drop in AUC |
|---|---|
| margin of victory | **0.119** |
| win % | 0.063 |
| prior elite finishes | 0.020 |
| elite this season | 0.013 |
| historical elite rate | −0.015 |

Dominance persists because **good teams stay good**, not because history
compounds. Current-season strength is 6× more informative than the entire
prior record. That is a mechanism the balance hypothesis does not require.

### Data-driven eras

PCA (74% variance in PC1) plus k-means on the balance metrics, and a binary
segmentation change-point scan on the NBA dominance series, put breaks at
**1988, 1994, 2004, 2012** — not at the deck's hand-picked 1981 / 1990 / 2000 /
2003. Since the deck's conclusions differ by era window, and the windows were
chosen by hand, this matters.

## 6. Individual sports

| Sport | relationship | R² | p |
|---|---|---|---|
| Masters | more new players → fewer returning top-10 | 0.285 | 3.3e-05 |
| Tour de France | more new entrants → less top-10 pedigree | 0.168 | 0.006 |
| Grand Slams | more distinct champions → smaller combined haul | 0.082 | 0.014 |
| Cricket World Cup | wider margins → more entrenched champion | 0.476 | 0.013 |
| Women's World Cup | no signal (n = 9) | 0.068 | 0.498 |

These are directionally consistent with the deck. All are small-n and none
were corrected for the time trends that proved decisive above; they should be
treated as descriptive.

## 7. Engineering notes

**Speed.** The fitting grid is 50×50 × 100 replicates × 43 seasons ≈ 10.7M
season simulations. Written as a Python loop over matches this is
days of compute. Two changes make it 82 seconds:

* Use `sigmoid(ln(c)·Δskill)` instead of evaluating `c**skill` twice per match.
* Resolve a whole season with one matrix product against the schedule's
  incidence matrix, batched over replicates.

Common random numbers across the grid also make the MSE surface smooth enough
to read, which is what makes the λ ridge visible.

**Storage.** Ten raw files disagree about what a season is (`"2000-2001"` vs
`2001`), which is what makes cross-sport joins error-prone — the NFL alignment
bug is exactly this class of error. `scripts/08_build_database.py` normalises
everything into SQLite behind one `season` dimension table with foreign keys.

| Source | Operation | Seconds | Relative |
|---|---|---|---|
| Excel (openpyxl) | load 3 leagues | 0.346 | 54.6× |
| CSV | load 3 leagues | 0.006 | 1.0× |
| SQLite | load 3 leagues | 0.012 | 1.8× |
| SQLite (indexed) | 50× era slice | 0.013 | 2.0× |
| CSV (rescan) | 50× era slice | 0.046 | 7.3× |

Re-parsing Excel dominates the pipeline's runtime at 55× the cost of CSV.
Indexed range queries beat re-scanning CSVs by 3.7×.

## 8. What this does not settle

The hypothesis may still be true. What the available data cannot do is
demonstrate it:

* **Missing leagues.** Soccer, MLB and the NHL — the deck's largest sections —
  have no data in the repo. See `DATA_GAPS.md`.
* **Power.** 42-54 seasons per league is too few to detect a modest effect
  alongside league-size and time confounds.
* **Design.** Season-level correlation cannot separate "balance entrenches
  elites" from "both respond to expansion, revenue sharing, or the draft".

The productive next step is the team-level framing, where the data is 200×
larger and the signal is unambiguous.
