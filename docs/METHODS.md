# Methods

## The model

Each team draws a latent skill

    skill_i ~ Normal(50, s), clipped to [0, 100]

and each match resolves as

    P(team 1 wins) = c^s1 / (c^s1 + c^s2)

with an optional draw parameter `b <= 1`:

    P(team 1 wins) = b*c^s1 / (b*c^s1 + c^s2)
    P(team 2 wins) = b*c^s2 / (b*c^s2 + c^s1)
    P(tie)         = 1 - P1 - P2

`s` is the spread of team strength — the simulated inverse of competitive
balance. `c` governs how deterministically the stronger team wins.

### The model is a Bradley-Terry model

Dividing through by `c^s1`:

    P = 1 / (1 + c^(s2 - s1)) = sigmoid(ln(c) * (s1 - s2))

so this is logistic in the skill difference. `engine.py` uses this form: it is
both faster and numerically safer than evaluating `c**skill` directly.

### `s` and `c` are not separately identifiable

`s1 - s2 ~ Normal(0, s*sqrt(2))` before clipping, so the win distribution
depends on the parameters only through

    lambda = s * |ln c|

Every `(s, c)` on a `lambda` contour gives an identical distribution of
results. The 50x50 grid search therefore has a *ridge* of equally good optima,
not a unique optimum. `scripts/06_identifiability.py` shows the MSE valley
tracing the analytic contour, all 2,500 grid points collapsing onto one curve
in `lambda`, and verifies that matched-`lambda` pairs agree to floating-point
precision while mismatched ones differ by many wins.

Practical consequences:

* The fitted `c` is an artifact of the grid and the tie-breaking rule, not an
  estimate. It comes out identical for every league and season because that is
  the only thing it *can* do.
* The fitted per-season `s` is meaningful only *conditional on the fixed* `c`.
  It is a rescaling of `lambda`, which is the real quantity.
* A 1-D sweep over `lambda` recovers the same fit at 1/50th the cost
  (`engine.lambda_curve`).

The degeneracy is exact until the clip to `[0, 100]` starts to bind, around
`s > 10` for a mean of 50. Every fitted value in this repo (NBA 2.9-8.6, NFL
1.8-10.0) sits inside the exact region.

## Fitting procedure

Reproduces the original notebook:

1. Grid: `s in linspace(0, 10, 50)`, `c in linspace(0.5, 1, 50)`.
2. For each grid point, simulate the season 100 times and record the MSE
   between the sorted simulated and sorted real win vectors.
3. Collapse the 100 replicates with the **mode** of the MSE distribution.
4. Choose one global `c` by lowest mean MSE across all seasons.
5. Choose per-season `s` at that fixed `c`.

Two implementation notes:

* **Common random numbers.** One block of standard normals and uniforms is
  drawn per season and reused across the whole grid, so neighbouring grid
  points differ because the parameters differ, not because the noise did. This
  makes the surface smooth enough to read.
* **The mode summary.** MSE values are discrete here (win counts are
  integers), so the mode is well defined, but it behaves closer to a minimum
  than to a centre. `fitting.py` takes `summary_stat="mean"` to switch.

Fitted global `c = 0.898` (NBA and NFL); the deck reports `0.908`. These are
adjacent points on the 50-value grid, and by the identifiability argument the
difference is absorbed into `s`.

## Metrics

### Competitive balance

| Function | Definition | Notes |
|---|---|---|
| `paper_win_pct_spread` | `sqrt(sum abs(w - 0.5))` | the deck's "SD of Win%" — not a standard deviation; no division by n, so it scales with league size |
| `paper_rating_spread` | `sqrt(sum abs(r[3:]) / N_all_rows)` | the deck's "SD of MOV/NRTG"; drops the top 3 teams and divides by the row count of the whole multi-season table |
| `win_pct_sd` | sample SD of win% | |
| `noll_scully` | `win_pct_sd / (0.5/sqrt(G))` | the standard cross-league comparable; removes season-length dependence |
| `hhi` | sum of squared shares | |
| `relative_sd` | coefficient of variation | the deck's "RSD" |
| `closeness` | mean `abs(w - 0.5)` | |

The `paper_*` versions are kept because the published charts are on their
scale, and reproducing the figures requires them. They are not recommended for
new work. `scripts/07_metric_robustness.py` re-runs every headline regression
under all six.

### Durable dominance

For each season, take the teams that finished on top and sum the share of
*previous* elite finishes they already owned. High values mean this season's
elite is the same elite as before.

| League | Elite definition | Aggregation |
|---|---|---|
| NBA | top-4 league finish | share of prior top-4 slots (%) |
| NFL | 12+ wins | share of prior 12-win slots (%) |
| NCAAF | top-10 AP finish | raw cumulative count |

The NCAAF version is a **count**, not a share, so it grows with time by
construction — it correlates with anything else that trends. This is the
subject of audit B in `07_metric_robustness.py`.

## Validity threats found

1. **Row-order misalignment (NFL).** The original script builds dominance from
   `df['Year'].unique()` (file order, descending) and balance from
   `groupby('Year')` (ascending), then slices both positionally. Each season's
   balance is paired with a different season's dominance. Fixing it flips the
   slope from +23.3 (p = 0.022) to -9.8 (p = 0.33).
2. **Spurious trend (NCAAF).** Dominance is 85% explained by year alone.
   Its R^2 = 0.49 against SRS spread falls to 0.014 (p = 0.39) once both
   series are detrended.
3. **In-sample reporting.** Every R^2 in the deck is in-sample. Under
   forward-chaining cross validation no model — linear or ensemble — achieves
   positive out-of-sample R^2 at the season level (`scripts/09`).

## Reproducing

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
export PYTHONPATH=src
.venv/bin/python scripts/01_build_clean_datasets.py
.venv/bin/python scripts/02_dominance_vs_balance.py
.venv/bin/python scripts/03_individual_sports.py
.venv/bin/python scripts/04_fit_simulation.py       # ~2 min
.venv/bin/python scripts/05_simulation_diagnostics.py
.venv/bin/python scripts/06_identifiability.py
.venv/bin/python scripts/07_metric_robustness.py
.venv/bin/python scripts/08_build_database.py --benchmark
.venv/bin/python scripts/09_machine_learning.py
.venv/bin/python -m pytest tests/ -q
```
