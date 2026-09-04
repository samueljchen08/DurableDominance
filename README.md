# Durable Dominance in Sports

Does competitive balance actually entrench a league's elite? This repo
reconstructs the analysis behind *Durable Dominance in Sports* from the raw
data, rebuilds the Monte Carlo season simulator, and stress-tests the
conclusions.

## Headline results

**The deck's central relationship does not survive scrutiny — but its
simulation does.** Every published figure reproduces from the raw data; the
problems are in the metrics and the inference, not the code's arithmetic.

| Claim | Status |
|---|---|
| NBA: more balance goes with more entrenched elites | **confounded** — the two significant metrics are 45% and 98% explained by league size alone. The NBA grew 23 → 30 teams; control for that and all 12 tests go null (p ≥ 0.19) |
| NFL: balance predicts dominance | **artifact** — the script pairs each season's balance with a *different* season's dominance. Corrected: p = 0.33 |
| NCAAF: SRS spread predicts dominance | **spurious** — dominance there is a cumulative count, 85% explained by year alone. Detrended: p = 0.39 |
| Out-of-sample predictive skill at season level | **none** — no model, linear or ensemble, beats predicting the mean |
| Simulation recovers competitive balance | **holds** — fitted `s` vs real win% spread, R² 0.57 (NBA), 0.69 (NFL) |
| Simulation's `s` relates to dominance | **no** — p = 0.40 (NBA), p = 0.25 (NFL) |
| The model's `(s, c)` are separately estimable | **no** — only `λ = s·\|ln c\|` is identified |
| Team-level dominance is predictable | **yes** — AUC 0.76–0.90, driven by current strength, not history |

Full write-up: [`reports/FINDINGS.md`](reports/FINDINGS.md) ·
Methods: [`docs/METHODS.md`](docs/METHODS.md) ·
What's missing: [`reports/DATA_GAPS.md`](reports/DATA_GAPS.md)

## Layout

```
src/durable_dominance/   the library
  datasets.py            loaders for all 10 raw files
  metrics.py             balance + dominance metrics (paper-faithful and corrected)
  leagues.py             per-league season panels
  simulation.py          readable reference simulator
  engine.py              vectorised simulator used for fitting
  fitting.py             the two-stage (s, c) grid search
  plotting.py            shared chart style
scripts/                 01-09, run in order
tests/                   engine correctness + the identifiability proof
reports/                 generated CSVs, figures and write-ups
docs/METHODS.md          model, metrics, validity threats
data/processed/          tidy CSVs (generated)
data/durable_dominance.sqlite   normalised database (generated)
```

The original exploratory scripts (`NBA Coding/`, `NFL Coding/`,
`Individual Sports/`, `Simulation.py`) are left untouched as the historical
record.

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
export PYTHONPATH=src

.venv/bin/python scripts/01_build_clean_datasets.py
.venv/bin/python scripts/04_fit_simulation.py
.venv/bin/python -m pytest tests/ -q
```

## The model

Team skills are drawn `Normal(50, s)` and clipped to `[0, 100]`; a match
resolves as `P(1 beats 2) = c^s1 / (c^s1 + c^s2)`, with an optional draw
parameter for soccer and hockey. Because this equals
`sigmoid(ln(c)·(s1 − s2))`, the whole model collapses to one estimable
parameter, `λ = s·|ln c|` — see `scripts/06_identifiability.py`.

## Data

10 datasets, 1875-2023. NBA (43 seasons), NFL (54), NCAA football (54), plus
the Masters, Tour de France, tennis majors, cricket, the Women's World Cup and
the Kentucky Derby. The soccer, MLB and NHL files the deck used were never
committed — see [`reports/DATA_GAPS.md`](reports/DATA_GAPS.md).
