# Durable Dominance in Sports

How concentrated is championship success, and how much of that concentration
is explained by the teams simply being better?

**The answer: about half of it isn't.** Fourteen franchises won the last 43 NBA
championships. Run the same seasons through a random walk driven by each
team's own measured strength and you get twenty. Titles are ~1.6× more
concentrated than team quality can explain, and the gap is a multi-season
effect rather than playoff luck.

📊 **[Durable Dominance](https://claude.ai/code/artifact/e597b4b6-6252-4e15-8fa6-46f4de57001c)** — the NBA deep dive
📐 **[Cross-Sport Dominance Atlas](https://claude.ai/code/artifact/ea033b22-6bf3-4718-83ec-1cb0ebeeaa58)** — seven sports on one scale

## Headline results

| Study | Question | Result |
|---|---|---|
| **A** | Do tight leagues crown veterans? | 3 of 84 correlations survive FDR, all NCAA football, all pointing the same way: **less** balanced seasons crown more decorated champions |
| **B** | Is the title record more concentrated than strength explains? | **Yes.** 14 unique champions vs. 20 predicted (95% CI 16–23); all five concentration statistics outside the interval, p ≤ 0.015 |
| **B** | Is the simulation trustworthy? | **Yes.** MLE temperature γ = 1.1, and the reliability curve tracks the diagonal — teams given a 52% chance won 53% of the time |
| **C** | Where does the excess come from? | A multi-season effect. Per-season predictions are calibrated; what's missing is correlation of outcomes across seasons for the same franchise |
| **C** | Is it one era? | Strongest in 1995–2009, **absent in 2010–2023** |
| **D** | Can ML beat the simulation? | **No.** Random walk log-loss 0.171 vs. 0.172 (random forest) and 0.199 (gradient boosting) |
| **E** | Which sport is most dominated? | Tour de France (3.14×), then NCAA football (2.89×) and the NBA (2.23×) |

Full write-up: [`reports/DOMINANCE_FINDINGS.md`](reports/DOMINANCE_FINDINGS.md)
Methods: [`docs/METHODS.md`](docs/METHODS.md) ·
Gaps: [`reports/DATA_GAPS.md`](reports/DATA_GAPS.md)

## The model

Team strength is a Bradley-Terry fit to the complete head-to-head matrix
(48,256 games, reconciling exactly with the standings). A playoff series is
seven weighted Bernoulli draws — win four and advance:

```
P(i beats j) = σ(β_i − β_j + home court)
```

Sampling 20,000 alternate 43-season leagues from that process gives the
distribution of championship concentration a league *should* produce if
nothing carried across seasons except visible quality. Comparing the real
record to it is the durable-dominance test.

## Layout

```
src/durable_dominance/
  datasets.py    loaders for all 10 raw files behind one season key
  strength.py    Bradley-Terry, margin-of-victory and record strength
  playoffs.py    bracket, best-of-7 random walk, concentration statistics
  dominance.py   champion pedigree and season competitiveness measures
  metrics.py     balance metrics, scale-free and legacy
  leagues.py     per-league season panels
  simulation.py / engine.py / fitting.py   season simulator and grid search
  plotting.py    shared chart style
scripts/         01-09 pipeline and prior-work audit; 11-16 the studies
tests/           22 tests, incl. the simulator vs. its closed form
reports/         generated CSVs, figures, write-ups and both HTML reports
```

## Quick start

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
export PYTHONPATH=src

.venv/bin/python scripts/01_build_clean_datasets.py
.venv/bin/python scripts/11_dominance_vs_competitiveness.py
.venv/bin/python scripts/12_playoff_random_walk.py     # the core test
.venv/bin/python scripts/13_dynasty_dynamics.py
.venv/bin/python scripts/14_championship_ml.py
.venv/bin/python scripts/15_cross_sport_atlas.py
.venv/bin/python scripts/16_build_reports.py           # rebuilds both HTML reports
.venv/bin/python -m pytest tests/ -q
```

## Data

NBA 1980-81 to 2022-23 with a verified champion and full head-to-head matrix
for every season; NCAA football 1970-2023 (AP #1 as champion); plus the
Masters, Tour de France, tennis majors, cricket, the Women's World Cup and the
Kentucky Derby. The soccer, MLB and NHL files referenced by earlier notebooks
were never committed — see [`reports/DATA_GAPS.md`](reports/DATA_GAPS.md).

The original exploratory scripts (`NBA Coding/`, `NFL Coding/`,
`Individual Sports/`, `Simulation.py`) are left untouched as the historical
record; [`reports/APPENDIX_prior_work_audit.md`](reports/APPENDIX_prior_work_audit.md)
explains which of their measures were safe to reuse and why.
