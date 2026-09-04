# Data gaps

What the Sloan deck presents that this repo cannot reproduce, and why.

## Missing source files

The deck's largest section is European soccer, and none of its data is in the
repo. The notebook `Soccer_simulation_with_ties_and_wins (1).ipynb` reads two
files that were never committed:

| File | Used for | Status |
|---|---|---|
| `Premier_Teams_Ranking_Enhanced.xlsx` | every Premier League panel, the ties/`b` fit | missing |
| `/content/premier_scorediff_dominance_adjusted.xlsx` | score-difference vs dominance | missing (Colab path) |

Also referenced by the deck but absent entirely:

| Sport | Deck panels | Status |
|---|---|---|
| Bundesliga, Serie A, Ligue 1, La Liga | upsets %, score diff, HHI, RSD scatters | no data |
| MLB | upsets %, closeness of win pct, 40 seasons of ranked-wins fits | no data |
| NHL | ties simulation, point-percentage spread | no data |
| NCAA March Madness | listed under "Data Collected" | no data |
| Formula One | listed under "Data Collected" | no data |
| Rugby Championship | "Miscellaneous Sports" panel | no data |
| Men's FIFA World Cup | "Miscellaneous Sports" panel | no data |

The `.numbers` files in `Harvested Data/` (`ncaa-womens-tournament-history`,
`Standard_Deviation_of_Score_Differences_by_Year`) are Apple Numbers archives,
not readable by pandas. Export them to CSV to bring them into the pipeline.

**Consequence.** Everything the deck claims about soccer, MLB and the NHL is
taken on trust here. The simulation machinery for those sports *is*
implemented and tested (`simulate_season_with_ties`, `score_grid_with_ties`),
so dropping the files in and running `scripts/04_fit_simulation.py` with a new
loader is all that is required.

## Reconstructed definitions

The deck states axis labels but not formulas for the "Miscellaneous Sports"
panels. `scripts/03_individual_sports.py` reconstructs each one from the axis
label and the observed value range, and documents its choice in the function
docstring. These should be treated as *a* defensible reading, not *the*
original definition:

| Panel | Reconstruction | Confidence |
|---|---|---|
| Masters new players vs. top-10 | first-time entrants; top-10 with a prior top-10 | medium — range 14-43 matches the deck |
| Tour de France | first-time starters; summed prior top-10s of this year's top 10 | medium |
| Grand Slams | 2-year rolling window of distinct champions | medium — a 2-year window is what produces the deck's 3-8 x-range |
| Cricket | SD of runs margin; champion's cumulative titles | low — "SD" is unlabelled in the deck |
| Women's World Cup | mean absolute goal margin; champion's title count | medium |

## Definitions that differ deliberately

* **First-season handling.** `dominance_index` skips the earliest season in a
  dataset, which has no history to draw on. The original NFL script skipped
  the *last* row instead, because it iterated the file in its stored
  (descending) order. See `reports/audit_nfl_alignment.csv`.
* **NFL elite threshold.** The deck defines the NFL elite as "12 or more
  wins", a fixed cut across seasons of 14, 16 and 17 games. This is preserved
  in `leagues.nfl_panel` but is not scale-free; `07_metric_robustness.py`
  quantifies how much it matters.

## Known defect in the original code

`Simulation.py` line 15 does not parse:

```python
p1 = (s1 ** c) / (s1 ** c + s2 ** c)s
```

The trailing `s` is a typo, so the file has never run as committed. It is left
untouched as the historical record; `src/durable_dominance/simulation.py` is
the working implementation. Note also that this line uses `skill ** c`, while
the deck and the soccer notebook both use `c ** skill` — the latter is what is
implemented here.
