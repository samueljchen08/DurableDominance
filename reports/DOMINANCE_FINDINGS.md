# Durable Dominance in Sports

How concentrated is championship success, and how much of that concentration
is explained by the teams simply being better?

Five studies. The NBA carries the weight because it is the only league here
with both a verified champion for every season (1980-81 to 2022-23) and a
complete head-to-head matrix; NCAA football, golf, cycling, tennis, cricket
and the Women's World Cup extend the comparison.

---

## The one-line answer

**Championships are about twice as concentrated as the teams' own measured
quality can explain, and that gap is a multi-season effect, not a playoff-luck
effect.** Fourteen franchises won the last 43 NBA titles. A random walk driven
by each season's real team strengths says there should have been twenty.

---

## Study A — Champion pedigree vs. league competitiveness

For every season: how decorated was the eventual champion (six
backward-looking measures), and how tight was the league (eight balance
measures)? That is 84 correlations, so they are read against a
Benjamini-Hochberg false-discovery correction rather than raw p-values, and
both sides are detrended on year first because pedigree counts accumulate
mechanically.

| | count |
|---|---|
| pairs tested | 84 |
| significant raw (p < .05) | 8 |
| significant detrended (p < .05) | 7 (≈4 expected by chance) |
| **surviving FDR (q < .10)** | **3** |

| League | Pedigree | Balance | r (detrended) | q |
|---|---|---|---|---|
| NCAAF | prior titles | Gini of wins | **+0.512** | 0.006 |
| NCAAF | prior titles | SD of win% | +0.452 | 0.017 |
| NCAAF | titles, last 10 yrs | Gini of wins | +0.457 | 0.017 |

Every surviving measure is oriented so a higher value means a *less* balanced
season. The positive sign therefore says: **less competitive seasons crown
more decorated champions.** Imbalance and entrenchment travel together — the
intuitive direction, and the opposite of the hypothesis that balance reinforces
elites.

Nothing survives in the NBA. Its champions are decorated regardless of how
tight the league was: the average NBA champion arrives with 2.1 prior titles,
5.1 prior top-4 finishes and a .570 career record, and 14 of 43 were
first-time winners.

---

## Study B — Championships against a random walk

A postseason is a short random walk. Each series is seven Bernoulli draws
weighted by the strength gap plus home court; win four and advance. Feeding
each season's *real* strengths through that process gives the title
distribution a league should produce if nothing carried across seasons except
quality already visible in the standings.

Strength comes from a Bradley-Terry fit to the full head-to-head matrix, so a
team is credited for *who* it beat. The matrix is complete and reconciles
exactly with the standings (82 games per team, zero mismatches).

### The model is calibrated

Scaling every strength gap by a temperature `γ` and maximising the likelihood
of the 43 real champions gives **γ = 1.0** — the raw fitted gaps need no
adjustment. The likelihood is flat from 0.8 to 1.4 (within 2 log-likelihood
units), so this pins the scale only loosely; what matters is that 1.0 sits
comfortably inside that range rather than at an edge. The reliability curve
below is the sharper check.

| Predicted title probability | Observed rate | n |
|---|---|---|
| 0.005 | 0.003 | 397 |
| 0.033 | 0.024 | 85 |
| 0.070 | 0.069 | 58 |
| 0.144 | 0.169 | 65 |
| 0.268 | 0.257 | 35 |
| 0.516 | 0.533 | 30 |

Well calibrated across the whole range, including the favourites.

### But reality is far more concentrated

Drawing 20,000 alternate 43-season leagues from the model:

| Statistic | Real | Simulated median | 95% interval | p |
|---|---|---|---|---|
| unique champions | **14** | 20 | 16 – 23 | 0.004 |
| title HHI | **0.120** | 0.073 | 0.058 – 0.098 | 0.001 |
| top team's share | **0.233** | 0.140 | 0.093 – 0.209 | 0.004 |
| title entropy | **2.345** | 2.781 | 2.551 – 2.976 | 0.0004 |
| back-to-back rate | **0.286** | 0.143 | 0.048 – 0.262 | 0.015 |

All five outside the 95% interval, all in the same direction. The result holds
for all three strength estimators (Bradley-Terry, margin of victory, raw
record) and with or without home court — p ≤ 0.004 in every combination.

### Who converts strength into rings

| Franchise | Expected titles | Actual | Excess |
|---|---|---|---|
| Los Angeles Lakers | 4.9 | 10 | **+5.1** |
| Chicago Bulls | 3.6 | 6 | +2.5 |
| Golden State Warriors | 2.4 | 4 | +1.6 |
| Miami Heat | 1.6 | 3 | +1.4 |
| … | | | |
| Utah Jazz | 1.5 | 0 | −1.5 |
| Phoenix Suns | 2.4 | 0 | **−2.4** |

The Suns earned 2.4 expected titles and won none. Read the aggregate test
above rather than this table: across 39 franchises a largest |z| near 2.8 is
roughly what chance alone produces.

---

## Study C — Where the excess comes from

Study B leaves a puzzle: the simulation already uses real strengths, and its
per-season predictions are calibrated, yet the 43-season record is far more
concentrated. Both can be true, because **calibration is a marginal property
and concentration is a joint one.** The gap means championship outcomes are
positively correlated across seasons for the same franchise, beyond what
measured strength explains.

**Quality is persistent, and that is already priced in.** Lag-1
autocorrelation of team strength is ρ = 0.58 (half-life 1.3 seasons). Notably,
within-franchise variation (SD 0.61) dwarfs between-franchise variation
(SD 0.23): franchises cycle rather than being permanently good.

**Is beating your strength a stable franchise trait?** Splitting history in
half and correlating each franchise's excess titles across the two halves:
**r = +0.24, p = 0.23, n = 28.** Positive, but not significant.

**Is it one era?**

| Era | Unique champions (real) | Simulated median | p |
|---|---|---|---|
| 1980-1994 | 6 | 9 | 0.203 |
| 1995-2009 | 6 | 10 | **0.003** |
| 2010-2023 | 9 | 9 | 0.214 |
| **1980-2023** | **14** | **20** | **0.002** |

The effect is strongest in 1995-2009 and **absent in 2010-2023**, where the
real and simulated counts are identical. Durable dominance in the NBA looks
like a 20th-century phenomenon that the modern league no longer shows.

**How big would the missing factor be?** Giving each franchise a fixed,
persistent strength bonus and sweeping its size, the observed concentration is
reproduced at a bonus SD of about **0.38 log-odds** — roughly 57% of the
within-season spread of team strength itself (0.67). That is a large latent
effect, which is itself a reason to treat the single-latent-trait explanation
sceptically.

---

## Study D — Can machine learning beat the physics?

Predicting the champion from regular-season features, trained on 1980-2008 and
tested on 2009-2023 (678 playoff team-seasons, 43 champions):

| Model | Held-out log-loss | AUC |
|---|---|---|
| **Random-walk simulation** | **0.1713** | 0.890 |
| Random forest | 0.1717 | 0.893 |
| Logistic regression | 0.1767 | 0.869 |
| Logistic + simulation as a feature | 0.1802 | 0.866 |
| Gradient boosting | 0.1990 | 0.865 |
| Uniform over the playoff field | 0.2383 | 0.500 |

**The generative simulation, with no learned parameters, matches or beats every
learned model.** Adding it as a feature to a logistic regression does not help,
which says it already contains what the other features carry.

Permutation importance ranks conference seed first (0.039), then prior top-4
finishes (0.017), margin of victory (0.014) and fitted strength (0.014).
Prior titles contribute ~0 and career win% is *negative* — franchise history
adds nothing once current quality is known.

Testing the Study B residual directly: a history-only model reaches AUC 0.540
out of sample, and the correlation between prior titles and (actual −
simulated) is +0.061, p = 0.373. **You cannot predict which franchises will
beat the random walk from their history.**

---

## Study E — A cross-sport atlas

Sports have wildly different field sizes, so each is scored against its own
null: draw a champion uniformly from the field that actually competed that
year. The **dominance multiple** is real concentration ÷ null concentration.

| Sport | Editions | Median field | Unique champions | Dominance multiple | p |
|---|---|---|---|---|---|
| Tour de France | 44 | 143 | 23 | **3.14×** | <0.001 |
| NCAA football | 54 | 120 | 20 | 2.89× | <0.001 |
| NBA | 43 | 29 | 14 | 2.23× | <0.001 |
| Women's World Cup | 9 | 16 | 5 | 2.09× | 0.001 |
| Tennis majors | 292 | 6 | 83 | 1.79× † | <0.001 |
| Masters golf | 54 | 50 | 37 | 1.71× | <0.001 |
| Cricket World Cup | 12 | 11 | 5 | 1.70× | 0.042 |

† field approximated by that year's finalists, so this is a lower bound.

Every sport is significantly more concentrated than chance among its own
entrants. Individual endurance sport (the Tour) is the most dominated;
short-format team tournaments the least.

The Kentucky Derby is reported descriptively only — 124 distinct sires in 148
runnings, top share 2%, effectively no durable dominance — because a sire is
only eligible in years he has three-year-olds running and that field is not
observable in the data.

---

## Conclusions

1. **Durable dominance is real and measurable.** Every sport tested is 1.7–3.1×
   more concentrated than chance among its own field.

2. **In the NBA it exceeds what team quality explains, by about 2×.** This is
   the strongest claim here, and it survives every robustness check applied:
   three strength estimators, home court on or off, and era splits.

3. **It is a multi-season effect, not playoff luck.** The single-season model
   is well calibrated; what it misses is the correlation of outcomes across
   seasons for the same franchise.

4. **But the mechanism is not identified.** Over-performance does not replicate
   across halves of history (p = 0.23) and is not predictable out of sample
   (AUC 0.54). Candidate explanations the data cannot separate: continuity of
   personnel that the standings under-measure, elite teams coasting in the
   regular season, playoff-specific quality, or simply that 43 seasons is a
   short record in which a few unusual runs dominate the statistics.

5. **Imbalance and entrenchment travel together** where they are detectable at
   all (NCAA football), which is the opposite of the idea that competitive
   balance reinforces elites.

6. **The modern NBA no longer shows the effect.** 2010-2023 matches its
   simulation exactly. Whether that is parity, player movement, or too short a
   window to detect is the natural next question.

## What would settle it

* **Playoff series data.** Everything here infers postseason behaviour from
  regular-season strength. Actual series results would let the model be tested
  where it matters rather than calibrated to a single outcome per year.
* **Roster continuity.** The leading candidate mechanism — that dynasties keep
  personnel the standings cannot see — is directly measurable and absent here.
* **More leagues with verified champions.** The NBA is one league and 43
  seasons; the concentration test needs long records to have power, which is
  why the era splits individually do not reach significance.

---

## Study F — Do tight regular seasons crown new champions?

Study A asked this as a correlation. This asks it as a rate: in the most
competitive third of seasons, how often is the champion a historical
powerhouse?

**The base rate first.** 71% of NBA champions across 38 seasons had already
won a title; 67% of college football's across 49. Repeat winners are the norm.

**Definitions.** A champion counts as a powerhouse under four escalating
tests — it had won before; it was among the three most decorated teams that
season; its career win% sat in the top quartile of its contemporaries; or it
already held three or more titles. The middle two are measured relative to
contemporaries, because raw title counts grow mechanically with time.

Tightness is measured seven ways, including the one the question names (mean
absolute distance from the median win%), plus the spread of records and of
scoring margin, the Gini of wins, the best-to-worst gap, and the upset rate.

**Why not 28 separate tests.** The balance measures correlate at 0.8-0.99;
one principal component absorbs 82% of their variance in the NBA and 58% in
college football. The primary analysis collapses them into a single tightness
index and runs eight logistic regressions — one per league and powerhouse
definition, controlling for year.

### Primary test

Odds of a powerhouse champion per 1 SD tighter season:

| League | Powerhouse definition | Base rate | Odds ratio | 95% CI | p | q |
|---|---|---|---|---|---|---|
| NBA | had won before | 71% | 0.93 | 0.43–2.00 | 0.855 | 0.855 |
| NBA | top-3 most decorated | 42% | 0.71 | 0.33–1.50 | 0.366 | 0.488 |
| NBA | top-quartile career win% | 47% | 0.71 | 0.34–1.47 | 0.350 | 0.488 |
| NBA | already had 3+ titles | 37% | 0.48 | 0.22–1.07 | 0.072 | 0.191 |
| NCAAF | had won before | 67% | 0.74 | 0.34–1.60 | 0.442 | 0.505 |
| **NCAAF** | **top-3 most decorated** | 29% | **0.27** | 0.11–0.66 | **0.004** | **0.033** |
| NCAAF | top-quartile career win% | 90% | 0.44 | 0.09–2.09 | 0.298 | 0.488 |
| **NCAAF** | **already had 3+ titles** | 20% | **0.26** | 0.09–0.76 | **0.014** | **0.057** |

**All eight point the same way.** Two survive the false-discovery correction,
both in college football and both under the strictest definitions.

### In plain numbers

Splitting college football's seasons into thirds by the average team's
distance from the median win%:

| | Tightest third | Loosest third |
|---|---|---|
| champion was a top-3 decorated programme | **6%** | **59%** |

Across the full 7 × 4 robustness grid, 34 of 52 cells show a lower powerhouse
rate in tighter seasons. Because the measures are near-duplicates, that is one
consistent pattern rather than 34 independent confirmations.

### Reading it

The effect is real but narrow: college football, strictest definitions,
directionally the same in the NBA without reaching significance (odds ratio
0.48 for a three-time champion, p = 0.07 over 38 seasons). That fits the
institutions — college football has no draft, no cap and the widest talent
gaps, so a loose season is exactly when the blue bloods should separate.

It does not contradict Study B. That excess concentration is an
*across-season* effect; this is a *within-season* one. A league can crown
fresh champions in its tightest years and still concentrate titles across four
decades far more than quality alone explains.

**The NFL is excluded.** This repo has its complete standings from 1970 but no
record of who won the title, and a champion cannot be inferred from a
regular-season table.
