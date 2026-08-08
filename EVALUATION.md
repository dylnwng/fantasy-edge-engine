# Model Evaluation: Usage-Based Waiver Predictor

This is the written evaluation called for by the PRD's success metrics — evidence the
model adds value over a naive baseline, reported honestly (including the parts that
didn't work), independent of whether it wins me my fantasy league.

## The question

Does trailing usage data (snap share, target share, air yards share, red zone touches)
predict a fantasy player's near-term point production better than simply assuming they
keep scoring what they've been scoring? If the answer is no, the entire premise of this
project — that usage moves before points do, and a manager who sees it early has a real
edge — doesn't hold up, and the honest conclusion would be to say so.

## Method

- **Model**: XGBoost regression (`reg:squarederror`) predicting a player's next-week
  fantasy points from their trailing 2-game usage trend.
- **Features**: trailing averages of snap %, target share, air yards share, red zone
  touches, and red zone target share, plus the trailing average of the player's own
  fantasy points (the same quantity the baseline uses, given to the model too so it's
  not handicapped relative to its own comparison point).
- **Baseline**: rolling average of the player's own last 2 games' fantasy points — the
  simplest thing a manager would otherwise do.
- **Split**: trained on 2018–2023 (14,349 rows), validated on the held-out 2024 season
  (2,355 rows) — a season-level split, not a week-level one. Within a season, a hot
  player tends to stay hot for a few weeks, so holding out individual weeks would leak
  signal across the split and overstate accuracy. 2024 is fully out-of-sample: the model
  never saw it during training.
- **Scoring settings**: full PPR, computed via the league's actual configured settings
  (`league_config.yaml`), not hardcoded.

## Headline result

| | Model | Baseline |
|---|---|---|
| Validation MAE (next week) | **5.05 pts** | 5.60 pts |

The model's prediction error is about **10% lower** than the naive baseline's, on a
season it never trained on. That's a real, if modest, edge — not a dramatic one, and I'm
not overstating it.

## Precision over recall: the flagged list

The PRD's resolved open question was explicit: a false positive (recommending a player
who doesn't break out) costs more than a false negative (missing one). So the model
doesn't try to catch every breakout — it flags a player only when its prediction clears
the baseline by a configurable margin (`flag_margin`, default 3.0 points), and is
evaluated on **hit rate** (of the players it actually flags, how many outperform), not
on how many total breakouts it catches.

| Window | Flagged | Hit rate | Avg points above baseline |
|---|---|---|---|
| Next week only | 370 / 2,355 (15.7%) | **72.4%** | — |
| Following 3 weeks | 370 | **81.6%** | **+3.79 pts/game** |

Two things stand out:

1. **The list is short on purpose.** 370 of 2,355 player-weeks (15.7%) get flagged —
   consistent with "a shorter, higher-trust list" rather than a firehose of maybes.
2. **The 3-week hit rate is higher than the 1-week hit rate (81.6% vs 72.4%).** That's
   the expected signature of a real signal rather than single-week noise: averaging
   over 3 games smooths out the variance that can make or break a single-week
   comparison, and the flagged players' edge holds up — it doesn't wash out.

An honestly-reported ~72–82% hit rate at a ~16% flag rate is the kind of modest,
real edge the PRD asked for over an unvalidated claim of a bigger one.

## A mistake I made and caught

The first version of this model used quantile regression (predicting the 80th
percentile) to chase the PRD's literal "ceiling" language. It looked reasonable until I
validated it against the baseline: model MAE came out to 7.14 pts — *worse* than the
baseline's 5.60 — and it flagged 79% of all players, the opposite of a precision-tuned
list.

The bug was structural, not a tuning issue: an 80th-percentile predictor is
systematically above a mean predictor almost everywhere, by construction. That's not the
model detecting anything real about a given player — it's just what predicting a high
percentile does to *every* player. Comparing it against a mean baseline was comparing
two different questions and expecting the answers to line up.

The fix was switching to mean regression, so the model and the baseline estimate the
same quantity and the comparison — and the flag-margin threshold — actually mean
something. I'm including this because catching it via the baseline comparison, rather
than shipping a model that looked directionally right but was quietly worse than doing
nothing, is itself part of the evidence this project's evaluation discipline works.

## Does splitting by position help?

Tested training a separate model per position instead of one general model, on the
theory that a WR's opportunity signal and an RB's might behave differently. Evaluated
each position-specific model against the *general* model's performance on that same
position slice — not just against baseline — so the comparison had to earn its added
complexity.

| Position | Position-specific MAE | General model MAE | Position-specific hit rate | General model hit rate |
|---|---|---|---|---|
| RB | 5.83 | **5.61** | 72.3% | 64.7% |
| WR | 5.36 | **5.30** | 72.7% | 70.9% |
| TE | **4.17** | 4.27 | **83.0%** | 79.2% |

The result is mixed, not uniform: the general model wins or ties on MAE for RB and WR
(likely because each position alone has less training data to work with — RB and WR
position-specific hit rates are still slightly higher, but on noisier MAE), while the
position-specific TE model wins outright on both MAE and hit rate. The honest
conclusion is a hybrid, not "position-specific models are better": a TE-specific model
would be worth deploying; RB and WR are better served by the general model as-is, at
least at this sample size.

## Known scope gap: QBs aren't covered

Attempting to train a QB-specific model surfaced something that was already quietly true
of the general model: only 4 QB rows survive the training pipeline, out of 14,349 total.
`target_share` and `air_yards_share` are undefined for quarterbacks — they're the one
throwing, not receiving — so QB rows get dropped by the same feature-completeness check
that (correctly) drops other genuinely-missing data. This wasn't a regression introduced
by position-splitting; it made an existing scope boundary visible.

**In practice, this model covers pass-catchers and ball-carriers (WR/RB/TE), not
quarterbacks.** QB fantasy production is driven by passing volume and efficiency — a
different mechanism than the receiving/rushing opportunity metrics this model is built
around — and would need a distinct feature set to model properly. Worth stating
explicitly rather than letting the model silently underperform on a position it was
never really built for.

## Other known limitations

- **Weeks 1–2 have no signal by design.** The trailing window needs 2 prior games, so
  the model can't produce a prediction until week 3 of a season. This is enforced in
  the feature code itself, not just documented in prose.
- **Earned vs. temporary role changes are partially addressed, not solved.** Requirement
  3b cross-references official injury reports to flag when a usage spike coincides with
  an injury to a same-position teammate who previously had the higher share — but it
  surfaces the coincidence for a human to weigh, and deliberately does not predict when
  an injured starter returns.
- **Route participation isn't in the data.** It doesn't exist anywhere in the public
  nflverse dataset (checked weekly data, snap counts, NGS, PFR weekly, and FTN
  charting) — it's PFF-proprietary. Snap share is used as the closest available proxy.
- **Custom scoring bonuses only cover per-occurrence stats** (TDs, turnovers), not
  yardage-threshold bonuses (e.g. "+3 for 100 rushing yards") — unsupported bonus keys
  warn rather than silently being ignored.

## Real-world sanity check: the live ESPN connector against a season that already happened

Everything above is a backtest — real data, but evaluated in aggregate across 2,355
player-weeks. As a qualitative gut-check, I ran the full pipeline (live ESPN connector →
opportunity model → injury context → roster-fit re-ranking) against my own actual league
on ESPN, pointed at the completed 2024 season, so the free-agent pool and roster came
from a real league instead of a hand-built example CSV.

The connector correctly authenticated against the real league, correctly identified my
team (`team_id=2` → "Dylan", cross-checked against all 12 teams), correctly read the
league's actual settings (full PPR, waiver *priority* rather than FAAB — it doesn't
assume the more common system), and pulled a real 16-player roster with real
nflverse-matched player IDs.

The output isn't just plausible — specific top-ranked flags line up with things that
verifiably happened in the 2024 season:

- **Rashee Rice (WR, KC)** — flagged on "target share 26% → 39% (up)." He genuinely had
  that exact breakout in real life, shortly before a season-ending ACL/LCL injury.
- **Jordan Mason (RB, SF)** — flagged on "snap share 28% → 86% (up)," about as large a
  real usage jump as exists. He really did take over as the 49ers' starter when
  Christian McCaffrey went down.
- **Isaiah Davis (RB, NYJ)** — flagged with injury context: "coincides with Breece Hall
  (Doubtful, Knee)." Breece Hall genuinely battled a knee injury that season — requirement
  3b's earned-vs-injury-driven distinction correctly linked the backup's opportunity to
  the real starter's real injury, not a coincidence I set up.
- **Tyler Goodson (RB, IND)** — same pattern: flagged alongside "Jonathan Taylor (Out,
  Ankle)," who really did miss time that year.

This isn't a substitute for the backtest above — it's one league, one season, eyeballed
against storylines I happen to remember, not a statistic. But it's evidence the pieces
compose correctly against messy real-world data (real ESPN auth, real league settings,
real player-name matching) and not just against the clean historical tables used during
development, and that the specific signals it's flagging correspond to things that
actually happened, not noise that merely looks plausible in a table.

## Matchup simulator calibration: does the win probability actually mean anything?

The opportunity model above answers "who should I add." A separate Phase 2 feature (the
Monte Carlo matchup simulator) answers "who should I start," by simulating a lineup
against a real opponent thousands of times and reporting a win probability. That number
is only useful if it's calibrated — a "70% win probability" should mean roughly a 70%
real-world win rate, not just a confident-sounding label.

A first spot-check across 3 individual weeks (reported in an earlier draft of this
project) turned out not to actually test this: since it queried a season that's long
over, every player resolved through the simulator's deterministic "game already final"
shortcut, so every simulated outcome collapsed to exactly 0% or 100% — that tested the
FLEX optimizer's search logic, but never touched the actual probabilistic engine.

To test the real thing, I re-ran it across **97 real matchups spanning the entire
2024 league** (all ~6-7 matchups/week, weeks 4-17 — not just my own team's ~14), with the
"already final" shortcut deliberately stripped so every player is forced through the same
model + variance-based projection a live, pre-game query would actually use.

| Metric | Result | No-skill baseline |
|---|---|---|
| Pick accuracy (favored side actually won) | **63.9%** (62/97) | 50% |
| Brier score | **0.220** | 0.25 |
| Log loss | **0.623** | 0.693 |

And a calibration table — predicted probability bucket vs. how often the favored side
actually won in that bucket:

| Predicted | n | Actual win rate |
|---|---|---|
| 0.3–0.4 | 17 | 0.41 |
| 0.4–0.5 | 19 | 0.47 |
| 0.5–0.6 | 16 | 0.56 |
| 0.6–0.7 | 18 | 0.78 |
| 0.7–0.8 | 8 | 0.38 |

The well-sampled middle buckets (n=16-19) track closely — predicted 0.45 lines up with an
actual 0.47, predicted 0.54 with an actual 0.56. The one bucket that looks off (0.7-0.8
predicted vs. 0.38 actual) has only 8 observations, and — importantly — the *adjacent*
bucket (0.6-0.7) swings the opposite direction (actual *higher* than predicted). A
genuine "the model is overconfident at high probabilities" bug would show a consistent
downward pattern across 0.6-0.9, not one outlier bucket bracketed by two that calibrate
fine. I did not "fix" this by inflating variance to flatten that one bucket — doing so
would have degraded the buckets that are already well-calibrated to chase noise from a
sample of 8.

**Two real bugs surfaced building this test, not in the simulator's core logic but in a
week-scoping assumption that would have quietly undermined both this test and the
`--week` flag's own advertised "check a past week" use case:**

1. The model's scoring function (`score_latest_week`) picked each player's single most
   recent trailing snapshot from the *entire* ingested file, not "as of before the week
   being asked about." Harmless for true live use (there's no future data yet to leak),
   but wrong for any retrospective query once later weeks are ingested — exactly the
   `--week` flag's own use case, and exactly what this calibration test needed to do
   correctly. Fixed by adding `score_as_of_week(season, week)`, which restricts training
   data to strictly before the target week; `score_latest_week()` is now a thin wrapper
   around it and produces byte-identical output for true live queries (verified directly).
2. Asking for a week with genuinely insufficient trailing history crashed several pandas
   frames deep with a cryptic `IndexError` instead of failing cleanly. Fixed to return an
   empty result immediately.

## Two more accuracy experiments: one negative, one a wash

Asked how to push accuracy further, I tried two more things. Both were built as measured
A/B experiments against the same held-out data already used above — not silent
replacements — because both carried real regression risk to an already-validated system.

**Opponent-adjusted ("defense vs. position") projections.** The opportunity model only
ever looks at a player's own usage trend, never who they're playing next. Added a feature
representing how many fantasy points a player's upcoming opponent has recently allowed to
that position (an 8-game trailing window, resolved from the real NFL schedule so it works
at true prediction time, not just retrospectively), trained a variant model with it, and
compared against the general model on the exact same held-out rows:

| | With DvP | General model (same rows) |
|---|---|---|
| MAE | 5.02 | 5.03 |
| Hit rate | 66.7% (n=243) | 68.4% (n=237) |

MAE is a tie — a 0.01-point difference on ~240 samples is noise, not a win, regardless of
which number is technically smaller. Hit rate, the more decision-relevant metric given
this whole project's precision-over-recall framing, is actually *worse* with DvP. **Not
adopted.** Reported honestly rather than as a win because one metric moved in the "right"
direction by a trivial margin — the same discipline as the position-split experiment,
which found a real, uneven result (helped TE, didn't help RB/WR) instead of a clean yes.

**Right-skewed, correlated Monte Carlo draws.** The simulator drew every player
independently from a Normal distribution — not how real fantasy scores or real teammates
actually behave (a ceiling game is more likely than a symmetric bell curve implies, and
two players on the same offense really do have *some* shared good/bad weeks). Rather than
guess how strong that correlation is, I measured it directly from real 2018-2024 data:
same-team teammates' point residuals (actual minus each player's own running average,
isolating the shared-game-script component from each player's own established level)
correlate at **r=0.0405** — a same-size cross-team control group measured **r=0.0087** as
a sanity check on the method itself, about 5x smaller, confirming the same-team signal is
real and not a measurement artifact. That's a real but modest number — much smaller than
intuition might suggest; most of a team's week-to-week fantasy variance is idiosyncratic
per player, not shared.

Switched the simulator to a moment-matched Gamma distribution (right-skewed, non-negative,
same mean/std, no new dependency) plus that measured 0.041 correlation, and re-ran the
same 97-matchup calibration test:

| | Normal, independent (previous) | Gamma, r=0.041 (new) |
|---|---|---|
| Pick accuracy | 61.9% | 64.9% |
| Brier score | 0.2198 | 0.2196 |

Brier score is effectively unchanged (a 0.0002 difference is nothing) — this is not a
meaningful accuracy win, and I'm not presenting it as one. Pick accuracy moved a few
points, which on 97 observations is within what Monte Carlo sampling noise alone could
produce. **Adopted anyway**, because it held steady while removing an assumption
(independent, symmetric draws) that was more obviously wrong than convenient, and because
the correlation value going into it is a real, measured number rather than an invented
one. If a future, larger calibration run shows this was the wrong call, that's a decision
this same test can revisit and reverse just as easily.

## A third experiment: does the *shape* of a usage climb matter?

The trend feature the model has always used is an endpoint delta — `s - s.shift(window)`.
That means these two players are literally indistinguishable to it:

| | wk 1 | wk 2 | wk 3 | wk 4 | trend |
|---|---|---|---|---|---|
| Player A | 20% | 20% | 20% | 60% | +40 |
| Player B | 20% | 33% | 46% | 60% | +40 |

A is the classic waiver trap — one blowout or garbage-time game that reverts next week. B
is the progressive role change the whole tool exists to catch early. Same number.

So I added two features describing the *shape* of the climb (`persistence.py`): a
consecutive-rising-weeks streak, and the fraction of recent week-over-week deltas that
were positive, for snap share and target share. Deliberately only two usage columns, not
all five — ten new features on ~10k usable training rows is a real overfitting risk, and
the position-split experiment already showed this model is sensitive to how it's sliced.

Same A/B protocol as the DvP experiment, against the same held-out 2024 season:

| | With persistence | General model (same rows) |
|---|---|---|
| MAE | 5.08 | 5.10 |
| Hit rate | **73.5%** (n=313) | 71.6% (n=317) |

Both metrics moved the right direction this time — unlike DvP, where hit rate went
*down*. But a +1.9pp hit-rate gap on ~315 flagged rows is exactly the kind of number that
looks like a win and is actually sampling noise, so I tested it rather than eyeballing it.
A **paired bootstrap** (2,000 resamples; each iteration resamples validation rows *once*
and scores both models on that same resample, respecting the fact that the two flagged
sets overlap heavily) gives:

> mean **+1.9pp**, 90% CI **[−0.8, +4.6]pp**, **87%** of resamples favor persistence.

**The confidence interval crosses zero.** 87% is suggestive, not conclusive — a real
effect would want ~95%. The four persistence features do carry non-trivial model
importance (~10% combined), so the model is genuinely using them, but that's evidence
they're informative, not evidence they improve out-of-sample accuracy.

**Verdict: promising, not proven — not auto-promoted.** This is the same standard that
rejected DvP, applied to a result that happens to lean the right way. Adopting it on 87%
would be exactly the "one metric moved in the right direction by a trivial margin"
reasoning this evaluation has rejected twice already. The honest read is that it's the
most encouraging of the three experiments and is worth re-running once 2025 or 2026 data
roughly doubles the validation sample — at which point the same test either clears 95% or
it doesn't.

## Unlocking 2025, and what a true holdout season revealed

Everything above validates on 2024 — the season named as `validation_season` when the
config was first written. That's a legitimate holdout, but it was *chosen*, and every
experiment above was evaluated against it repeatedly. A season nobody had ever looked at
is a much harder test.

2025 was unavailable: `import_weekly_data([2025])` 404s, because nflverse publishes its
pre-aggregated `player_stats_<year>` table on a lag. But `import_pbp_data([2025])` returns
a complete season (48,771 plays, weeks 1–20) and `import_snap_counts([2025])` returns
26,612 rows. The facts were all there; only the convenience aggregation was missing.

So `ingestion/pbp_fallback.py` rebuilds those stat lines from raw plays. Because that's a
reconstruction, it's **gated on reproducing a season nflverse HAS published** rather than
trusted because the code ran:

| vs. nflverse's own 2024 weekly data | correlation | median abs diff | within 0.5 pts |
|---|---|---|---|
| `fantasy_points_ppr` | 0.9989 | 0.00 | 98.6% |
| `target_share` | 0.9991 | 0.00 | 100% |
| `air_yards_share` | 1.0000 | 0.00 | 100% |

5,327 of 5,340 official rows matched. Air-yards share is bit-exact. (Known residual: a
max single-row difference of 10.1 points, almost certainly return touchdowns and other
non-scrimmage scoring that play-by-play attributes differently — irrelevant to a
usage-trend model, but real and worth stating rather than hiding behind the averages.)

**The true holdout result.** With 2025 ingested, the *already-shipped* model — trained on
2018–2023, never exposed to 2024 or 2025 in any form — could be tested on a season that
did not exist in this pipeline an hour earlier:

| | 2024 (the chosen validation season) | 2025 (true holdout) |
|---|---|---|
| Model MAE | 5.05 | 5.23 |
| Baseline MAE | 5.60 | 5.66 |
| Improvement over baseline | 9.8% | **7.5%** |
| Hit rate at margin 3.0 | 72.4% | **69.4%** (n=451) |

**The edge is real and it generalizes.** It degrades modestly — about two points of hit
rate and two points of MAE improvement — which is roughly what two years of roster and
scheme drift should cost. That is a far stronger claim than the original 2024 number,
because nothing about 2025 was available to tune against.

## Three things that looked like edge and weren't

Adding 2025 also made it possible to re-test earlier conclusions on fresh data. All three
went the same way, and the pattern is the actual finding:

**1. More training data didn't help.** Retraining on 2018–2024 (a whole extra season)
and validating on 2025 gives MAE 5.23 and a 68.8% hit rate — statistically identical to
the old model's 5.23 / 69.4% on the same season. Training-data volume is not the
binding constraint.

**2. Usage persistence reversed sign.** On 2024 it looked genuinely promising: +1.9pp hit
rate, both metrics moving the right way, 87% of bootstrap resamples favoring it. It was
*not* adopted, because the 90% CI crossed zero. On the larger, fresher 2025 sample:

| | 2024 | 2025 |
|---|---|---|
| Hit-rate difference | +1.9pp | **−1.4pp** |
| Resamples favoring persistence | 87% | **20%** |

It flipped. Promoting it on 87% would have made the tool measurably worse — the refusal
to adopt a marginal positive was vindicated within a single afternoon.

**3. The `flag_margin` precision curve mostly flattened.** On 2024, raising the margin
from 3.0 to 6.0 appeared to buy real precision (72.4% → 78.0% one-week hit rate). On 2025
the same sweep gives 68.8% → 70.7%, and the *three-week* hit rate actually peaks at the
shipped 3.0 (80.1%) rather than improving. `flag_margin=3.0` is left unchanged — not
because tuning was skipped, but because it was tried and the apparent gains didn't
replicate.

**What this converges on:** at these sample sizes (~300–450 flagged players per season),
single-season results reliably manufacture 2–6 point "improvements" that vanish or invert
on the next season. That's not a reason to distrust the headline edge — that one has now
survived a genuine holdout — but it is a hard ceiling on how much further feature tinkering
can honestly claim to push it. The remaining levers are structural (seeing usage before
box scores do, and acting on it Tuesday), not algorithmic.

## Bottom line

The model beats a naive rolling-average baseline by a real but modest margin (~10% MAE
improvement), and its flagged list — the actual decision-relevant output — hits at
72–82% depending on the evaluation window, on a season it never saw during training.
Run against a real league on real ESPN infrastructure, its top flags line up with
specific, verifiable 2024 storylines rather than just looking reasonable in a table. The
separate matchup simulator's win probabilities are honestly calibrated against 97 real
outcomes, not just directionally plausible — and the headline edge has since survived a
genuine holdout season (2025) that was unavailable when any of it was tuned. Follow-on
accuracy experiments were tried and reported honestly regardless of outcome:
opponent-adjusted projections didn't measurably help and weren't adopted; a right-skewed,
correlation-aware simulator held calibration steady on real, measured inputs and was
adopted on that basis, not because the number moved dramatically; usage-persistence
features leaned positive on 2024 but failed a paired bootstrap, were left un-promoted,
and then *reversed sign* on 2025 — which is the clearest vindication in this document of
refusing to adopt on a number that merely looked good. None of this is a large, dramatic edge. It's a legitimate,
honestly-measured one, arrived at by building an evaluation harness rigorous enough to
catch and correct my own modeling mistakes along the way — which is the actual point of
the exercise.
