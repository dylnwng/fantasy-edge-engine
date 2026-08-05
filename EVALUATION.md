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

## Bottom line

The model beats a naive rolling-average baseline by a real but modest margin (~10% MAE
improvement), and its flagged list — the actual decision-relevant output — hits at
72–82% depending on the evaluation window, on a season it never saw during training.
Run against a real league on real ESPN infrastructure, its top flags line up with
specific, verifiable 2024 storylines rather than just looking reasonable in a table.
That's not a large, dramatic edge. It's a legitimate, honestly-measured one, arrived at
by building an evaluation harness rigorous enough to catch and correct my own modeling
mistake along the way — which is the actual point of the exercise.
