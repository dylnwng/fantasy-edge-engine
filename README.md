# Edge Engine

**Usage moves before points do.** ESPN's default waiver rankings sort by trailing
fantasy points — a lagging indicator. By the time a role change shows up in the box
score, the cheap waiver window to grab that player is usually already closed. Snap
share, target share, air yards share, and red zone touches move *first*. Edge Engine
watches those instead, and ranks your league's free agents by predicted opportunity
rather than what already happened.

It beats a naive "just look at recent points" baseline by a real, modest, honestly-measured
margin — see [`EVALUATION.md`](EVALUATION.md) for the full writeup, including the mistakes
I made and caught along the way.

The numbers below are a snapshot. For the live ones straight out of the trained models,
run `python scripts/show_metrics.py` — docs drift, that file doesn't.

| | Model | Naive baseline |
|---|---|---|
| Next-week point error (2025 holdout MAE) | **5.17** | 5.66 |
| Hit rate on flagged players (next week / following 3 weeks) | **67.6% / 80.3%** | — |

Those are **held-out** numbers on 2025, a season that was unavailable to this project
until the play-by-play reconstruction in `ingestion/pbp_fallback.py` was built and
validated — so nothing was tuned against it. On the originally-chosen 2024 validation
season the model scores better (5.05 MAE, 72.4% hit rate); the gap is roughly what a
year of roster and scheme drift costs, and the smaller number is the honest one.

A second capability answers a different, related question — not "who should I pick up,"
but "who should I start": a Monte Carlo matchup simulator plays out your actual
upcoming matchup against your actual ESPN opponent thousands of times and recommends a
FLEX swap when one meaningfully improves your odds against them specifically.

## How it works

```
nflverse (play-by-play, snap counts, injuries)
        │
        ▼
  ingestion pipeline  ──►  per-player, per-week usage table (Parquet)
        │
        ▼
  opportunity model  ──►  XGBoost, trailing 2-game usage trend → next-week points
        │                 trained 2018–2024, validated on held-out 2025
        ▼
  injury context     ──►  flags when a usage spike coincides with an injury to
        │                 the same-position player who had the role before
        ▼
  roster-state source ──► your roster + free agents, from either:
        │                   • manual CSV/YAML, or
        │                   • a live ESPN connector (cookie-authenticated)
        ▼
  ranking + roster-fit ──► free agents ranked by opportunity, adjusted for your
        │                   league's position scarcity and bye-week collisions
        ▼
  CLI table  /  Streamlit dashboard
```

Every stage is a swappable interface, not a hardcoded path — the model and ranking
layers never know or care whether roster data came from a CSV or a live ESPN league.

The matchup simulator (Phase 2) is a second, separate pipeline:

```
per-player variance  ──►  your lineup vs. your actual current-week ESPN opponent's
(trailing 6-game std       actual lineup, each player's outcome drawn from a distribution
of real points)            (10,000 Monte Carlo simulations)
                                    │
                                    ▼
                        brute-force FLEX optimizer  ──►  win probability +
                                                          recommended swap, if any
```

It's ESPN-only because it needs an opponent's roster, and the manual CSV source has no
concept of one — there's no payoff to hand-maintaining an opponent's roster every week.

## Quickstart

> **Setting this up for a league that isn't mine?** Start with
> [`SETUP.md`](SETUP.md) — it walks through the whole thing end to end, and
> its §4 tells you up front whether your league is even compatible (ESPN
> only, no auction drafts, and the model doesn't cover kickers or defenses).

One-time setup:

```bash
python3 -m venv .venv && source .venv/bin/activate  # Python 3.11+
bash scripts/install.sh          # nfl_data_py needs a two-step install, see the script
python -m edge_engine.ingestion.pipeline --seasons 2018 2019 2020 2021 2022 2023 2024 2025
python -m edge_engine.model.train        # WR / RB / TE
python -m edge_engine.model.train_qb     # QB (separate feature space)
```

`train` fits a static model artifact — it doesn't need to be re-run weekly. Only
re-run it if you intentionally want to retrain (e.g. once a lot more of a season's
data exists), not as part of your regular routine below.

By default, roster/free-agent data comes from the example files in `data/roster_state/`.
To point it at a real ESPN league instead:

```bash
cp .env.example .env   # fill in your league ID, team ID, and auth cookies — see the
                        # file's comments for exactly how to find each one
```

Every week:

```bash
python -m edge_engine.weekly     # refreshes this season's data, prints rankings +
                                  # your live matchup (if EDGE_ENGINE_ROSTER_SOURCE=espn)
streamlit run app.py             # interactive dashboard, same data
```

`weekly` exists because re-running `ingestion.pipeline` without `--force-refresh`
silently serves whatever's already cached — it never hits the network again, with no
warning that it didn't. `weekly` force-refreshes only the current season (historical
training seasons don't change, so it never re-pulls those) and turns an expired ESPN
session or a not-yet-published nflverse season into a clear message instead of a raw
traceback. The individual pieces are still available directly if you want just one of
them:

```bash
python -m edge_engine.draft                             # draft board, live ADP (Phase 5-lite)
python -m edge_engine.draft --live                      # live pick tracker + tier cliffs
python -m edge_engine.draft --serve [--espn]            # draft-night browser screen
python -m edge_engine.trade --out "A" --in "B"          # trade surfacer (no verdict)
python -m edge_engine.insights                          # what's wrong with my roster (Phase 3)
python -m edge_engine.insights --week 12 --all          # a past week, neutral players listed
python -m edge_engine.ranking.roster_fit                # free-agent rankings only
python -m edge_engine.ranking.roster_fit --top 20        # top 20 regardless of tier
python -m edge_engine.ranking.roster_fit --all           # every candidate, no truncation
python -m edge_engine.simulation.matchup_cli             # this week's actual matchup
python -m edge_engine.simulation.matchup_cli --week 15   # or any past week, e.g. for a post-mortem
```

Rankings show **High/Medium-confidence candidates only** by default. A real live run
produced 293 ranked free agents — 5 High, 38 Medium, 249 Low — so printing everything
means 85% of the weekly output is the tool's own "$1 speculative claim" tier scrolling
past the handful of names actually worth a bid. `--all` (on either `weekly` or
`roster_fit`) gets the full list back.

## Project layout

```
src/edge_engine/
  ingestion/     nflverse pull + normalize → per-player-week usage table
  model/         TWO models: WR/RB/TE on receiving usage (features/train/predict) and
                 QB on passing volume (qb_features/train_qb/predict_qb), unioned at
                 the prediction layer. Plus injury context and per-player variance.
  roster/        the roster-state interface + manual CSV and live ESPN implementations,
                 plus the ESPN-only matchup/opponent-data protocol
  ranking/       free-agent ranking, usage-trend explanations, roster-fit re-ranking
  insights/      Phase 3 roster diagnosis — usage-vs-production divergence, bye-adjusted
                 worst-week scarcity, bye/stacking/injury exposure
  draft/         Phase 5-lite draft board — ADP pricing behind a MarketPriceSource
                 Protocol, tier cliffs, positional-run detection, live pick tracking,
                 ESPN draft polling, and a zero-dependency draft-night browser screen
  trade/         Phase 4 trade surfacer — divergence + observed rates, no verdict
  experiments/   built, measured, REJECTED. Nothing imports these; kept so the next
                 person doesn't rebuild a known dead end. See its __init__.py
  simulation/    Monte Carlo matchup simulation + brute-force FLEX optimizer (ESPN-only)
  weekly.py      single weekly entry point — refresh current-season data, run rankings
                 and (if live) the matchup simulator, with clear errors instead of tracebacks
app.py           Streamlit dashboard
theme.py         its visual design system (dark, DIN Condensed, film-room terminal)
model_config.yaml        opportunity model settings (seasons, thresholds)
simulation_config.yaml   matchup simulator settings (sim count, variance window, swap threshold)
EVALUATION.md    the full written evaluation — methodology, results, and what didn't work
data/roster_state/README.md   the manual input format, if not using the ESPN connector
```

## What this is (and isn't)

Built as a real, working tool for one specific private league, and as a demonstration of
a complete pipeline — ingestion, modeling, evaluation, and a live external API integration
— rather than a general product. It doesn't predict FAAB bid amounts or scan opponent
rosters for trades — deliberately out of scope. Lineup optimization under uncertainty is
scoped narrowly on purpose too: it only considers your FLEX slot(s) against your actual
opponent, via straightforward brute-force enumeration rather than a general-purpose
solver — the real decision space (a handful of bench players) doesn't need one, and
adding one would be complexity without a corresponding benefit. It also doesn't cover
kickers or defenses — nflverse has essentially no player-level usage data for those
positions, so there is nothing to model. Quarterbacks *are* covered, by a separate
volume-based model (`model/qb_features.py`); `EVALUATION.md` documents how a scope
boundary that had been assumed for most of this project turned out to be wrong when
it was finally measured.

## Tests

```bash
pytest
```

249 tests, no live network calls or credentials required.
