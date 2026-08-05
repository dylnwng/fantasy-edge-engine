# Edge Engine

**Usage moves before points do.** ESPN's default waiver rankings sort by trailing
fantasy points — a lagging indicator. By the time a role change shows up in the box
score, the cheap waiver window to grab that player is usually already closed. Snap
share, target share, air yards share, and red zone touches move *first*. Edge Engine
watches those instead, and ranks your league's free agents by predicted opportunity
rather than what already happened.

It beats a naive "just look at recent points" baseline by a real, modest, honestly-measured
margin — see [`EVALUATION.md`](EVALUATION.md) for the full writeup, including a mistake
I made and caught along the way.

| | Model | Naive baseline |
|---|---|---|
| Next-week point error (validation MAE) | **5.05** | 5.60 |
| Hit rate on flagged players (next week / following 3 weeks) | **72.4% / 81.6%** | — |

## How it works

```
nflverse (play-by-play, snap counts, injuries)
        │
        ▼
  ingestion pipeline  ──►  per-player, per-week usage table (Parquet)
        │
        ▼
  opportunity model  ──►  XGBoost, trailing 2-game usage trend → next-week points
        │                 trained 2018–2023, validated on held-out 2024
        ▼
  injury context     ──►  flags when a usage spike coincides with an injury to
        │                 the same-position player who had the role before
        ▼
  roster-state source ──► your roster + free agents, from either:
        │                   • manual CSV/YAML, or
        │                   • a live ESPN connector (cookie-authenticated)
        ▼
  ranking + roster-fit ──► free agents ranked by opportunity, adjusted for your
                            league's position scarcity and bye-week collisions
        │
        ▼
  CLI table  /  Streamlit dashboard
```

Every stage is a swappable interface, not a hardcoded path — the model and ranking
layers never know or care whether roster data came from a CSV or a live ESPN league.

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate  # Python 3.11+
bash scripts/install.sh          # nfl_data_py needs a two-step install, see the script
python -m edge_engine.ingestion.pipeline --seasons 2018 2019 2020 2021 2022 2023 2024
python -m edge_engine.model.train
python -m edge_engine.ranking.roster_fit     # CLI table
streamlit run app.py                          # interactive dashboard
```

By default, roster/free-agent data comes from the example files in `data/roster_state/`.
To point it at a real ESPN league instead:

```bash
cp .env.example .env   # fill in your league ID, team ID, and auth cookies — see the
                        # file's comments for exactly how to find each one
```

## Project layout

```
src/edge_engine/
  ingestion/     nflverse pull + normalize → per-player-week usage table
  model/         features, XGBoost training/prediction, injury context, position-split
                 comparison, historical accuracy tracking
  roster/        the roster-state interface + manual CSV and live ESPN implementations
  ranking/       free-agent ranking, usage-trend explanations, roster-fit re-ranking
app.py           Streamlit dashboard
theme.py         its visual design system (dark, DIN Condensed, film-room terminal)
EVALUATION.md    the full written evaluation — methodology, results, and what didn't work
data/roster_state/README.md   the manual input format, if not using the ESPN connector
```

## What this is (and isn't)

Built as a real, working tool for one specific private league, and as a demonstration of
a complete pipeline — ingestion, modeling, evaluation, and a live external API integration
— rather than a general product. It doesn't predict FAAB bid amounts, optimize a full
lineup under variance, or scan opponent rosters for trades — those are deliberately out
of scope for this phase. It also doesn't cover quarterbacks — see `EVALUATION.md` for why
that's a real, stated scope boundary rather than an oversight.

## Tests

```bash
pytest
```

43 tests, no live network calls or credentials required.
