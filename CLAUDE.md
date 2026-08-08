# Edge Engine — orientation for a new agent

Fantasy football waiver-wire tool for one specific private ESPN league. Python 3.11+,
161 tests, no network calls or credentials required to run the suite.

```bash
source .venv/bin/activate && python -m pytest tests/ -q
```

## The thesis

Rank free agents by **predicted opportunity**, not trailing fantasy points. Snap share,
target share, air-yards share and red-zone touches move *before* points do, so watching
usage catches a role change 1–2 weeks before the box score makes it obvious to the rest
of the league. That timing gap is the entire product.

Two separate capabilities:

1. **Waiver predictor (Phase 1)** — "who should I pick up"
2. **Monte Carlo matchup simulator (Phase 2)** — "who should I start" (ESPN-only; it
   needs an opponent's roster, and the manual source has no concept of one)

## Architecture

```
nflverse (play-by-play, snap counts, injuries, schedules)
   └─► ingestion/   → per-player-week usage table (Parquet, upserted on
        │              (season, week, player_id))
        └─► model/  → XGBoost: trailing 2-game usage trend → next-week points
             └─► ranking/  → free agents ranked, explained, roster-fit re-ranked
                  └─► CLI table / Streamlit dashboard
```

Roster data goes through the `RosterStateSource` Protocol (`roster/interface.py`) with two
implementations — manual CSV/YAML and a live cookie-authenticated ESPN connector. Nothing
downstream knows which is active; `EDGE_ENGINE_ROSTER_SOURCE=manual|espn` is the whole
switch. Keep it that way.

## Entry points

```bash
python -m edge_engine.weekly                              # the weekly command
python -m edge_engine.insights [--week N] [--all]         # roster diagnosis (Phase 3)
python -m edge_engine.ranking.roster_fit [--top N|--all]  # rankings only
python -m edge_engine.simulation.matchup_cli [--week N]   # matchup + FLEX optimizer
python -m edge_engine.model.train                         # occasional, NOT weekly
streamlit run app.py
```

`train` writes a static model artifact; `predict` only needs fresh *features*, so
retraining is not part of the weekly loop.

`weekly.py` exists because re-running the ingestion pipeline without `--force-refresh`
silently serves cache forever, with no warning. It force-refreshes only the current
season and turns expired ESPN cookies / unpublished nflverse seasons into clear messages
instead of tracebacks.

## Measured performance

| | Model | Naive baseline |
|---|---|---|
| 2025 **true holdout** MAE | 5.23 | 5.66 (7.5% better) |
| Hit rate on flagged players, 1wk / 3wk | 68.8% / 80.1% | — |

Matchup simulator: 63.9% pick accuracy, 0.220 Brier over 97 real 2024 matchups
(no-skill baseline: 50% / 0.25).

The edge is real, modest, and honestly measured. Don't inflate these numbers in docs.

## Non-obvious things that will bite you

**2025 data is reconstructed, not fetched.** `import_weekly_data([2025])` 404s — nflverse
publishes its aggregated `player_stats_<year>` table on a lag, while the play-by-play it's
derived from is already complete. `ingestion/pbp_fallback.py` rebuilds the stat lines from
raw plays, gated on reproducing 2024 where both sources exist (0.9989 correlation on
fantasy points, exactly 1.0000 on air-yards share). **Call
`fetch_weekly_data_or_reconstruct()`, not `fetch_weekly_data()`.**

**This project rejects marginal improvements on principle, and that has been empirically
vindicated.** Four accuracy experiments run, three rejected:

- Opponent-adjusted (DvP) projections — hit rate got *worse*. Rejected.
- Gamma distribution + measured team correlation (r=0.0405, measured not guessed) — a
  wash; adopted only because it removes an assumption that was obviously wrong.
- **Usage persistence** — looked good on 2024 (+1.9pp hit rate, 87% of bootstrap
  resamples favoring). Not adopted, because the 90% CI crossed zero. It then **reversed
  sign on 2025** (−1.4pp, 20% favoring). Do not resurrect without a far larger sample.
- `flag_margin` sweep — the 2024 precision curve flattened on 2025. Left at 3.0.

The convergent finding: at ~300–450 flagged players per season, single-season results
reliably manufacture 2–6pp "improvements" that vanish or invert on the next season.
**Treat any feature that only looks good on one season as noise until proven otherwise.**

**Design invariants that are load-bearing:**

- **Context is surfaced, never baked into the score.** Injury context, non-injury roster
  status (suspended/PUP/reserve), blocker trajectory and bye-week collisions all appear in
  explanation text. They never silently move `predicted_score`.
- **Never fabricate from missing data.** `bye_weeks.py` refuses to infer a bye from an
  incomplete schedule; blocker-trajectory refuses to infer recovery from *absence* from
  the injury report; K/D-ST get `std=0` rather than an invented spread.
- **`(player_id, season, week)` uniqueness is enforced with loud errors** in
  `build_features`, `compute_trailing_points_std`, and `persistence`. Duplicate rows
  silently corrupt every rolling window (one repro produced a fabricated std of 699.3).
- **Config values are validated at load.** An out-of-range `team_correlation` used to
  produce silent NaN through every simulated score while still reporting an exact,
  fabricated "0.0% win probability."

## Phase 3 (`insights/`) — one thing to know

`divergence.py` defines its own usage composite (mean of per-metric z-scores against the
positional pool). The Phase 3–5 PRD says to "reuse the existing `opportunity_score`
composite" — **there is no such composite**; that instruction is wrong, and the
correction is written up in `docs/PRD-phase-3-5-corrections.md`. The composite here is
descriptive, for divergence only. **Never feed it back into the opportunity model as a
feature** — it's computed from the same window it describes, so that would be a leak.

## Explicit scope boundaries (PRD non-goals — don't "helpfully" add these)

No FAAB dollar amounts (confidence tiers only) · no trade scanner or opponent-roster
mining · no non-ESPN platforms · no QBs in the opportunity model · no news scraping or
sentiment analysis · no start/sit logic inside roster-fit (that's Phase 2's job).

## Read before making changes

`EVALUATION.md` — full methodology, every experiment including the failures and a
modeling mistake that was caught and reversed mid-project. It is the source of truth for
what has already been tried and why it was or wasn't adopted.
