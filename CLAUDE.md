# Edge Engine — orientation for a new agent

Fantasy football waiver-wire tool for one specific private ESPN league. Python 3.12+,
291 tests, no network calls or credentials required to run the suite.

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
        └─► model/  → TWO models, unioned at the prediction layer:
        │              WR/RB/TE on receiving usage, QB on passing volume
             └─► ranking/  → free agents ranked, explained, roster-fit re-ranked
                  ├─► insights/   → roster diagnosis (Phase 3)
                  ├─► trade/      → trade surfacer (Phase 4, no projection)
                  ├─► draft/      → ADP board + draft-night screen (Phase 5-lite)
                  └─► CLI + Streamlit dashboard (4 tabs)
```

**`experiments/` is dead code on purpose.** Six modules that were built, measured and
rejected. Nothing imports them. They're kept so nobody rebuilds a known dead end — read
`experiments/__init__.py` before touching anything in there.

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
python -m edge_engine.model.train_qb                      # the separate QB model
streamlit run app.py
```

`train` writes a static model artifact; `predict` only needs fresh *features*, so
retraining is not part of the weekly loop.

`weekly.py` exists because re-running the ingestion pipeline without `--force-refresh`
silently serves cache forever, with no warning. It force-refreshes only the current
season and turns expired ESPN cookies / unpublished nflverse seasons into clear messages
instead of tracebacks.

## Measured performance

**Don't trust a number written in a doc — including this one. Run:**

```bash
python scripts/show_metrics.py
```

It reads `models/training_metrics*.json`, which the training scripts rewrite on every
run, so it can't go stale. Hand-copied figures already drifted once: this file spent a
while quoting a MAE that a retrain had superseded.

As of the last run: main model 5.17 MAE vs a 5.66 baseline (8.7% better), 67.6% hit
rate; QB model 6.57 vs 7.36 (10.8% better), 74.0% hit rate. Matchup simulator: 63.9%
pick accuracy, 0.220 Brier over 97 real 2024 matchups (no-skill: 50% / 0.25).

The edge is real, modest, and honestly measured. Don't inflate these numbers.

## Non-obvious things that will bite you

**2025 data is reconstructed, not fetched.** `import_weekly_data([2025])` 404s — nflverse
publishes its aggregated `player_stats_<year>` table on a lag, while the play-by-play it's
derived from is already complete. `ingestion/pbp_fallback.py` rebuilds the stat lines from
raw plays, gated on reproducing 2024 where both sources exist (0.9989 correlation on
fantasy points, exactly 1.0000 on air-yards share). **Call
`fetch_weekly_data_or_reconstruct()`, not `fetch_weekly_data()`.**

**QBs use a SECOND model.** `model/qb_features.py` + `train_qb.py` + `predict_qb.py`.
Its features are volume (attempts, pass share, rush attempts), not receiving usage, and
its rows are unioned into `score_as_of_week`'s output with an identical column contract
— downstream code is unaware two models exist. Snap share is deliberately excluded (it's
~binary for QBs). If `models/qb_model.json` is absent this is a silent no-op. QBs carry
their own `usage_explanation`, because `usage_trend_explanation` reads receiving metrics
that are null for a passer.

**This project rejects marginal improvements on principle, and that has been empirically
vindicated.** Five accuracy experiments run, three rejected, one un-promoted, one shipped:

- Opponent-adjusted (DvP) projections — hit rate got *worse*. Rejected.
- Gamma distribution + measured team correlation (r=0.0405, measured not guessed) — a
  wash; adopted only because it removes an assumption that was obviously wrong.
- **Usage persistence** — looked good on 2024 (+1.9pp hit rate, 87% of bootstrap
  resamples favoring). Not adopted, because the 90% CI crossed zero. It then **reversed
  sign on 2025** (−1.4pp, 20% favoring). Do not resurrect without a far larger sample.
- `flag_margin` sweep — the 2024 precision curve flattened on 2025. Left at 3.0.
- **QB volume features — SHIPPED.** The only experiment to replicate: +0.73 [+0.42,
  +1.04] on 2024 and +0.89 [+0.57, +1.23] on 2025, 100% of resamples both times. Note
  the lesson: this scope boundary had been *assumed* for the whole project on an
  argument that sounded mechanical and was never measured.

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

## Phase 5-lite (`draft/`) — what it does and doesn't claim

**The board prices everything at ADP on purpose and claims no edge over the market.**
Its value is bookkeeping under a 90-second clock (tier cliffs, positional runs, a
shrinking pool), which is why it ships with no kill gate. The one narrow edge claim is
the divergence tag: a player whose usage spiked in weeks 13–17 is priced at an ADP
reflecting his *full-season* average. That's a tag, never a re-ranking.

Do **not** add custom preseason projections without the §4.3 gate (beat ADP on a
held-out season). A failed model must degrade the tool to an ADP board, not ship
projections styled identically to validated ones.

`data/draft/adp.example.csv` has **synthetic** ADP (real names, ordered by last
season's points). It exists so the board runs out of the box; replace it with a real
export before trusting anything. Live ESPN draft sync (§4.4) is unbuilt — it needs the
`mDraftDetail` spike against a mock draft first.

## Phase 4 (`trade/`) — the ROS model failed its gate

`trade/ros.py` is a **rejected experiment**. It beats replacement level massively and
beats season-to-date PPG essentially not at all (every 2024 bootstrap CI crosses zero; a
dead tie at week 12; rank correlation *worse* than the baseline at week 4). Mechanically:
the shrinkage is the model, so it helps while a sample is thin and adds noise once that
sample is the best estimate.

Per PRD §2.5/§3.7 that means **Phase 4 ships as a pure surfacer** — `trade/compare.py`
shows divergence and observed rates, with no ROS number, no verdict, no fairness score.
Re-run `scripts/validate_ros.py` before ever wiring `ros.py` into anything; it is kept
only so nobody rebuilds it from scratch. **Phase 5-full (custom draft projections) is
not built** for the same reason.

## Explicit scope boundaries (PRD non-goals — don't "helpfully" add these)

No FAAB dollar amounts (confidence tiers only) · no trade scanner or opponent-roster
mining · no non-ESPN platforms · no kickers or defenses (no usage data exists) · no news scraping or
sentiment analysis · no start/sit logic inside roster-fit (that's Phase 2's job).

## Read before making changes

`EVALUATION.md` — full methodology, every experiment including the failures and a
modeling mistake that was caught and reversed mid-project. It is the source of truth for
what has already been tried and why it was or wasn't adopted.
