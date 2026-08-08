# Corrections to `PRD-phase-3-5.md`

Verified against the codebase at commit `c04f34c`. The document's structure, sequencing
and kill gates are sound — these are factual corrections to claims about existing code,
plus one budget line I think should be cut.

---

## 1. §1.4 — `opportunity_score` does not exist

> "Where `opportunity_score` is the existing composite the ranking module already builds
> (snap share, target share, air-yards share, red-zone touches). **Reuse it. Do not define
> a second one.**"

There is no such composite. `grep -rn "opportunity_score" src/` returns nothing. What
exists is:

- `predicted_score` — the raw XGBoost output, already in fantasy-points units
- the individual usage columns on `player_week` (`snap_pct`, `target_share`,
  `air_yards_share`, `red_zone_touches`, `red_zone_target_share`)

This matters because the instruction is self-defeating as written: an implementer told
"reuse the composite, don't define a second one" will find no composite and define one
anyway, silently, with no stated definition — the exact outcome the sentence was trying
to prevent.

`predicted_score` is also the wrong thing to substitute. It's trained to predict *points*,
so `usage_z − points_z` computed from it measures model-vs-actual error, not
usage-vs-production divergence. Those are different quantities and only the second is
what §1.4 describes.

**Resolution taken in the implementation:** a new, explicitly-named descriptive composite
in `insights/divergence.py`, defined as the mean of per-metric z-scores against the
positional pool over the same window. It is documented as *not* the model's opportunity
score and is used only for divergence. §1.4 should be rewritten to specify this rather
than pointing at something that isn't there.

## 2. §0.2 / §2.3 — the QB exclusion is emergent, not deliberate, and the budget is wrong

> "**QBs are excluded from the opportunity model.** ... This is a prerequisite, not a
> nice-to-have" · §2.3 budgets the QB sub-model at 0.8 Phase-1-units.

The conclusion is right; the mechanism is not, and the mechanism is what matters.

QBs aren't filtered out anywhere. `target_share` and `air_yards_share` are structurally
`NaN` for quarterbacks (they throw, they aren't targeted), and `features.py` drops rows
with null features. The exclusion is a *side effect* of that null-drop. Only 4 QB rows out
of 14,349 survive the pipeline.

**Why the distinction is load-bearing:** an emergent boundary can be reversed by accident,
and it was — during the 2025 play-by-play backfill, one commit before this review. The
reconstruction computed `target_share = targets / team_targets`, which for a QB is
`0 / positive = 0.0` — a valid number where nflverse emits `NaN`. 659 QB rows silently
survived the null-drop and were scored on features carrying no signal for them, inflating
the validation set from 2,413 to 3,196 rows and contaminating every metric measured from
it. Nothing crashed, no test failed, and the reconstruction still validated at 0.999
correlation, because the validator compared only rows where both sources were non-null —
precisely the rows where the bug wasn't. Fixed in `c04f34c` and documented in
`EVALUATION.md`.

**Recommendation: cut the QB sub-model from the critical path.** At 0.8 units it is nearly
a second Phase 1, and §0.2 asserts it is a prerequisite without testing that claim. Phase 3
divergence works on WR/RB/TE, which is what the model actually scores (a live run returns
181 WR / 99 TE / 88 RB). Defer QBs until Phase 4 or 5 demonstrably needs them, and treat
the estimate as provisional until someone has scoped the feature set.

## 3. §2.4 — the validation protocol is stale

> "Train on 2022-2023, validate on 2024, **hold out 2025 entirely** until the model is
> frozen."

Not available any more. The shipped `model_config.yaml` now trains on 2018–2024 and
validates on 2025. There is no untouched season left in the dataset.

**Suggested replacement:** hold out 2025 *for the ROS model specifically* by training it on
≤2024 only, and state plainly that the weekly model has already seen 2025 so the two
models' holdout claims are not interchangeable. The §2.5 kill gate ("CI excludes zero on
2024 AND sign replicates on 2025") still works under that framing — it just needs to say
which model each season is clean for.

## 4. §5.1 — add a null-semantics invariant

The existing invariant "use `fetch_weekly_data_or_reconstruct()`, never
`fetch_weekly_data`" is necessary but not sufficient, per §2 above.

**Proposed addition:** *when reconstructing an external data format, verify null semantics,
not just numeric agreement. A reconstruction can correlate at 0.999 on overlapping rows
while disagreeing about what "missing" means, and downstream null-drops may be
load-bearing.*

## 5. Smaller notes

- **§1.9 appears twice** (acceptance criteria and kill gate share a number); §3.7 likewise.
- **§1.3 lists `deadweight.py`** but §1.2's question 4 is the only place it's specified, and
  §1.7's output contract has no field for it. Either give it a contract field or fold it
  into divergence as a low-usage flag.
- **§1.5's `league_median_starter(pos)`** needs a definition for the manual roster source,
  which has no league-wide rostered pool — only your roster and the free-agent list.
  Implementation falls back to the position's league-wide startable distribution and
  records the substitution in `data_gaps`.
- **Phase 3's 0.25 estimate** is optimistic given §1 above, but it remains the right first
  build.
