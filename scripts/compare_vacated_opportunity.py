"""Does vacated opportunity belong in the model, or only in the explanation?

The boundary being tested
-------------------------
CLAUDE.md holds an invariant: "Context is surfaced, never baked into the
score." Injury context is computed today and shown beside a candidate as
text; it never moves `predicted_score`.

That rule is right for context a human should weigh. It is a separate
claim that the underlying fact carries no predictive signal. A backup's
snap share rises when the starter ahead of him is hurt -- about as direct
a leading indicator as football offers -- and it is excluded from the
model by design rather than by measurement.

The QB model is the precedent. That scope boundary had been assumed for
the entire project on an argument that sounded mechanical and was never
tested; when it finally was, it replicated on two seasons and shipped.
This script is the same move applied to the next assumed boundary.

What adopting it would mean
---------------------------
Not a refactor. Amending a documented, load-bearing invariant, which
should happen deliberately and in the open if the evidence supports it --
and not at all if it doesn't. Note the invariant would survive in spirit
for everything else: bye collisions, roster status and blocker trajectory
would remain explanation-only. Only this one measured fact would move.

The comparison
--------------
    candidate A: the shipped features
    candidate B: the shipped features + vacated_snap_share, blocker_severity

Same estimator, same hyperparameters, same rows, same target, same folds.
The two new columns are attached DIRECT rather than trailed: the signal is
a step change ("his blocker just got ruled out"), and running it through
another trailing window would smear the transition worth detecting. They
are computed from a lookback window already.

No leakage: both features use injury reports from weeks <= the feature
week and trailing snap shares from weeks strictly before it -- the same
windows injury_context already uses, so nothing here needs information
that wouldn't exist when the tool runs on a Tuesday.

The verdict
-----------
FOLD AGREEMENT. An effect that shows up pooled and in one season is the
signature of noise; three rejections in this project rest on that.

Watch the hit-rate columns as carefully as MAE. The opponent-adjustment
experiment was caught precisely because it moved MAE and hit rate in
opposite directions, and a feature that fires only for injured-backup
situations could plausibly improve average error while making the short,
high-trust flagged list worse -- which is the list anyone actually acts on.

This changes nothing.

Usage:
    python scripts/compare_vacated_opportunity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_engine.model.config import load_model_config  # noqa: E402
from edge_engine.model.features import feature_columns  # noqa: E402
from edge_engine.model.injury_context import load_injury_reports  # noqa: E402
from edge_engine.model.train import build_estimator, build_training_table_with_merged  # noqa: E402
from edge_engine.model.vacated_opportunity import (  # noqa: E402
    VACATED_COLUMNS,
    attach_vacated_opportunity,
    build_vacated_opportunity,
)
from edge_engine.model.walk_forward import run_walk_forward  # noqa: E402


def _fit_on(feat_cols: list[str]):
    def _fp(train_df, val_df, _ignored):
        model = build_estimator()
        model.fit(train_df[feat_cols], train_df["label_next_week_points"])
        return model.predict(val_df[feat_cols])

    return _fp


def _print_comparison(report_a: dict, report_b: dict, n_fired: int, n_rows: int) -> None:
    header = f"{'Season':<8}{'shipped':>12}{'+vacated':>12}{'Diff':>10}{'Hit A':>9}{'Hit B':>9}"
    print("\nMAE on next-week points (lower is better)\n")
    print(header)
    print("-" * len(header))

    b_better = hit_b_better = hit_compared = 0
    for fa, fb in zip(report_a["folds"], report_b["folds"]):
        diff = fa["model_mae"] - fb["model_mae"]
        b_better += diff > 0
        ha = f"{fa['hit_rate']:.1%}" if fa["hit_rate"] is not None else "n/a"
        hb = f"{fb['hit_rate']:.1%}" if fb["hit_rate"] is not None else "n/a"
        if fa["hit_rate"] is not None and fb["hit_rate"] is not None:
            hit_compared += 1
            hit_b_better += fb["hit_rate"] > fa["hit_rate"]
        print(f"{fa['validation_season']:<8}{fa['model_mae']:>12.3f}{fb['model_mae']:>12.3f}"
              f"{diff:>+10.3f}{ha:>9}{hb:>9}")
    print("-" * len(header))
    print(f"{'Pooled':<8}{report_a['pooled_model_mae']:>12.3f}{report_b['pooled_model_mae']:>12.3f}"
          f"{report_a['pooled_model_mae'] - report_b['pooled_model_mae']:>+10.3f}")

    n = len(report_a["folds"])
    print(f"\nFold agreement (MAE):      vacated wins {b_better}/{n} seasons   <- the test to believe")
    print(f"Fold agreement (hit rate): vacated wins {hit_b_better}/{hit_compared} seasons")
    print(f"\nThe feature fires on {n_fired:,} of {n_rows:,} rows ({n_fired / n_rows:.1%}) -- "
          "it is zero for everyone with no injured blocker ahead of them,")
    print("so a small pooled MAE move can still be a large effect where it applies.")

    print()
    if b_better > n / 2 and hit_b_better >= hit_compared / 2:
        print("Replicates on both MAE and hit rate. That is the strongest case available")
        print("short of a second independent measurement, and it would justify opening the")
        print("question of amending the 'context is never baked into the score' invariant.")
    elif b_better > n / 2:
        print("MAE improves but the hit rate does NOT follow. That is the exact pattern that")
        print("sank opponent adjustment: the flagged list is what people act on, so a better")
        print("average with a worse short list is a downgrade. Do not adopt on MAE alone.")
    else:
        print("Does NOT replicate across seasons. On this project's own standard that is a")
        print("rejection, and the existing invariant stands -- injury context stays text.")
    print()


def main() -> None:
    config = load_model_config()
    seasons = sorted(set(config.train_seasons) | {config.validation_season})

    features, merged = build_training_table_with_merged(config)

    print(f"Loading injury reports for {seasons[0]}-{seasons[-1]}...")
    injuries = load_injury_reports(seasons)

    vacated = build_vacated_opportunity(
        merged, injuries,
        usage_gap=config.injury_ahead_usage_gap,
        lookback=config.injury_lookback_weeks,
    )
    table = attach_vacated_opportunity(features, vacated)

    cols_a = feature_columns(config.trailing_window)
    cols_b = cols_a + list(VACATED_COLUMNS)

    usable = table.dropna(subset=[*cols_b, "label_next_week_points"])
    n_fired = int((usable["vacated_snap_share"] > 0).sum())
    print(f"{len(usable):,} rows usable by both candidates "
          f"({len(cols_a)} features vs {len(cols_b)})")

    shared = dict(table=usable, flag_margin=config.flag_margin)
    report_a = run_walk_forward(**shared, feat_cols=cols_a, fit_predict=_fit_on(cols_a))
    report_b = run_walk_forward(**shared, feat_cols=cols_b, fit_predict=_fit_on(cols_b))

    _print_comparison(report_a, report_b, n_fired, len(usable))
    print("Nothing was changed. Adopting these features means amending a documented")
    print("CLAUDE.md invariant deliberately, and wiring them into features.py.\n")


if __name__ == "__main__":
    main()
