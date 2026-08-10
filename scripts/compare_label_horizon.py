"""Does training on a 3-week target beat training on next week?

The hypothesis
--------------
The shipped model predicts NEXT WEEK's points, but a waiver claim is a
multi-week commitment -- train.py already reports hit rate at both
horizons because three weeks is closer to the decision being made. If the
label were aligned with the decision, the model might rank candidates
better for the thing you actually do with the ranking.

The trap this design avoids
---------------------------
The obvious version of this experiment is fatally broken: train one model
on the 1-week label and one on the 3-week label, then compare each one's
MAE against its own label. That comparison always favours the 3-week
model, and not because it is better. A 3-game AVERAGE is mechanically
smoother than a single game -- less variance in the target means lower
MAE for any predictor at all. You would measure the smoothing and call it
an improvement.

So both candidates are scored against the SAME target, on the SAME rows:

    candidate A: trained on label_next_week_points
    candidate B: trained on label_forward_points (3-game mean)
    both evaluated against label_forward_points, the decision horizon

Only the training target differs. Rows are restricted to those where both
labels exist, so neither candidate gets an easier or larger validation
set than the other.

The verdict
-----------
Run under walk_forward's protocol, so the number that matters is FOLD
AGREEMENT -- how many seasons independently favour B over A. Per
EVALUATION.md, a result that shows up pooled and in one season is the
signature of noise, not of an effect. This project has rejected three
accuracy ideas on exactly that basis.

This changes nothing. It prints a verdict; adopting the 3-week label
would be a separate, deliberate change to train.py and model_config.yaml.

Usage:
    python scripts/compare_label_horizon.py
    python scripts/compare_label_horizon.py --horizon 4
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_engine.model.config import load_model_config  # noqa: E402
from edge_engine.model.features import feature_columns  # noqa: E402
from edge_engine.model.labels import attach_forward_label  # noqa: E402
from edge_engine.model.scoring import compute_points_for_seasons  # noqa: E402
from edge_engine.model.train import build_estimator, build_training_table_with_merged  # noqa: E402
from edge_engine.model.walk_forward import run_walk_forward  # noqa: E402
from edge_engine.roster.interface import get_default_source  # noqa: E402

EVAL_LABEL = "label_forward_points"


def _fit_on(label_col: str):
    """A fit_predict that trains the shipped estimator on `label_col`.

    Only the training target varies between the two candidates -- same
    estimator, same hyperparameters, same features -- so any difference
    is attributable to the label and not to a second thing that changed
    at the same time.
    """

    def _fp(train_df, val_df, feat_cols):
        model = build_estimator()
        model.fit(train_df[feat_cols], train_df[label_col])
        return model.predict(val_df[feat_cols])

    return _fp


def _print_side_by_side(name_a: str, report_a: dict, name_b: str, report_b: dict) -> None:
    header = f"{'Season':<8}{name_a:>14}{name_b:>14}{'Diff':>10}"
    print(f"\nMAE against the {EVAL_LABEL} target (lower is better)\n")
    print(header)
    print("-" * len(header))

    folds_b_better = 0
    for fa, fb in zip(report_a["folds"], report_b["folds"]):
        diff = fa["model_mae"] - fb["model_mae"]  # positive => B better
        if diff > 0:
            folds_b_better += 1
        print(f"{fa['validation_season']:<8}{fa['model_mae']:>14.3f}"
              f"{fb['model_mae']:>14.3f}{diff:>+10.3f}")
    print("-" * len(header))
    print(f"{'Pooled':<8}{report_a['pooled_model_mae']:>14.3f}"
          f"{report_b['pooled_model_mae']:>14.3f}"
          f"{report_a['pooled_model_mae'] - report_b['pooled_model_mae']:>+10.3f}")

    n = len(report_a["folds"])
    print(f"\nFold agreement: the 3-week label wins in {folds_b_better}/{n} seasons"
          "   <- the test to believe")

    print("\nHit rate on flagged players (over the same 3-week horizon):")
    for fa, fb in zip(report_a["folds"], report_b["folds"]):
        ha = f"{fa['hit_rate']:.1%}" if fa["hit_rate"] is not None else "n/a"
        hb = f"{fb['hit_rate']:.1%}" if fb["hit_rate"] is not None else "n/a"
        print(f"  {fa['validation_season']}: {name_a} {ha:>7}  |  {name_b} {hb:>7}"
              f"   (n={fa['n_flagged']} vs {fb['n_flagged']})")

    print()
    if folds_b_better > n / 2:
        print("The 3-week label wins a majority of seasons. That is worth a closer look,")
        print("but check the per-season spread above before adopting -- a win driven by")
        print("one large season is the pattern EVALUATION.md warns about.")
    else:
        print("The 3-week label does NOT replicate across seasons. On this project's own")
        print("standard that is a rejection, not a marginal result to tune further.")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=3,
                        help="how many following games the alternative label averages (default 3)")
    args = parser.parse_args()

    config = load_model_config()
    features, merged = build_training_table_with_merged(config)

    seasons = sorted(set(config.train_seasons) | {config.validation_season})
    scoring = get_default_source().get_league_config().scoring
    points_table = compute_points_for_seasons(seasons, scoring)

    table = attach_forward_label(features, points_table, horizon=args.horizon)
    feat_cols = feature_columns(config.trailing_window)

    # Both labels required, so neither candidate is evaluated on rows the
    # other never saw. This is what keeps the comparison honest.
    usable = table.dropna(subset=[*feat_cols, "label_next_week_points", EVAL_LABEL]).copy()
    usable[EVAL_LABEL] = usable[EVAL_LABEL].astype(float)

    print(f"{len(usable):,} rows with both a 1-week and a {args.horizon}-week label "
          f"across seasons {seasons[0]}-{seasons[-1]}")

    shared = dict(
        table=usable, feat_cols=feat_cols, flag_margin=config.flag_margin, label_col=EVAL_LABEL
    )
    report_a = run_walk_forward(**shared, fit_predict=_fit_on("label_next_week_points"))
    report_b = run_walk_forward(**shared, fit_predict=_fit_on(EVAL_LABEL))

    _print_side_by_side("trained 1wk", report_a, f"trained {args.horizon}wk", report_b)

    pooled_diff = report_a["pooled_model_mae"] - report_b["pooled_model_mae"]
    print(f"Pooled MAE difference: {pooled_diff:+.4f} in favour of "
          f"{'the 3-week label' if pooled_diff > 0 else 'the shipped 1-week label'}")
    print("Nothing was changed. Adopting a new label means editing train.py deliberately.\n")


if __name__ == "__main__":
    main()
