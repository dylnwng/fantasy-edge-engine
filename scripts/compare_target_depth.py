"""Does depth of target add anything over air-yards share?

The hypothesis
--------------
`air_yards_share` says how much of a team's downfield passing runs
through a player. It cannot tell a checkdown back from a field stretcher:
two players can hold similar shares while one accumulates his yards eight
at a time and the other twenty. aDOT -- air yards per target -- is that
missing dimension.

The honest prior
----------------
This is the most INCREMENTAL of the candidate features. It sits adjacent
to a signal already in the model rather than opening a new axis, which is
precisely the shape of the two ideas this project has already rejected
(opponent adjustment and usage persistence were both rearrangements of
existing signals). Expect it to fail, and be pleased rather than
surprised if it doesn't.

A population caveat that matters
--------------------------------
Trailing aDOT is undefined for a player with no targets across the
window, and that is left as null rather than filled with 0.0 -- a
non-receiving back has no depth of target, not a depth of zero. So the
comparison necessarily runs on the sub-population of players who were
actually thrown to.

BOTH candidates are restricted to those same rows, so neither is
evaluated on an easier set. But the result generalises only to that
sub-population, and the row count is printed so the size of that
restriction is visible rather than buried. If aDOT wins on receivers but
the restriction has thrown away half the pool, that is a narrower finding
than the headline number suggests.

The comparison
--------------
    candidate A: the shipped features
    candidate B: the shipped features + trailing_adot

Same estimator, same hyperparameters, same rows, same target, same folds.

The verdict
-----------
FOLD AGREEMENT, and the hit-rate columns alongside MAE.

This changes nothing.

Usage:
    python scripts/compare_target_depth.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_engine.model.config import load_model_config  # noqa: E402
from edge_engine.model.features import feature_columns  # noqa: E402
from edge_engine.model.target_depth import (  # noqa: E402
    DEPTH_COLUMNS,
    attach_target_depth,
    build_target_depth,
    trailing_adot,
)
from edge_engine.model.train import build_estimator, build_training_table_with_merged  # noqa: E402
from edge_engine.model.walk_forward import run_walk_forward  # noqa: E402


def _fit_on(feat_cols: list[str]):
    def _fp(train_df, val_df, _ignored):
        model = build_estimator()
        model.fit(train_df[feat_cols], train_df["label_next_week_points"])
        return model.predict(val_df[feat_cols])

    return _fp


def _print_comparison(report_a: dict, report_b: dict, n_kept: int, n_total: int) -> None:
    header = f"{'Season':<8}{'shipped':>12}{'+aDOT':>12}{'Diff':>10}{'Hit A':>9}{'Hit B':>9}"
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
    print(f"\nFold agreement (MAE):      aDOT wins {b_better}/{n} seasons   <- the test to believe")
    print(f"Fold agreement (hit rate): aDOT wins {hit_b_better}/{hit_compared} seasons")
    print(f"\nPopulation: {n_kept:,} of {n_total:,} rows ({n_kept / n_total:.1%}) had a defined "
          "trailing aDOT and were used by both candidates.")
    if n_kept / n_total < 0.75:
        print("That restriction is large. Any win here applies to players who get targeted,")
        print("not to the ranked pool as a whole.")

    print()
    if b_better > n / 2 and hit_b_better >= hit_compared / 2:
        print("Replicates on both MAE and hit rate, which is more than the prior expected.")
        print("Read the population line above before treating it as a general result.")
    elif b_better > n / 2:
        print("MAE improves but the hit rate does not follow — the pattern that sank")
        print("opponent adjustment. The flagged list is what people act on. Do not adopt.")
    elif hit_compared and hit_b_better > hit_compared / 2:
        print("Hit rate replicates but MAE does not. That is the sub-population that matters")
        print("most -- the flagged list is the product, not the average error -- but it is")
        print("also by far the smallest sample here (a few hundred players a season), which")
        print("is exactly the size that manufactures effects. Suggestive, not a result:")
        print("re-run before acting, and do not adopt on this alone.")
    else:
        print("Does NOT replicate across seasons. On this project's own standard that is a")
        print("rejection, and the expected one: incremental variants of existing signals are")
        print("exactly where the previous rejections cluster.")
    print()


def main() -> None:
    config = load_model_config()
    seasons = sorted(set(config.train_seasons) | {config.validation_season})

    features, _merged = build_training_table_with_merged(config)

    print(f"Building target depth for {seasons[0]}-{seasons[-1]}...")
    raw_depth = pd.concat([build_target_depth(s) for s in seasons], ignore_index=True)
    depth = trailing_adot(raw_depth, window=config.trailing_window)
    table = attach_target_depth(features, depth)

    cols_a = feature_columns(config.trailing_window)
    cols_b = cols_a + list(DEPTH_COLUMNS)

    before = table.dropna(subset=[*cols_a, "label_next_week_points"])
    usable = table.dropna(subset=[*cols_b, "label_next_week_points"])
    print(f"{len(usable):,} rows usable by both candidates "
          f"({len(cols_a)} features vs {len(cols_b)})")

    shared = dict(table=usable, flag_margin=config.flag_margin)
    report_a = run_walk_forward(**shared, feat_cols=cols_a, fit_predict=_fit_on(cols_a))
    report_b = run_walk_forward(**shared, feat_cols=cols_b, fit_predict=_fit_on(cols_b))

    _print_comparison(report_a, report_b, len(usable), len(before))
    print("Nothing was changed. Adopting aDOT means editing features.py and the ingestion")
    print("transform deliberately, then re-ingesting.\n")


if __name__ == "__main__":
    main()
