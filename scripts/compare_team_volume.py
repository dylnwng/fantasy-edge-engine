"""Does knowing a team's volume improve the opportunity model?

The hypothesis
--------------
Every shipped feature is a SHARE. A 25% target share on an offence
running 70 plays is not the opportunity a 25% share on a 55-play offence
is, and the model cannot currently tell those apart. Team pace and
passing volume are the multiplier that turns a share into a count.

Why this one is worth measuring at all
--------------------------------------
The accuracy ideas this project rejected -- opponent adjustment, usage
persistence -- were variants of signals already in the feature set. Team
volume is a different axis. And the shape has precedent: the QB model
already carries `team_attempts` beside `pass_share`, and the QB model is
the single experiment that replicated across two seasons.

The comparison
--------------
Identical in every respect except the feature set:

    candidate A: the shipped features
    candidate B: the shipped features + trailing team volume

Same estimator, same hyperparameters, same rows, same target, same folds.
Rows are restricted to those where the team-volume features exist, so A
is not evaluated on a larger or easier validation set than B.

Both trail through build_features, so team volume is averaged over games
played BEFORE the predicted week. Feeding the model a team's
contemporaneous pace would leak the game being predicted.

The verdict
-----------
FOLD AGREEMENT -- how many seasons independently favour B. Per
EVALUATION.md, an effect that shows up pooled and in one season is the
signature of noise. Three rejections in this project rest on that.

This changes nothing. Adopting team volume means deliberately editing
features.USAGE_COLUMNS and the ingestion transform, and re-ingesting.

Usage:
    python scripts/compare_team_volume.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_engine.model.config import load_model_config  # noqa: E402
from edge_engine.model.features import USAGE_COLUMNS, build_features, feature_columns  # noqa: E402
from edge_engine.model.scoring import compute_points_for_seasons  # noqa: E402
from edge_engine.model.team_volume import (  # noqa: E402
    TEAM_VOLUME_COLUMNS,
    attach_team_volume,
    build_team_volume,
)
from edge_engine.model.train import _ensure_seasons_ingested, build_estimator  # noqa: E402
from edge_engine.model.walk_forward import run_walk_forward  # noqa: E402
from edge_engine.roster.interface import get_default_source  # noqa: E402


def _fit_on(feat_cols: list[str]):
    """Trains the shipped estimator on a given feature set. Only the
    columns differ between candidates."""

    def _fp(train_df, val_df, _ignored):
        model = build_estimator()
        model.fit(train_df[feat_cols], train_df["label_next_week_points"])
        return model.predict(val_df[feat_cols])

    return _fp


def _print_comparison(report_a: dict, report_b: dict) -> None:
    header = f"{'Season':<8}{'shipped':>12}{'+team vol':>12}{'Diff':>10}{'Hit A':>9}{'Hit B':>9}"
    print("\nMAE on next-week points (lower is better)\n")
    print(header)
    print("-" * len(header))

    b_better = 0
    for fa, fb in zip(report_a["folds"], report_b["folds"]):
        diff = fa["model_mae"] - fb["model_mae"]  # positive => B better
        b_better += diff > 0
        ha = f"{fa['hit_rate']:.1%}" if fa["hit_rate"] is not None else "n/a"
        hb = f"{fb['hit_rate']:.1%}" if fb["hit_rate"] is not None else "n/a"
        print(f"{fa['validation_season']:<8}{fa['model_mae']:>12.3f}{fb['model_mae']:>12.3f}"
              f"{diff:>+10.3f}{ha:>9}{hb:>9}")
    print("-" * len(header))
    print(f"{'Pooled':<8}{report_a['pooled_model_mae']:>12.3f}{report_b['pooled_model_mae']:>12.3f}"
          f"{report_a['pooled_model_mae'] - report_b['pooled_model_mae']:>+10.3f}")

    n = len(report_a["folds"])
    print(f"\nFold agreement: team volume wins {b_better}/{n} seasons   <- the test to believe")
    print(f"Baseline (naive trailing points) MAE, pooled: {report_a['pooled_baseline_mae']:.3f}")

    print()
    if b_better > n / 2:
        print("Team volume replicates across a majority of seasons. Check the per-season")
        print("spread and the hit-rate columns before adopting -- MAE and hit rate can")
        print("disagree, which is how the opponent-adjustment experiment was caught.")
    else:
        print("Team volume does NOT replicate across seasons. On this project's own")
        print("standard that is a rejection, not a result to tune further.")
    print()


def main() -> None:
    config = load_model_config()
    seasons = sorted(set(config.train_seasons) | {config.validation_season})

    player_week = _ensure_seasons_ingested(seasons)
    player_week = player_week[player_week["season"].isin(seasons)]

    print(f"Building team volume for {seasons[0]}-{seasons[-1]}...")
    volume = pd.concat([build_team_volume(s) for s in seasons], ignore_index=True)
    player_week = attach_team_volume(player_week, volume)

    scoring = get_default_source().get_league_config().scoring
    points_table = compute_points_for_seasons(seasons, scoring)
    merged = player_week.merge(points_table, on=["season", "week", "player_id"], how="inner")

    extended = list(USAGE_COLUMNS) + list(TEAM_VOLUME_COLUMNS)
    table = build_features(merged, merged["points"], window=config.trailing_window,
                           usage_columns=extended)

    cols_a = feature_columns(config.trailing_window)
    cols_b = feature_columns(config.trailing_window, usage_columns=extended)

    # Both candidates on the same rows: B needs the team-volume features,
    # so A must be held to that restriction too or it gets an easier set.
    usable = table.dropna(subset=[*cols_b, "label_next_week_points"])
    print(f"{len(usable):,} rows usable by both candidates "
          f"({len(cols_a)} features vs {len(cols_b)})")

    shared = dict(table=usable, flag_margin=config.flag_margin)
    report_a = run_walk_forward(**shared, feat_cols=cols_a, fit_predict=_fit_on(cols_a))
    report_b = run_walk_forward(**shared, feat_cols=cols_b, fit_predict=_fit_on(cols_b))

    _print_comparison(report_a, report_b)
    print("Nothing was changed. Adopting team volume means editing USAGE_COLUMNS and the")
    print("ingestion transform deliberately, then re-ingesting.\n")


if __name__ == "__main__":
    main()
