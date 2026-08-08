"""What flag_margin should this league actually run at?

model_config.yaml ships flag_margin=3.0, which was picked by hand and
never swept. It's the single knob that decides *which names you see on a
Tuesday*, so it's worth knowing the real precision/volume curve instead
of trusting a round number.

Reports, for each candidate margin on the held-out validation season:
  - how many players get flagged (volume)
  - what fraction of them beat their own naive baseline the following
    week (1-week hit rate -- the PRD's precision metric)
  - what fraction beat it averaged over the following up-to-3 games
    (3-week hit rate -- closer to how a waiver claim actually pays off,
    since you don't drop a pickup after one week)
  - average points above baseline, so "high hit rate on trivially small
    wins" is visible rather than hidden behind a percentage

Usage:
    python scripts/sweep_flag_margin.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_engine.model.config import load_model_config  # noqa: E402
from edge_engine.model.features import feature_columns  # noqa: E402
from edge_engine.model.scoring import compute_points_for_seasons  # noqa: E402
from edge_engine.model.train import MODEL_PATH, build_training_table  # noqa: E402
from edge_engine.roster.interface import get_default_source  # noqa: E402

MARGINS = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]


def _forward_3game_avg(points_table: pd.DataFrame) -> pd.DataFrame:
    """Each player-week's average points over their NEXT up-to-3 played
    games (not counting the current one). Same
    'next played game' convention as features.py's label -- byes are
    simply absent rows, so they're skipped rather than counted as zeros."""
    df = points_table.sort_values(["player_id", "season", "week"]).copy()
    grouped = df.groupby(["player_id", "season"], group_keys=False)["points"]
    # shift(-1) first so the current week is excluded, then average the
    # next three.
    df["following_avg_points"] = grouped.transform(
        lambda s: s.shift(-1).rolling(3, min_periods=1).mean().shift(-2)
    )
    return df[["season", "week", "player_id", "following_avg_points"]]


def main() -> None:
    config = load_model_config()
    table = build_training_table(config)
    feat_cols = feature_columns(config.trailing_window)

    league_config = get_default_source().get_league_config()
    points_table = compute_points_for_seasons([config.validation_season], league_config.scoring)
    forward = _forward_3game_avg(points_table)

    usable = table.dropna(subset=[*feat_cols, "label_next_week_points"])
    val = usable[usable["season"] == config.validation_season].copy()

    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    val["predicted_score"] = model.predict(val[feat_cols])
    val["baseline_score"] = val["trailing_points_avg"]
    val = val.merge(forward, on=["season", "week", "player_id"], how="left")

    print(f"Validation season {config.validation_season} — {len(val)} scored player-weeks")
    print(f"(shipped flag_margin = {config.flag_margin})\n")
    header = f"{'margin':<9}{'flagged':<10}{'1wk hit':<10}{'3wk hit':<10}{'avg pts over baseline':<24}"
    print(header)
    print("-" * len(header))

    for margin in MARGINS:
        flagged = val[(val["predicted_score"] - val["baseline_score"]) > margin]
        if flagged.empty:
            print(f"{margin:<9.1f}{0:<10}{'n/a':<10}{'n/a':<10}{'n/a':<24}")
            continue

        hit_1wk = float((flagged["label_next_week_points"] > flagged["baseline_score"]).mean())
        over_baseline = float((flagged["label_next_week_points"] - flagged["baseline_score"]).mean())

        with_forward = flagged.dropna(subset=["following_avg_points"])
        hit_3wk = (
            float((with_forward["following_avg_points"] > with_forward["baseline_score"]).mean())
            if len(with_forward)
            else float("nan")
        )

        marker = "   <- shipped" if np.isclose(margin, config.flag_margin) else ""
        print(
            f"{margin:<9.1f}{len(flagged):<10}{hit_1wk:<10.1%}"
            f"{hit_3wk:<10.1%}{over_baseline:<+24.2f}{marker}"
        )

    print(
        "\nRead this as a tradeoff curve, not a leaderboard: a higher margin should buy a "
        "higher hit rate at the cost of volume. If it doesn't -- if hit rate is flat across "
        "margins -- then the margin isn't actually selecting for anything and the current "
        "value is as good as any."
    )


if __name__ == "__main__":
    main()
