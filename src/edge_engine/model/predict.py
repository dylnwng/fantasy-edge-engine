"""Score the most recent trailing-window player-weeks with the trained
opportunity model. This is the "ranked score, not just a point estimate"
deliverable from requirement 3 — requirement 4 (free-agent ranking
output) filters this down to the free-agent pool and adds explanations.

Usage:
    python -m edge_engine.model.predict
"""

from __future__ import annotations

import pandas as pd
import xgboost as xgb

from edge_engine.model.config import ModelConfig, load_model_config
from edge_engine.model.features import build_features, feature_columns
from edge_engine.model.injury_context import get_injury_context, load_injury_reports
from edge_engine.model.scoring import compute_points_for_seasons
from edge_engine.model.train import MODEL_PATH
from edge_engine.paths import PLAYER_WEEK_PATH
from edge_engine.roster.interface import get_default_source


def load_model() -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    return model


def score_latest_week(config: ModelConfig | None = None) -> pd.DataFrame:
    """One row per player, using their most recent feature snapshot:
    predicted_score, baseline_score, and whether the flag_margin
    threshold is cleared. Ranked by predicted_score descending.

    Players with fewer than `trailing_window` games played this season
    have no valid features yet (see features.py) and are excluded here,
    not scored with degraded/null inputs."""
    config = config or load_model_config()
    model = load_model()

    player_week = pd.read_parquet(PLAYER_WEEK_PATH)
    latest_season = int(player_week["season"].max())
    player_week = player_week[player_week["season"] == latest_season]

    league_config = get_default_source().get_league_config()
    points_table = compute_points_for_seasons([latest_season], league_config.scoring)
    merged = player_week.merge(points_table, on=["season", "week", "player_id"], how="inner")

    features = build_features(merged, merged["points"], window=config.trailing_window)
    feat_cols = feature_columns(config.trailing_window)

    latest = (
        features.dropna(subset=feat_cols)
        .sort_values(["player_id", "week"])
        .groupby("player_id", as_index=False)
        .tail(1)
        .copy()
    )

    latest["predicted_score"] = model.predict(latest[feat_cols])
    latest["baseline_score"] = latest["trailing_points_avg"]
    latest["flagged"] = (latest["predicted_score"] - latest["baseline_score"]) > config.flag_margin

    # req 3b: attach injury context per candidate so a downstream ranking
    # explanation (requirement 4) can state whether the usage spike might
    # be injury-driven rather than an earned role change.
    injuries = load_injury_reports([latest_season])
    contexts = [
        get_injury_context(
            player_week,
            injuries,
            row.player_id,
            int(row.season),
            int(row.week),
            usage_gap=config.injury_ahead_usage_gap,
            lookback=config.injury_lookback_weeks,
        )
        for row in latest.itertuples()
    ]
    latest["has_injury_context"] = [c.has_injury_context for c in contexts]
    latest["injury_explanation"] = [c.explanation for c in contexts]

    return latest.sort_values("predicted_score", ascending=False).reset_index(drop=True)


if __name__ == "__main__":
    result = score_latest_week()
    cols = ["season", "week", "player_id", "position", "team", "predicted_score", "baseline_score", "flagged"]
    print(result[cols].head(25).to_string(index=False))
    print()
    print("--- flagged players with injury context ---")
    with_context = result[result["flagged"] & result["has_injury_context"]]
    if with_context.empty:
        print("(none of the flagged players this week have injury context)")
    else:
        for row in with_context.itertuples():
            print(f"{row.player_id} ({row.position}, {row.team}): {row.injury_explanation}")
