"""Score quarterbacks with the QB model.

Returns the same column contract `predict.score_as_of_week` produces, so
QB rows can be unioned into the main scored frame and every downstream
consumer (rankings, roster-fit, insights, the dashboard) works unchanged
without knowing a second model exists.

The same week-scoping discipline applies: features come only from games
played strictly before `target_week`. Getting this wrong is exactly the
leak that was found and fixed in the main model.
"""

from __future__ import annotations

import pandas as pd
import xgboost as xgb

from edge_engine.model.config import ModelConfig, load_model_config
from edge_engine.model.injury_context import get_injury_context, load_injury_reports
from edge_engine.model.qb_features import (
    DEFAULT_WINDOW,
    build_qb_features,
    build_qb_volume,
    qb_feature_columns,
    qb_usage_explanation,
)
from edge_engine.model.scoring import compute_points_for_seasons
from edge_engine.model.train_qb import QB_MODEL_PATH
from edge_engine.roster.interface import get_default_source
from edge_engine.roster.roster_status import get_flagged_roster_statuses, roster_status_note

OUTPUT_COLUMNS = [
    "season", "week", "player_id", "position", "team",
    "trailing_points_avg", "predicted_score", "baseline_score", "flagged",
    "has_injury_context", "injury_explanation", "roster_status_note",
    # The main model's usage_trend_explanation reads receiving metrics that
    # are meaningless for a passer, so QBs carry their own volume-based
    # explanation and ranking/output.py prefers it when present.
    "usage_explanation",
]


def qb_model_available() -> bool:
    """Whether a trained QB artifact exists. Callers union QB rows in only
    when it does, so a checkout that never ran `train_qb` degrades to the
    pre-QB behaviour instead of crashing."""
    return QB_MODEL_PATH.exists()


def score_qbs_as_of_week(
    season: int,
    target_week: int,
    config: ModelConfig | None = None,
    player_week: pd.DataFrame | None = None,
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """One row per quarterback, using each QB's trailing volume snapshot
    from his last game played strictly before `target_week`."""
    config = config or load_model_config()
    empty = pd.DataFrame(columns=OUTPUT_COLUMNS)

    if not qb_model_available():
        return empty

    try:
        volume = build_qb_volume(season)
    except RuntimeError:
        return empty  # no play-by-play for this season yet

    volume = volume[volume["week"] < target_week]
    if volume.empty:
        return empty

    scoring = get_default_source().get_league_config().scoring
    points = compute_points_for_seasons([season], scoring)
    features = build_qb_features(volume, points, window=window)

    feat_cols = qb_feature_columns(window)
    latest = (
        features.dropna(subset=feat_cols)
        .sort_values(["player_id", "week"])
        .groupby("player_id", as_index=False)
        .tail(1)
        .copy()
    )
    if latest.empty:
        return empty

    model = xgb.XGBRegressor()
    model.load_model(str(QB_MODEL_PATH))
    latest["predicted_score"] = model.predict(latest[feat_cols])
    latest["baseline_score"] = latest["trailing_points_avg"]
    latest["flagged"] = (latest["predicted_score"] - latest["baseline_score"]) > config.flag_margin

    # Injury context works for QBs too, and arguably means MORE here: a
    # backup's snap share only rises when the starter ahead of him is
    # hurt, which is the single biggest driver of QB fantasy relevance.
    if player_week is not None and not player_week.empty:
        injuries = load_injury_reports([season])
        contexts = [
            get_injury_context(
                player_week, injuries, row.player_id, int(row.season), int(row.week),
                usage_gap=config.injury_ahead_usage_gap,
                lookback=config.injury_lookback_weeks,
            )
            for row in latest.itertuples()
        ]
        latest["has_injury_context"] = [c.has_injury_context for c in contexts]
        latest["injury_explanation"] = [c.explanation for c in contexts]
    else:
        latest["has_injury_context"] = False
        latest["injury_explanation"] = ""

    statuses = get_flagged_roster_statuses()
    latest["roster_status_note"] = [roster_status_note(row.player_id, statuses) for row in latest.itertuples()]
    latest["usage_explanation"] = [
        qb_usage_explanation(volume, row.player_id, int(row.season), int(row.week), window)
        for row in latest.itertuples()
    ]

    return latest[OUTPUT_COLUMNS].reset_index(drop=True)
