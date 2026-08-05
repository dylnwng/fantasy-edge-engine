"""Accuracy experiment: does adding an opponent-adjusted ("defense vs.
position" / DvP) feature improve the opportunity model?

Trains a variant model with dvp_points_allowed_avg added to
feature_columns(), evaluates MAE/hit-rate on the same held-out
validation season as the general model, and reports both side by side
-- a measured comparison, not a replacement. Promotion to the default
feature set (features.py's feature_columns(), predict.py's
score_as_of_week) only happens after a human reviews
training_metrics_dvp.json and decides it's worth it -- mirroring
train_by_position.py's "report honestly, human decides" pattern (which
itself found a mixed result: helped TE, didn't help RB/WR).

Usage:
    python -m edge_engine.model.train_with_dvp
"""

from __future__ import annotations

import json
import logging

import pandas as pd
import xgboost as xgb

from edge_engine.model.config import ModelConfig, load_model_config
from edge_engine.model.features import feature_columns
from edge_engine.model.opponent_defense import (
    build_points_allowed_table,
    build_schedule_opponent_table,
    compute_dvp_feature,
)
from edge_engine.model.train import MODEL_PATH, build_training_table_with_merged
from edge_engine.paths import ROOT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = ROOT_DIR / "models" / "dvp"
METRICS_PATH = ROOT_DIR / "models" / "training_metrics_dvp.json"
DVP_FEATURE_COL = "dvp_points_allowed_avg"


def _build_table_with_dvp(config: ModelConfig, dvp_window: int) -> tuple[pd.DataFrame, list[str]]:
    features, merged = build_training_table_with_merged(config)
    all_seasons = sorted(set(config.train_seasons) | {config.validation_season})

    schedule = build_schedule_opponent_table(all_seasons)
    allowed = build_points_allowed_table(merged, merged["points"], window=dvp_window)
    dvp = compute_dvp_feature(merged, schedule, allowed, target_week=None)

    augmented = features.merge(dvp, on=["season", "week", "player_id"], how="left")
    feat_cols = feature_columns(config.trailing_window) + [DVP_FEATURE_COL]
    return augmented, feat_cols


def train_and_evaluate_with_dvp(config: ModelConfig | None = None, dvp_window: int = 8) -> dict:
    config = config or load_model_config()
    table, feat_cols = _build_table_with_dvp(config, dvp_window)

    usable = table.dropna(subset=[*feat_cols, "label_next_week_points"])
    train_df = usable[usable["season"].isin(config.train_seasons)]
    val_df = usable[usable["season"] == config.validation_season]

    if train_df.empty or val_df.empty:
        raise ValueError("Empty train or validation set with DvP feature added -- check dvp_window and seasons.")

    model = xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=200, max_depth=4, learning_rate=0.05, random_state=0,
    )
    model.fit(train_df[feat_cols], train_df["label_next_week_points"])

    val_df = val_df.copy()
    val_df["predicted_score"] = model.predict(val_df[feat_cols])
    val_df["baseline_score"] = val_df["trailing_points_avg"]

    model_mae = float((val_df["predicted_score"] - val_df["label_next_week_points"]).abs().mean())
    baseline_mae = float((val_df["baseline_score"] - val_df["label_next_week_points"]).abs().mean())

    flagged = val_df[val_df["predicted_score"] - val_df["baseline_score"] > config.flag_margin]
    hit_rate = (
        float((flagged["label_next_week_points"] > flagged["baseline_score"]).mean()) if len(flagged) else None
    )

    # Compare against the ALREADY-TRAINED general model, evaluated on
    # exactly the same DvP-augmented validation rows -- rows dropped for
    # lacking a DvP value (e.g. early-season weeks before a defense has
    # `dvp_window` trailing games) are dropped from both sides
    # identically, so this is apples to apples, not just a bigger
    # validation set for one side.
    general_model = xgb.XGBRegressor()
    general_model.load_model(str(MODEL_PATH))
    general_feat_cols = feature_columns(config.trailing_window)
    general_predicted = general_model.predict(val_df[general_feat_cols])
    general_mae = float((general_predicted - val_df["label_next_week_points"]).abs().mean())
    general_flagged_mask = (general_predicted - val_df["baseline_score"].to_numpy()) > config.flag_margin
    general_hit_rate = (
        float(
            (
                val_df.loc[general_flagged_mask, "label_next_week_points"]
                > val_df.loc[general_flagged_mask, "baseline_score"]
            ).mean()
        )
        if general_flagged_mask.sum()
        else None
    )

    metrics = {
        "validation_season": config.validation_season,
        "train_seasons": config.train_seasons,
        "dvp_window": dvp_window,
        "n_train_rows": int(len(train_df)),
        "n_validation_rows": int(len(val_df)),
        "with_dvp": {"model_mae": model_mae, "n_flagged": int(len(flagged)), "hit_rate": hit_rate},
        "general_model_same_rows": {
            "model_mae": general_mae,
            "n_flagged": int(general_flagged_mask.sum()),
            "hit_rate": general_hit_rate,
        },
        "baseline_mae": baseline_mae,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODELS_DIR / "opportunity_model_with_dvp.json"))
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    winner = "DvP model wins" if model_mae < general_mae else "general model wins or ties"
    logger.info(
        "With DvP: MAE %.2f, hit rate %s (n=%d) | General (same rows): MAE %.2f, hit rate %s (n=%d) | "
        "baseline MAE %.2f -- %s",
        model_mae,
        f"{hit_rate:.1%}" if hit_rate is not None else "n/a",
        len(flagged),
        general_mae,
        f"{general_hit_rate:.1%}" if general_hit_rate is not None else "n/a",
        int(general_flagged_mask.sum()),
        baseline_mae,
        winner,
    )
    return metrics


if __name__ == "__main__":
    train_and_evaluate_with_dvp()
