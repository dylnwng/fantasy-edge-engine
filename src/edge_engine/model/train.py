"""Train the opportunity model (requirement 3): predicts each player's
next-week fantasy-point ceiling from trailing usage trends, validated on
a held-out SEASON (not held-out weeks — within a season, a hot player
stays hot for a few weeks, so a week-level split would leak signal and
overstate accuracy) and compared against a naive rolling-average
baseline.

Usage:
    python -m edge_engine.model.train
"""

from __future__ import annotations

import json
import logging

import pandas as pd
import xgboost as xgb

from edge_engine.ingestion.pipeline import ingest_seasons
from edge_engine.model.config import ModelConfig, load_model_config
from edge_engine.model.features import build_features, feature_columns
from edge_engine.model.scoring import compute_points_for_seasons
from edge_engine.paths import PLAYER_WEEK_PATH, ROOT_DIR
from edge_engine.roster.interface import get_default_source

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PATH = ROOT_DIR / "models" / "opportunity_model.json"
METRICS_PATH = ROOT_DIR / "models" / "training_metrics.json"

# NOTE: an earlier version of this used quantile regression (alpha=0.8) to
# chase the PRD's "ceiling" language literally. That broke the comparison
# against the baseline: an 80th-percentile predictor is *systematically*
# above a mean predictor almost everywhere, which isn't the model
# detecting anything — it's just what predicting a high percentile does.
# It made model MAE look worse than the baseline's (7.1 vs 5.6 pts) and
# flagged 79% of all players at the default margin, the opposite of the
# short, high-trust list the PRD calls for. Mean regression puts the
# model and the (also-mean) baseline on the same footing, so MAE and the
# margin threshold are actually comparable. "Ceiling" here means "usage
# trends imply upside versus the naive baseline," not a literal
# percentile estimate.


def _ensure_seasons_ingested(seasons: list[int]) -> pd.DataFrame:
    if not PLAYER_WEEK_PATH.exists():
        ingest_seasons(seasons)
    player_week = pd.read_parquet(PLAYER_WEEK_PATH)
    missing = sorted(set(seasons) - set(player_week["season"].unique()))
    if missing:
        logger.info("Ingesting missing seasons: %s", missing)
        ingest_seasons(missing)
        player_week = pd.read_parquet(PLAYER_WEEK_PATH)
    return player_week


def build_training_table_with_merged(config: ModelConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Same as build_training_table(), but also returns the intermediate
    merged frame (player_week + points, before build_features() drops
    `opponent`/other non-feature columns) -- needed by the opponent_defense
    (DvP) experiment, which requires `opponent` to resolve each player's
    upcoming defense. A pure additive refactor: build_training_table()
    below is unchanged in behavior, just delegates here."""
    all_seasons = sorted(set(config.train_seasons) | {config.validation_season})
    player_week = _ensure_seasons_ingested(all_seasons)
    player_week = player_week[player_week["season"].isin(all_seasons)]

    league_config = get_default_source().get_league_config()
    points_table = compute_points_for_seasons(all_seasons, league_config.scoring)

    merged = player_week.merge(points_table, on=["season", "week", "player_id"], how="inner")
    features = build_features(merged, merged["points"], window=config.trailing_window)
    return features, merged


def build_training_table(config: ModelConfig) -> pd.DataFrame:
    return build_training_table_with_merged(config)[0]


def train_and_evaluate(config: ModelConfig | None = None) -> dict:
    config = config or load_model_config()
    table = build_training_table(config)

    feat_cols = feature_columns(config.trailing_window)
    usable = table.dropna(subset=[*feat_cols, "label_next_week_points"])

    train_df = usable[usable["season"].isin(config.train_seasons)]
    val_df = usable[usable["season"] == config.validation_season]

    if train_df.empty or val_df.empty:
        raise ValueError("Empty train or validation set — check model_config.yaml seasons.")

    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        random_state=0,
    )
    model.fit(train_df[feat_cols], train_df["label_next_week_points"])

    val_df = val_df.copy()
    val_df["predicted_score"] = model.predict(val_df[feat_cols])
    val_df["baseline_score"] = val_df["trailing_points_avg"]

    model_mae = float((val_df["predicted_score"] - val_df["label_next_week_points"]).abs().mean())
    baseline_mae = float((val_df["baseline_score"] - val_df["label_next_week_points"]).abs().mean())

    # Precision-oriented evaluation per the PRD: of the players the
    # threshold actually flags, what fraction outperformed the naive
    # baseline — not how many total breakouts were caught (that's a
    # recall framing the PRD explicitly deprioritized).
    flagged = val_df[val_df["predicted_score"] - val_df["baseline_score"] > config.flag_margin]
    hit_rate = (
        float((flagged["label_next_week_points"] > flagged["baseline_score"]).mean())
        if len(flagged)
        else None
    )

    metrics = {
        "validation_season": config.validation_season,
        "train_seasons": config.train_seasons,
        "trailing_window": config.trailing_window,
        "flag_margin": config.flag_margin,
        "n_train_rows": int(len(train_df)),
        "n_validation_rows": int(len(val_df)),
        "model_mae": model_mae,
        "baseline_mae": baseline_mae,
        "n_flagged": int(len(flagged)),
        "hit_rate": hit_rate,
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODEL_PATH))
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("Validation MAE — model: %.2f pts, baseline: %.2f pts", model_mae, baseline_mae)
    logger.info(
        "Flagged %d/%d players at margin=%.1f; hit rate: %s",
        len(flagged),
        len(val_df),
        config.flag_margin,
        f"{hit_rate:.1%}" if hit_rate is not None else "n/a (nothing flagged)",
    )

    return metrics


if __name__ == "__main__":
    train_and_evaluate()
