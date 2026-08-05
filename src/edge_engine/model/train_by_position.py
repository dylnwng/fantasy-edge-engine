"""P1: position-specific models.

Trains a separate XGBoost model per position (QB/RB/WR/TE) instead of
one general model, on the theory that different positions' opportunity
signals behave differently (e.g. red-zone touches probably matter more
for RB/TE, target share more for WR/TE). This has to earn its added
complexity: evaluates each position model against the *general* model's
performance on that same position slice, not just against baseline, so
"splitting by position helps" is a measured claim, not an assumption.

Usage:
    python -m edge_engine.model.train_by_position
"""

from __future__ import annotations

import json
import logging

import pandas as pd
import xgboost as xgb

from edge_engine.model.config import ModelConfig, load_model_config
from edge_engine.model.features import feature_columns
from edge_engine.model.train import MODEL_PATH, build_training_table
from edge_engine.paths import ROOT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

POSITIONS = ["QB", "RB", "WR", "TE"]
MODELS_DIR = ROOT_DIR / "models" / "by_position"
METRICS_PATH = ROOT_DIR / "models" / "training_metrics_by_position.json"
MIN_TRAIN_ROWS = 50  # below this, a position-specific model is more likely to overfit than help


def _train_one_position(table: pd.DataFrame, position: str, config: ModelConfig, feat_cols: list[str]) -> dict:
    pos_table = table[table["position"] == position]
    usable = pos_table.dropna(subset=[*feat_cols, "label_next_week_points"])
    train_df = usable[usable["season"].isin(config.train_seasons)]
    val_df = usable[usable["season"] == config.validation_season]

    if len(train_df) < MIN_TRAIN_ROWS or val_df.empty:
        return {
            "position": position,
            "skipped": True,
            "reason": f"insufficient data (train={len(train_df)}, val={len(val_df)})",
        }

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

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODELS_DIR / f"{position}.json"))

    return {
        "position": position,
        "skipped": False,
        "n_train_rows": int(len(train_df)),
        "n_validation_rows": int(len(val_df)),
        "model_mae": model_mae,
        "baseline_mae": baseline_mae,
        "n_flagged": int(len(flagged)),
        "hit_rate": hit_rate,
    }


def _general_model_metrics_by_position(table: pd.DataFrame, config: ModelConfig, feat_cols: list[str]) -> dict:
    """Load the already-trained general model (requirement 3) and compute
    the same metrics sliced by position, so the comparison is
    apples-to-apples against the position-specific models above."""
    general_model = xgb.XGBRegressor()
    general_model.load_model(str(MODEL_PATH))

    usable = table.dropna(subset=[*feat_cols, "label_next_week_points"])
    val_df = usable[usable["season"] == config.validation_season].copy()
    val_df["predicted_score"] = general_model.predict(val_df[feat_cols])
    val_df["baseline_score"] = val_df["trailing_points_avg"]

    results = {}
    for position in POSITIONS:
        pos_val = val_df[val_df["position"] == position]
        if pos_val.empty:
            continue
        model_mae = float((pos_val["predicted_score"] - pos_val["label_next_week_points"]).abs().mean())
        flagged = pos_val[pos_val["predicted_score"] - pos_val["baseline_score"] > config.flag_margin]
        hit_rate = (
            float((flagged["label_next_week_points"] > flagged["baseline_score"]).mean()) if len(flagged) else None
        )
        results[position] = {"model_mae": model_mae, "n_flagged": int(len(flagged)), "hit_rate": hit_rate}
    return results


def train_and_evaluate_by_position(config: ModelConfig | None = None) -> dict:
    config = config or load_model_config()
    table = build_training_table(config)
    feat_cols = feature_columns(config.trailing_window)

    per_position = {pos: _train_one_position(table, pos, config, feat_cols) for pos in POSITIONS}
    general_by_position = _general_model_metrics_by_position(table, config, feat_cols)

    results = {"per_position_model": per_position, "general_model_by_position": general_by_position}
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    for pos in POSITIONS:
        p = per_position[pos]
        g = general_by_position.get(pos)
        if p.get("skipped"):
            logger.info("%s: skipped — %s", pos, p["reason"])
            continue
        g_mae = f"{g['model_mae']:.2f}" if g else "n/a"
        g_hit = f"{g['hit_rate']:.1%}" if g and g["hit_rate"] is not None else "n/a"
        p_hit = f"{p['hit_rate']:.1%}" if p["hit_rate"] is not None else "n/a"
        better = "position model wins" if g and p["model_mae"] < g["model_mae"] else "general model wins or ties"
        logger.info(
            "%s — position-specific MAE %.2f (hit rate %s) vs general MAE %s (hit rate %s) — %s",
            pos, p["model_mae"], p_hit, g_mae, g_hit, better,
        )

    return results


if __name__ == "__main__":
    train_and_evaluate_by_position()
