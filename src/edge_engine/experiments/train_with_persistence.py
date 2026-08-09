"""Accuracy experiment: do usage-*persistence* features improve the
opportunity model?

See persistence.py for the motivation: the existing trend feature is an
endpoint delta, so a one-week spike and a sustained multi-week climb look
identical to it. This trains a variant model with the persistence
features added and compares it against the already-trained general model
on exactly the same held-out validation rows.

Reports honestly and does NOT wire itself into the default pipeline --
same "measured comparison, human decides" discipline as
train_with_dvp.py (which found a genuine negative result and was
rejected) and train_by_position.py (mixed: helped TE, not RB/WR).

Usage:
    python -m edge_engine.model.train_with_persistence
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import xgboost as xgb

from edge_engine.model.config import ModelConfig, load_model_config
from edge_engine.model.features import feature_columns
from edge_engine.experiments.persistence import (
    PERSISTENCE_WINDOW,
    compute_persistence_features,
    persistence_feature_columns,
)
from edge_engine.model.train import MODEL_PATH, build_training_table_with_merged
from edge_engine.paths import ROOT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = ROOT_DIR / "models" / "persistence"
METRICS_PATH = ROOT_DIR / "models" / "training_metrics_persistence.json"


def _bootstrap_hit_rate_difference(
    persistence_pred, general_pred, baseline, actual, flag_margin: float, n_boot: int = 2000, seed: int = 0
) -> dict:
    """Paired bootstrap on the hit-rate difference (persistence minus
    general).

    A raw hit-rate gap of a couple of points on ~300 flagged rows is
    exactly the kind of number that looks like a win and is actually
    sampling noise -- this project already rejected the DvP experiment
    for moving metrics by trivial margins, so a marginal *positive*
    result deserves the same scrutiny rather than a pass.

    Paired, not two independent bootstraps: each iteration resamples
    validation ROWS once and scores both models on that same resample,
    which respects the fact that the two flagged sets overlap heavily
    (mostly the same players). Two independent resamples would
    artificially inflate the variance of the difference."""
    rng = np.random.default_rng(seed)
    n = len(actual)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        b, a = baseline[idx], actual[idx]
        pers_flagged = (persistence_pred[idx] - b) > flag_margin
        gen_flagged = (general_pred[idx] - b) > flag_margin
        if not pers_flagged.any() or not gen_flagged.any():
            continue
        diffs.append(
            float((a[pers_flagged] > b[pers_flagged]).mean() - (a[gen_flagged] > b[gen_flagged]).mean())
        )

    if not diffs:
        return {"n_boot_effective": 0}
    diffs = np.array(diffs)
    return {
        "n_boot_effective": int(len(diffs)),
        "mean_hit_rate_diff": float(diffs.mean()),
        "ci_5th": float(np.percentile(diffs, 5)),
        "ci_95th": float(np.percentile(diffs, 95)),
        "fraction_favoring_persistence": float((diffs > 0).mean()),
    }


def _build_table_with_persistence(
    config: ModelConfig, window: int
) -> tuple[pd.DataFrame, list[str]]:
    features, merged = build_training_table_with_merged(config)
    persistence = compute_persistence_features(merged, window=window)
    augmented = features.merge(persistence, on=["season", "week", "player_id"], how="left")
    feat_cols = feature_columns(config.trailing_window) + persistence_feature_columns()
    return augmented, feat_cols


def train_and_evaluate_with_persistence(
    config: ModelConfig | None = None, window: int = PERSISTENCE_WINDOW
) -> dict:
    config = config or load_model_config()
    table, feat_cols = _build_table_with_persistence(config, window)

    usable = table.dropna(subset=[*feat_cols, "label_next_week_points"])
    train_df = usable[usable["season"].isin(config.train_seasons)]
    val_df = usable[usable["season"] == config.validation_season]

    if train_df.empty or val_df.empty:
        raise ValueError(
            "Empty train or validation set with persistence features added -- check window and seasons."
        )

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

    # Same apples-to-apples comparison as train_with_dvp.py: the
    # already-trained general model, scored on the identical validation
    # rows. Rows dropped for lacking a persistence value (a player's
    # first few games of a season, before `window - 1` real deltas
    # exist) are dropped from both sides equally.
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

    importances = dict(
        zip(feat_cols, (float(v) for v in model.feature_importances_))
    )
    persistence_importance = {k: v for k, v in importances.items() if k in persistence_feature_columns()}

    significance = _bootstrap_hit_rate_difference(
        persistence_pred=val_df["predicted_score"].to_numpy(),
        general_pred=general_predicted,
        baseline=val_df["baseline_score"].to_numpy(),
        actual=val_df["label_next_week_points"].to_numpy(),
        flag_margin=config.flag_margin,
    )

    metrics = {
        "validation_season": config.validation_season,
        "train_seasons": config.train_seasons,
        "persistence_window": window,
        "n_train_rows": int(len(train_df)),
        "n_validation_rows": int(len(val_df)),
        "with_persistence": {"model_mae": model_mae, "n_flagged": int(len(flagged)), "hit_rate": hit_rate},
        "general_model_same_rows": {
            "model_mae": general_mae,
            "n_flagged": int(general_flagged_mask.sum()),
            "hit_rate": general_hit_rate,
        },
        "baseline_mae": baseline_mae,
        "persistence_feature_importance": persistence_importance,
        "hit_rate_significance": significance,
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(str(MODELS_DIR / "opportunity_model_with_persistence.json"))
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        "With persistence: MAE %.2f, hit rate %s (n=%d) | General (same rows): MAE %.2f, hit rate %s (n=%d) "
        "| baseline MAE %.2f",
        model_mae,
        f"{hit_rate:.1%}" if hit_rate is not None else "n/a",
        len(flagged),
        general_mae,
        f"{general_hit_rate:.1%}" if general_hit_rate is not None else "n/a",
        int(general_flagged_mask.sum()),
        baseline_mae,
    )
    logger.info("Persistence feature importances: %s", persistence_importance)
    if significance.get("n_boot_effective"):
        logger.info(
            "Paired bootstrap on hit-rate difference: mean %+.1f pp, 90%% CI [%+.1f, %+.1f] pp, "
            "%.0f%% of resamples favor persistence",
            significance["mean_hit_rate_diff"] * 100,
            significance["ci_5th"] * 100,
            significance["ci_95th"] * 100,
            significance["fraction_favoring_persistence"] * 100,
        )
    return metrics


if __name__ == "__main__":
    train_and_evaluate_with_persistence()
