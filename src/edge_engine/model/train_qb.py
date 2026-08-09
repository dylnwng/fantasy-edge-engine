"""Train the quarterback opportunity model.

A separate artifact from the main model, not an extension of it: QBs have
a different feature space (volume, not receiving usage), so they get their
own estimator. Nothing here touches the validated WR/RB/TE model.

Validation mirrors train.py's, and additionally checks the thing that
justified building this at all -- that it beats a trailing-points baseline
with a paired bootstrap whose interval excludes zero. See
scripts/qb_signal_probe.py for the two-season replication that cleared
this for shipping in the first place.

Usage:
    python -m edge_engine.model.train_qb
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import xgboost as xgb

from edge_engine.model.config import ModelConfig, load_model_config
from edge_engine.model.qb_features import (
    DEFAULT_WINDOW,
    build_qb_features,
    build_qb_volume,
    qb_feature_columns,
)
from edge_engine.model.scoring import compute_points_for_seasons
from edge_engine.paths import ROOT_DIR
from edge_engine.roster.interface import get_default_source

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

QB_MODEL_PATH = ROOT_DIR / "models" / "qb_model.json"
QB_METRICS_PATH = ROOT_DIR / "models" / "training_metrics_qb.json"


def build_qb_training_table(config: ModelConfig, window: int = DEFAULT_WINDOW) -> pd.DataFrame:
    seasons = sorted(set(config.train_seasons) | {config.validation_season})
    scoring = get_default_source().get_league_config().scoring

    frames = []
    for season in seasons:
        try:
            volume = build_qb_volume(season)
        except RuntimeError as e:
            logger.warning("Skipping %s: %s", season, e)
            continue
        points = compute_points_for_seasons([season], scoring)
        frames.append(build_qb_features(volume, points, window=window))

    if not frames:
        raise RuntimeError("No QB data could be built for any configured season.")
    return pd.concat(frames, ignore_index=True)


def _bootstrap_gain(model_err: np.ndarray, base_err: np.ndarray, n_boot: int = 2000) -> dict:
    """Paired bootstrap on MAE improvement over the baseline. Positive
    means the model is better; the interval must exclude zero for this
    model to deserve shipping."""
    rng = np.random.default_rng(0)
    diffs = np.array([
        base_err[idx].mean() - model_err[idx].mean()
        for idx in (rng.integers(0, len(model_err), len(model_err)) for _ in range(n_boot))
    ])
    return {
        "mean_mae_gain": float(diffs.mean()),
        "ci_5th": float(np.percentile(diffs, 5)),
        "ci_95th": float(np.percentile(diffs, 95)),
        "fraction_favoring_model": float((diffs > 0).mean()),
    }


def train_and_evaluate_qb(config: ModelConfig | None = None, window: int = DEFAULT_WINDOW) -> dict:
    config = config or load_model_config()
    table = build_qb_training_table(config, window)
    feat_cols = qb_feature_columns(window)

    usable = table.dropna(subset=[*feat_cols, "label_next_week_points"])
    train_df = usable[usable["season"].isin(config.train_seasons)]
    val_df = usable[usable["season"] == config.validation_season].copy()

    if train_df.empty or val_df.empty:
        raise ValueError("Empty QB train or validation set — check model_config.yaml seasons.")

    model = xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=200, max_depth=4,
        learning_rate=0.05, random_state=0,
    )
    model.fit(train_df[feat_cols], train_df["label_next_week_points"])

    val_df["predicted_score"] = model.predict(val_df[feat_cols])
    val_df["baseline_score"] = val_df["trailing_points_avg"]

    actual = val_df["label_next_week_points"].to_numpy()
    model_err = np.abs(val_df["predicted_score"].to_numpy() - actual)
    base_err = np.abs(val_df["baseline_score"].to_numpy() - actual)
    significance = _bootstrap_gain(model_err, base_err)

    flagged = val_df[val_df["predicted_score"] - val_df["baseline_score"] > config.flag_margin]
    hit_rate = (
        float((flagged["label_next_week_points"] > flagged["baseline_score"]).mean())
        if len(flagged) else None
    )

    metrics = {
        "validation_season": config.validation_season,
        "train_seasons": config.train_seasons,
        "window": window,
        "n_train_rows": int(len(train_df)),
        "n_validation_rows": int(len(val_df)),
        "model_mae": float(model_err.mean()),
        "baseline_mae": float(base_err.mean()),
        "n_flagged": int(len(flagged)),
        "hit_rate": hit_rate,
        "vs_baseline": significance,
    }

    QB_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(QB_MODEL_PATH))
    with open(QB_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        "QB validation MAE — model: %.2f, baseline: %.2f (n=%d)",
        metrics["model_mae"], metrics["baseline_mae"], len(val_df),
    )
    logger.info(
        "Paired bootstrap: %+.2f MAE gain, 90%% CI [%+.2f, %+.2f], %.0f%% favor the model",
        significance["mean_mae_gain"], significance["ci_5th"], significance["ci_95th"],
        significance["fraction_favoring_model"] * 100,
    )
    if significance["ci_5th"] <= 0:
        logger.warning(
            "The QB model no longer beats its baseline with the CI excluding zero. It shipped "
            "on a two-season replication; if that has stopped holding, stop serving it rather "
            "than quietly serving a model that lost its justification."
        )
    logger.info(
        "Flagged %d/%d QBs at margin=%.1f; hit rate: %s",
        len(flagged), len(val_df), config.flag_margin,
        f"{hit_rate:.1%}" if hit_rate is not None else "n/a (nothing flagged)",
    )
    return metrics


if __name__ == "__main__":
    train_and_evaluate_qb()
