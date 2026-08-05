from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from edge_engine.paths import ROOT_DIR

DEFAULT_MODEL_CONFIG_PATH = ROOT_DIR / "model_config.yaml"


@dataclass(frozen=True)
class ModelConfig:
    train_seasons: list[int]
    validation_season: int
    trailing_window: int
    flag_margin: float
    # req 3b: how much higher a same-position teammate's trailing snap_pct
    # must be, over the same lookback window, to count as "ahead of" the
    # candidate (a real pecking-order gap, not noise).
    injury_ahead_usage_gap: float
    # req 3b: how many weeks back (from the week being explained) to check
    # for that teammate's injury report status.
    injury_lookback_weeks: int


def load_model_config(path: Path | None = None) -> ModelConfig:
    path = path or DEFAULT_MODEL_CONFIG_PATH
    with open(path) as f:
        raw = yaml.safe_load(f)
    return ModelConfig(
        train_seasons=raw["train_seasons"],
        validation_season=raw["validation_season"],
        trailing_window=raw.get("trailing_window", 2),
        flag_margin=raw.get("flag_margin", 3.0),
        injury_ahead_usage_gap=raw.get("injury_ahead_usage_gap", 0.15),
        injury_lookback_weeks=raw.get("injury_lookback_weeks", 2),
    )
