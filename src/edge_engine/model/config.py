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
    """flag_margin must be >= 0: ranking.output.confidence_tier() computes
    `margin >= 2*flag_margin` / `margin >= flag_margin` to assign High/
    Medium/Low -- a negative flag_margin flips those comparisons around,
    so a player predicted to score *below* their own baseline would
    silently qualify as "High confidence, worth a real FAAB bid" (a QA
    pass confirmed this concretely: confidence_tier(5.0, 10.0, -3.0)
    returns "High"). Caught here, not in confidence_tier itself, since
    this is the one place a hand-edited YAML value enters the system."""
    path = path or DEFAULT_MODEL_CONFIG_PATH
    with open(path) as f:
        raw = yaml.safe_load(f)

    flag_margin = raw.get("flag_margin", 3.0)
    # A second QA pass found the range check below crashes with a raw,
    # confusing TypeError if flag_margin is accidentally quoted as a
    # string in the YAML (e.g. `flag_margin: "3.0"`) -- catch the wrong
    # type explicitly first, same fix as simulation/config.py's
    # _require_number.
    if isinstance(flag_margin, bool) or not isinstance(flag_margin, (int, float)):
        raise ValueError(
            f"model_config.yaml: flag_margin must be a number, got {flag_margin!r} "
            f"({type(flag_margin).__name__}) -- check for accidental quotes in the YAML."
        )
    if flag_margin < 0:
        raise ValueError(f"model_config.yaml: flag_margin must be >= 0, got {flag_margin}")

    train_seasons = raw["train_seasons"]
    validation_season = raw["validation_season"]
    if validation_season in train_seasons:
        # The PRD's whole reason for a held-out SEASON split (not
        # held-out weeks within a season) is to avoid leakage -- a
        # validation_season that's also in train_seasons doesn't crash
        # anything downstream (train.py's train_and_evaluate happily
        # splits on both), it just silently validates the model on data
        # it was trained on, producing an artificially low MAE and
        # invalidating every accuracy number in EVALUATION.md with no
        # warning. A second QA pass found this had no guard at all.
        raise ValueError(
            f"model_config.yaml: validation_season={validation_season} is also in "
            f"train_seasons={train_seasons} -- this defeats the whole point of a "
            "held-out split (the model would be validated on data it trained on)."
        )

    return ModelConfig(
        train_seasons=train_seasons,
        validation_season=validation_season,
        trailing_window=raw.get("trailing_window", 2),
        flag_margin=flag_margin,
        injury_ahead_usage_gap=raw.get("injury_ahead_usage_gap", 0.15),
        injury_lookback_weeks=raw.get("injury_lookback_weeks", 2),
    )
