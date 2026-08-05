from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from edge_engine.paths import ROOT_DIR

DEFAULT_SIMULATION_CONFIG_PATH = ROOT_DIR / "simulation_config.yaml"


@dataclass(frozen=True)
class SimulationConfig:
    n_sims: int
    variance_window: int
    min_games_for_variance: int
    # Points of simulated win-probability improvement required before
    # recommending a FLEX swap -- mirrors model_config.yaml's flag_margin
    # philosophy: a buffer above Monte Carlo sampling noise, not "any
    # improvement, however small."
    win_prob_improvement_threshold: float
    # Fallback std for positions with real (but sparse) usage history that
    # falls into the espn_projection_only tier -- K/D-ST get 0.0 instead,
    # hardcoded in projections.py, since there's no usage history at all to
    # even approximate a fallback from.
    untracked_position_std: float


def load_simulation_config(path: Path | None = None) -> SimulationConfig:
    path = path or DEFAULT_SIMULATION_CONFIG_PATH
    with open(path) as f:
        raw = yaml.safe_load(f)
    return SimulationConfig(
        n_sims=raw.get("n_sims", 10_000),
        variance_window=raw.get("variance_window", 6),
        min_games_for_variance=raw.get("min_games_for_variance", 3),
        win_prob_improvement_threshold=raw.get("win_prob_improvement_threshold", 0.01),
        untracked_position_std=raw.get("untracked_position_std", 5.0),
    )
