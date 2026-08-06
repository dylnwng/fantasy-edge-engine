from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from edge_engine.paths import ROOT_DIR

DEFAULT_SIMULATION_CONFIG_PATH = ROOT_DIR / "simulation_config.yaml"

_VALID_DISTRIBUTIONS = {"normal", "gamma"}


def _require_number(field: str, value):
    """A second QA pass found that the range checks below crash with a
    raw, confusing TypeError ('<=' not supported between instances of
    ...) if a YAML value is accidentally quoted as a string (e.g.
    `n_sims: "10000"` -- an easy mistake, and YAML's own bool/int
    overlap means `bool` sneaking in here is also worth ruling out
    explicitly, since `True <= 1.0` would otherwise silently pass as 1).
    Catching the wrong type here keeps every config validation failure
    equally clear, not just the ones that happen to be the right type
    but out of range."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"simulation_config.yaml: {field} must be a number, got {value!r} "
            f"({type(value).__name__}) -- check for accidental quotes in the YAML."
        )
    return value


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
    # "normal" (default, validated -- 63.9% pick accuracy / 0.220 Brier
    # against 97 real 2024 matchups, see EVALUATION.md) or "gamma" (a
    # right-skewed, non-negative alternative). Only change this after
    # re-running scripts/calibration_test.py and confirming it holds up --
    # see monte_carlo.py's module docstring.
    distribution: str
    # 0.0 (default, validated) to 1.0: induces same-team correlation in
    # the Monte Carlo draws. Set from scripts/measure_team_correlation.py's
    # real, data-derived measurement -- never a guessed number (same
    # discipline as K/D-ST's std=0 decision in EVALUATION.md).
    team_correlation: float


def load_simulation_config(path: Path | None = None) -> SimulationConfig:
    """Validates the handful of fields whose out-of-range values wouldn't
    just be wrong -- they'd be silently, confidently wrong. A QA pass
    found three real failure modes from unvalidated values: n_sims=0
    crashes simulate_matchup with a raw ZeroDivisionError; team_correlation
    outside [0,1] produces sqrt(negative) internally, silently turning
    every simulated score into NaN while still reporting an exact,
    fabricated 0.0% win probability (NaN comparisons are always False);
    and an unrecognized distribution string (e.g. a "Gamma"/"guassian"
    typo) silently falls through to the Normal-distribution branch with
    no indication the configured value was never read. All three are
    real risks from a hand-edited YAML, not just theoretical -- config
    values should fail loudly here, not corrupt output downstream."""
    path = path or DEFAULT_SIMULATION_CONFIG_PATH
    with open(path) as f:
        raw = yaml.safe_load(f)

    n_sims = _require_number("n_sims", raw.get("n_sims", 10_000))
    if n_sims <= 0:
        raise ValueError(f"simulation_config.yaml: n_sims must be positive, got {n_sims}")

    distribution = raw.get("distribution", "normal")
    if distribution not in _VALID_DISTRIBUTIONS:
        raise ValueError(
            f"simulation_config.yaml: distribution={distribution!r} is not recognized "
            f"(expected one of {sorted(_VALID_DISTRIBUTIONS)}) -- a typo here would "
            "otherwise silently fall back to 'normal' with no warning."
        )

    team_correlation = _require_number("team_correlation", raw.get("team_correlation", 0.0))
    if not (0.0 <= team_correlation <= 1.0):
        raise ValueError(
            f"simulation_config.yaml: team_correlation must be between 0.0 and 1.0, "
            f"got {team_correlation} -- see scripts/measure_team_correlation.py for "
            "how to derive a real value."
        )

    return SimulationConfig(
        n_sims=n_sims,
        variance_window=raw.get("variance_window", 6),
        min_games_for_variance=raw.get("min_games_for_variance", 3),
        win_prob_improvement_threshold=raw.get("win_prob_improvement_threshold", 0.01),
        untracked_position_std=raw.get("untracked_position_std", 5.0),
        distribution=distribution,
        team_correlation=team_correlation,
    )
