"""Monte Carlo matchup simulation.

Two knobs, both config-driven (SimulationConfig) and defaulted to the
already-validated v1 behavior until re-validated (see
EVALUATION.md's "Matchup simulator calibration" section and
scripts/calibration_test.py):

  - `distribution`: "normal" (default, validated) or "gamma". Gamma is a
    moment-matched (mean, std) right-skewed, non-negative distribution
    -- closer to how fantasy scores actually behave (a ceiling game is
    more likely than a symmetric normal implies) -- computed with numpy's
    native `rng.gamma`, no scipy dependency. A player with std=0
    (deterministic: bye, already-final game, or a zero-variance position
    like K/D-ST) always returns exactly their mean regardless of
    `distribution` -- computed directly, never routed through the gamma
    formula (`rng.gamma(shape, scale=0)` silently returns exactly 0
    regardless of shape, which would corrupt every deterministic score).
    A player with a real std but mean<=0 (Gamma has no support there --
    in practice this should essentially never fire, since the one
    position that can score negative, D/ST, always has std=0) falls back
    to the normal draw.
  - `team_correlation` (0.0-1.0, default 0.0): induces same-team
    correlation via a shared per-team "game script" shock `Z_T ~ N(0,1)`,
    reused across every teammate in a simulation run: `draws_i = mean_i +
    sqrt(1-c)*(raw_i-mean_i) + sqrt(c)*std_i*Z_T`. This is affine in
    `raw_i` and `Z_T`, so regardless of `raw_i`'s distribution it
    preserves each player's own (mean, std) *exactly* while introducing
    real cross-teammate correlation (Corr(i,j) = c for two teammates
    sharing Z_T) -- see scripts/measure_team_correlation.py for how `c`
    should be set (measured from real data, not invented).

Known v1 simplifications still true regardless of these settings:
  - No cross-side correlation (e.g. weather affecting both offenses).
  - No floor at 0 (an earlier version clipped every draw at 0 on the
    assumption fantasy points "can't go negative in practice" -- wrong
    for D/ST scoring specifically; confirmed against real 2024 data
    where a finalized D/ST score of -4.0 was getting silently floored to
    0.0). A low-mean, high-std skill-position player can occasionally
    simulate a small negative score too under "normal" -- a much smaller
    and rarer approximation error than systematically erasing real
    negative outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from edge_engine.simulation.projections import PlayerProjection

Distribution = Literal["normal", "gamma"]

# Below this, a player is treated as fully deterministic regardless of
# `distribution` -- real std values from variance.py are never exactly
# 0.0 from floating-point noise, so this is a hygiene floor, not a
# meaningful threshold choice.
_MIN_STOCHASTIC_STD = 1e-9


@dataclass(frozen=True)
class MatchupSimResult:
    n_sims: int
    my_win_probability: float
    tie_probability: float
    my_mean_score: float
    opponent_mean_score: float
    my_score_p10: float
    my_score_p50: float
    my_score_p90: float


def _draw_independent(
    means: np.ndarray, stds: np.ndarray, n_sims: int, rng: np.random.Generator, distribution: Distribution
) -> np.ndarray:
    """Each player's own independent draw, shape (n_sims, n_players),
    before any team-correlation mixing. A std<=_MIN_STOCHASTIC_STD player
    always returns exactly `mean`, regardless of `distribution`."""
    draws = np.tile(means, (n_sims, 1)).astype(float)
    stochastic = stds > _MIN_STOCHASTIC_STD
    if not stochastic.any():
        return draws

    if distribution == "gamma":
        gamma_eligible = stochastic & (means > 0)
        if gamma_eligible.any():
            m, s = means[gamma_eligible], stds[gamma_eligible]
            shape = (m**2) / (s**2)
            scale = (s**2) / m
            draws[:, gamma_eligible] = rng.gamma(shape=shape, scale=scale, size=(n_sims, gamma_eligible.sum()))

        # mean<=0 with real std -- Gamma has no support there. Falls
        # back to normal (allows negative values). Should be rare: the
        # one position that can score negative (D/ST) always has
        # std=0 today, so this is a safety net, not the common path.
        normal_fallback = stochastic & (means <= 0)
        if normal_fallback.any():
            draws[:, normal_fallback] = rng.normal(
                loc=means[normal_fallback], scale=stds[normal_fallback], size=(n_sims, normal_fallback.sum())
            )
    else:
        draws[:, stochastic] = rng.normal(
            loc=means[stochastic], scale=stds[stochastic], size=(n_sims, stochastic.sum())
        )
    return draws


def _apply_team_correlation(
    draws: np.ndarray,
    means: np.ndarray,
    stds: np.ndarray,
    teams: list[str],
    n_sims: int,
    rng: np.random.Generator,
    team_correlation: float,
) -> np.ndarray:
    """draws_i = mean_i + sqrt(1-c)*(raw_i-mean_i) + sqrt(c)*std_i*Z_T,
    where Z_T is one shared N(0,1) draw per team per simulation run.
    Preserves each player's own (mean, std) exactly by construction --
    see module docstring for the derivation."""
    if team_correlation <= 0.0:
        return draws

    unique_teams = sorted(set(teams))
    team_index = {t: i for i, t in enumerate(unique_teams)}
    team_shocks = rng.normal(0.0, 1.0, size=(n_sims, len(unique_teams)))
    player_team_cols = np.array([team_index[t] for t in teams])
    z_per_player = team_shocks[:, player_team_cols]  # (n_sims, n_players)

    c = team_correlation
    return means + np.sqrt(1 - c) * (draws - means) + np.sqrt(c) * stds * z_per_player


def simulate_side(
    projections: list[PlayerProjection],
    n_sims: int,
    rng: np.random.Generator,
    distribution: Distribution = "normal",
    team_correlation: float = 0.0,
) -> np.ndarray:
    """Public (not just an internal helper of simulate_matchup): the FLEX
    optimizer needs to draw one side's totals independently, reusing the
    opponent's draw across many candidate lineups (common random numbers)."""
    if not projections:
        return np.zeros(n_sims)
    means = np.array([p.mean for p in projections])
    stds = np.array([p.std for p in projections])
    teams = [p.team for p in projections]

    raw = _draw_independent(means, stds, n_sims, rng, distribution)
    mixed = _apply_team_correlation(raw, means, stds, teams, n_sims, rng, team_correlation)
    return mixed.sum(axis=1)


def simulate_matchup(
    my_projections: list[PlayerProjection],
    opponent_projections: list[PlayerProjection],
    n_sims: int = 10_000,
    rng: np.random.Generator | None = None,
    distribution: Distribution = "normal",
    team_correlation: float = 0.0,
) -> MatchupSimResult:
    """my_projections/opponent_projections should already be filtered to
    starters only -- this function is slot-agnostic, it just sums
    whatever it's handed."""
    rng = rng or np.random.default_rng()
    my_totals = simulate_side(my_projections, n_sims, rng, distribution, team_correlation)
    opp_totals = simulate_side(opponent_projections, n_sims, rng, distribution, team_correlation)

    wins = int((my_totals > opp_totals).sum())
    ties = int((my_totals == opp_totals).sum())

    return MatchupSimResult(
        n_sims=n_sims,
        my_win_probability=wins / n_sims,
        tie_probability=ties / n_sims,
        my_mean_score=float(my_totals.mean()),
        opponent_mean_score=float(opp_totals.mean()),
        my_score_p10=float(np.percentile(my_totals, 10)),
        my_score_p50=float(np.percentile(my_totals, 50)),
        my_score_p90=float(np.percentile(my_totals, 90)),
    )
