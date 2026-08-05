"""Monte Carlo matchup simulation.

Known v1 simplifications, stated up front rather than silently assumed
away:
  - Each player's points are drawn from a normal distribution. This
    doesn't model the real right-skew of a fantasy scoring distribution
    (a ceiling game is more likely than a symmetric normal implies) --
    reasonable for a v1 given an empirical-std input, standard in public
    fantasy sim tools, but a known approximation.
  - No floor at 0. An earlier version of this clipped every draw at 0 on
    the assumption fantasy points "can't go negative in practice" -- that
    assumption is simply wrong for D/ST scoring (a bad defensive week is
    routinely a negative score under standard scoring), confirmed
    against real 2024 data where a finalized D/ST score of -4.0 was
    getting silently floored to 0.0, inflating that side's total by
    exactly the amount clipped away. Not clipping means a low-mean,
    high-std skill-position player can occasionally simulate a small
    negative score too -- a much smaller and rarer approximation error
    than systematically erasing real negative outcomes.
  - Players are drawn independently -- no correlation modeling (e.g. two
    of your RBs both having a good or bad week together because their
    offense as a whole did).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from edge_engine.simulation.projections import PlayerProjection


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


def simulate_side(projections: list[PlayerProjection], n_sims: int, rng: np.random.Generator) -> np.ndarray:
    """Public (not just an internal helper of simulate_matchup): the FLEX
    optimizer needs to draw one side's totals independently, reusing the
    opponent's draw across many candidate lineups (common random numbers)."""
    if not projections:
        return np.zeros(n_sims)
    means = np.array([p.mean for p in projections])
    stds = np.array([p.std for p in projections])
    draws = rng.normal(loc=means, scale=stds, size=(n_sims, len(projections)))
    return draws.sum(axis=1)


def simulate_matchup(
    my_projections: list[PlayerProjection],
    opponent_projections: list[PlayerProjection],
    n_sims: int = 10_000,
    rng: np.random.Generator | None = None,
) -> MatchupSimResult:
    """my_projections/opponent_projections should already be filtered to
    starters only -- this function is slot-agnostic, it just sums
    whatever it's handed."""
    rng = rng or np.random.default_rng()
    my_totals = simulate_side(my_projections, n_sims, rng)
    opp_totals = simulate_side(opponent_projections, n_sims, rng)

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
