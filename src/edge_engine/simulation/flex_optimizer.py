"""Brute-force FLEX-slot lineup optimization.

Not a real ILP/MILP solver, by deliberate choice: the decision space is
small (a handful of FLEX-eligible bench players competing for 1-2
flexible slots -- realistically at most C(8,2)=28 candidate lineups),
so exhaustive enumeration is provably optimal for this problem size and
needs no new solver dependency.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Callable

import numpy as np

from edge_engine.ranking.roster_fit import FLEX_ELIGIBLE_POSITIONS
from edge_engine.roster.matchup import MatchupPlayer
from edge_engine.roster.models import Player
from edge_engine.simulation.monte_carlo import Distribution, simulate_side
from edge_engine.simulation.projections import PlayerProjection

_NON_STARTER_SLOTS = ("FLEX", "BE", "IR")


def _player_key(mp: MatchupPlayer) -> str:
    # player_id can be None for an unresolved player -- fall back to name
    # so distinct unresolved players don't collide as "the same player".
    return mp.player.player_id or mp.player.name


def enumerate_flex_candidates(my_lineup: list[MatchupPlayer], lineup_slots: dict[str, int]) -> list[list[MatchupPlayer]]:
    """Every valid alternative starting lineup obtainable by choosing
    which FLEX-eligible player(s) (currently in FLEX, or on the bench)
    fill the league's FLEX slot(s), holding every other starter fixed.
    If the league has no FLEX slot, returns just the current starters."""
    non_flex_starters = [mp for mp in my_lineup if mp.player.slot_position not in _NON_STARTER_SLOTS]
    # Clamped at 0: a malformed lineup_slots value (the same class of bug
    # fixed in roster_fit.py's position_need) would otherwise reach
    # itertools.combinations(flex_pool, n_flex) below with a negative r,
    # which raises its own unhelpful "ValueError: r must be non-negative"
    # -- a slot count can never be genuinely negative, so treating it the
    # same as "no FLEX slot" is the correct floor, not a guess.
    n_flex = max(0, lineup_slots.get("FLEX", 0))
    if n_flex == 0:
        return [non_flex_starters]

    flex_pool = [
        mp for mp in my_lineup
        if mp.player.slot_position in ("FLEX", "BE") and mp.player.position in FLEX_ELIGIBLE_POSITIONS
    ]
    if len(flex_pool) < n_flex:
        # itertools.combinations(flex_pool, n_flex) silently yields nothing
        # when the pool is smaller than n_flex, which left find_best_flex_
        # lineup() with an empty `scored` list and crashed on scored[0]
        # with a bare IndexError -- confirmed with a 2-FLEX league and a
        # thin bench (only 1 eligible player). Fail with a message that
        # actually explains what's short, instead.
        raise ValueError(
            f"Can't fill {n_flex} FLEX slot(s): only {len(flex_pool)} FLEX-eligible "
            f"bench player(s) available ({FLEX_ELIGIBLE_POSITIONS})."
        )
    return [non_flex_starters + list(combo) for combo in itertools.combinations(flex_pool, n_flex)]


@dataclass(frozen=True)
class FlexRecommendation:
    current_lineup_win_probability: float
    best_lineup_win_probability: float
    swapped_in: list[Player]
    swapped_out: list[Player]
    improved: bool


def find_best_flex_lineup(
    my_lineup: list[MatchupPlayer],
    opponent_starter_projections: list[PlayerProjection],
    project_fn: Callable[[MatchupPlayer], PlayerProjection],
    lineup_slots: dict[str, int],
    n_sims: int = 10_000,
    win_prob_improvement_threshold: float = 0.01,
    rng: np.random.Generator | None = None,
    distribution: Distribution = "normal",
    team_correlation: float = 0.0,
) -> FlexRecommendation:
    rng = rng or np.random.default_rng()
    candidates = enumerate_flex_candidates(my_lineup, lineup_slots)

    # Common random numbers: the opponent's totals are drawn ONCE and reused
    # across every candidate lineup -- a real variance-reduction technique,
    # not just an optimization. Without it, two lineups with genuinely equal
    # win probability could rank differently purely from independent MC
    # sampling noise between separate opponent draws.
    opponent_totals = simulate_side(opponent_starter_projections, n_sims, rng, distribution, team_correlation)

    current_flex_keys = {_player_key(mp) for mp in my_lineup if mp.player.slot_position == "FLEX"}

    scored = []
    for candidate in candidates:
        my_totals = simulate_side([project_fn(mp) for mp in candidate], n_sims, rng, distribution, team_correlation)
        win_prob = float((my_totals > opponent_totals).mean())
        flex_keys = {_player_key(mp) for mp in candidate if mp.player.slot_position in ("FLEX", "BE")}
        scored.append((candidate, win_prob, flex_keys))

    current = next((c for c in scored if c[2] == current_flex_keys), scored[0])
    best = max(scored, key=lambda c: c[1])

    swapped_out_keys = current[2] - best[2]
    swapped_in_keys = best[2] - current[2]
    swapped_out = [mp.player for mp in my_lineup if _player_key(mp) in swapped_out_keys]
    swapped_in = [mp.player for mp in best[0] if _player_key(mp) in swapped_in_keys]

    return FlexRecommendation(
        current_lineup_win_probability=current[1],
        best_lineup_win_probability=best[1],
        swapped_in=swapped_in,
        swapped_out=swapped_out,
        improved=(best[1] - current[1]) > win_prob_improvement_threshold,
    )
