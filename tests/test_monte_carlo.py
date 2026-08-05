import math

import numpy as np
import pytest

from edge_engine.simulation.monte_carlo import simulate_matchup
from edge_engine.simulation.projections import PlayerProjection


def _proj(name, mean, std):
    return PlayerProjection(name, name, "RB", mean, std, "model", "")


def test_seeded_run_is_reproducible():
    my = [_proj("A", 15.0, 5.0)]
    opp = [_proj("B", 12.0, 5.0)]

    r1 = simulate_matchup(my, opp, n_sims=5000, rng=np.random.default_rng(42))
    r2 = simulate_matchup(my, opp, n_sims=5000, rng=np.random.default_rng(42))

    assert r1.my_win_probability == r2.my_win_probability
    assert r1.my_mean_score == r2.my_mean_score


def test_win_probability_matches_closed_form():
    # No clipping happens anywhere in simulate_matchup, so this is exact
    # (up to Monte Carlo sampling noise), not just "negligible when the
    # clip rarely fires": my_total - opp_total ~ Normal(m1-m2,
    # s1^2+s2^2), so P(win) = Phi((m1-m2) / sqrt(s1^2+s2^2)), computed
    # via stdlib erf (no new scipy dependency). Large n_sims keeps this
    # well within a tolerance many multiples of the MC standard error --
    # a real statistical check, not a tautology, without being flaky.
    my = [_proj("A", 60.0, 8.0), _proj("B", 40.0, 6.0)]
    opp = [_proj("C", 55.0, 7.0), _proj("D", 42.0, 6.0)]

    m1, s1 = 100.0, math.sqrt(8.0**2 + 6.0**2)
    m2, s2 = 97.0, math.sqrt(7.0**2 + 6.0**2)
    z = (m1 - m2) / math.sqrt(s1**2 + s2**2)
    expected = 0.5 * (1 + math.erf(z / math.sqrt(2)))

    result = simulate_matchup(my, opp, n_sims=200_000, rng=np.random.default_rng(0))
    assert result.my_win_probability == pytest.approx(expected, abs=0.01)


def test_a_deterministic_negative_score_is_not_floored_at_zero():
    # Real bug caught against actual 2024 ESPN data: a finalized D/ST
    # score of -4.0 was getting silently clipped to 0.0 by an earlier
    # version of this function, inflating that side's total by exactly
    # the amount clipped away. D/ST scoring routinely goes negative
    # under standard scoring -- there is no floor at 0 in real fantasy
    # scoring, so there must not be one here either.
    my = [_proj("A", 20.0, 0.0), _proj("B", -4.0, 0.0)]  # a real D/ST-shaped final score
    opp = [_proj("C", 10.0, 0.0)]

    result = simulate_matchup(my, opp, n_sims=100, rng=np.random.default_rng(0))
    assert result.my_mean_score == pytest.approx(16.0)  # 20.0 + (-4.0), not 20.0 + 0.0


def test_high_variance_low_mean_can_simulate_negative():
    # A known, accepted v1 simplification (documented in the module
    # docstring): without a floor, a low-mean/high-std skill-position
    # player can occasionally draw a small negative score. Confirms the
    # simplification actually behaves as documented, not accidentally
    # re-clipped somewhere.
    my = [_proj("A", 1.0, 10.0)]
    opp = [_proj("B", 1.0, 10.0)]

    result = simulate_matchup(my, opp, n_sims=5000, rng=np.random.default_rng(0))
    assert result.my_score_p10 < 0.0


def test_std_zero_is_deterministic():
    my = [_proj("A", 20.0, 0.0)]
    opp = [_proj("B", 15.0, 0.0)]

    result = simulate_matchup(my, opp, n_sims=1000, rng=np.random.default_rng(0))
    assert result.my_mean_score == 20.0
    assert result.my_score_p10 == result.my_score_p90 == 20.0
    assert result.my_win_probability == 1.0


def test_empty_side_scores_zero():
    result = simulate_matchup([], [_proj("B", 10.0, 2.0)], n_sims=1000, rng=np.random.default_rng(0))
    assert result.my_mean_score == 0.0
    assert result.my_win_probability == 0.0
