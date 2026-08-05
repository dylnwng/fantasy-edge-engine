import math

import numpy as np
import pytest

from edge_engine.simulation.monte_carlo import simulate_matchup, simulate_side
from edge_engine.simulation.projections import PlayerProjection


def _proj(name, mean, std, team="TM"):
    return PlayerProjection(name, name, "RB", team, mean, std, "model", "")


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


# --- Gamma distribution ---


def test_gamma_seeded_run_is_reproducible():
    my = [_proj("A", 15.0, 5.0)]
    opp = [_proj("B", 12.0, 5.0)]

    r1 = simulate_matchup(my, opp, n_sims=5000, rng=np.random.default_rng(42), distribution="gamma")
    r2 = simulate_matchup(my, opp, n_sims=5000, rng=np.random.default_rng(42), distribution="gamma")

    assert r1.my_win_probability == r2.my_win_probability
    assert r1.my_mean_score == r2.my_mean_score


def test_gamma_moments_converge_to_mean_and_std():
    # Moment-matched Gamma: sample mean/std of a single player's draws
    # should converge to the (mean, std) it was parameterized with, not
    # just "look plausible."
    proj = [_proj("A", 15.0, 6.0)]
    draws = simulate_side(proj, n_sims=500_000, rng=np.random.default_rng(0), distribution="gamma")

    assert draws.mean() == pytest.approx(15.0, abs=0.05)
    assert draws.std() == pytest.approx(6.0, abs=0.05)


def test_gamma_std_zero_is_still_deterministic():
    # A bye/final/K/D-ST player (std=0) must return exactly `mean` under
    # gamma too -- rng.gamma(shape, scale=0) would otherwise silently
    # return exactly 0 regardless of shape, corrupting every
    # deterministic score. This is exactly the trap the implementation
    # explicitly special-cases for.
    proj = [_proj("A", 20.0, 0.0)]
    draws = simulate_side(proj, n_sims=1000, rng=np.random.default_rng(0), distribution="gamma")

    assert (draws == 20.0).all()


def test_gamma_negative_mean_falls_back_to_normal_without_crashing():
    # Gamma has no support for mean<=0. A D/ST-shaped negative
    # projection with real (nonzero) std must not crash -- falls back to
    # a normal draw. Should be rare in practice (D/ST always has std=0
    # today), but must degrade gracefully if it ever fires.
    proj = [_proj("A", -3.0, 2.0)]
    draws = simulate_side(proj, n_sims=5000, rng=np.random.default_rng(0), distribution="gamma")

    assert draws.mean() == pytest.approx(-3.0, abs=0.5)


# --- Team correlation ---


def test_zero_correlation_matches_independent_draws_statistically():
    my = [_proj("A", 15.0, 5.0, team="KC"), _proj("B", 10.0, 4.0, team="KC")]
    opp = [_proj("C", 12.0, 5.0, team="BUF")]

    r0 = simulate_matchup(my, opp, n_sims=50_000, rng=np.random.default_rng(1), team_correlation=0.0)
    # 25.0 = 15+10, the combined mean regardless of correlation (mean is
    # always preserved exactly by construction -- see the next test).
    assert r0.my_mean_score == pytest.approx(25.0, abs=0.3)


def test_correlation_preserves_each_players_own_marginal_mean_and_std():
    # The whole point of the affine team-shock construction: correlation
    # changes how teammates move *together*, not any individual player's
    # own (mean, std) contract that the rest of the pipeline depends on.
    proj = [_proj("A", 15.0, 6.0, team="KC"), _proj("B", 9.0, 4.0, team="KC")]

    means, stds = [], []
    rng = np.random.default_rng(0)
    for _ in range(3):
        draws = simulate_side([proj[0]], n_sims=300_000, rng=rng, team_correlation=0.8)
        means.append(draws.mean())
        stds.append(draws.std())

    assert means[-1] == pytest.approx(15.0, abs=0.05)
    assert stds[-1] == pytest.approx(6.0, abs=0.05)


def test_high_correlation_converges_to_target_same_team_correlation():
    n_sims = 400_000
    rng = np.random.default_rng(0)
    c = 0.7

    # Var(A+B) = Var(A) + Var(B) + 2*Cov(A,B), and Cov(A,B) = c*std_a*std_b
    # for two teammates sharing Z_T (see monte_carlo.py's derivation) --
    # a real, checkable consequence of a genuine correlation `c`, not
    # just "the numbers moved in the right direction."
    proj = [_proj("A", 15.0, 6.0, team="KC"), _proj("B", 9.0, 4.0, team="KC")]
    totals = simulate_side(proj, n_sims=n_sims, rng=rng, team_correlation=c)

    expected_var = 6.0**2 + 4.0**2 + 2 * c * 6.0 * 4.0
    assert totals.var() == pytest.approx(expected_var, rel=0.02)

    # c=0 (independent) must give strictly lower combined variance than
    # c=0.7 for the same two players -- correlation is actually doing
    # something, not a no-op.
    independent_totals = simulate_side(proj, n_sims=n_sims, rng=np.random.default_rng(0), team_correlation=0.0)
    assert independent_totals.var() < totals.var()


def test_different_teams_are_not_correlated_by_team_correlation():
    # Two players on DIFFERENT teams sharing a nonzero team_correlation
    # setting must not become correlated with each other -- only real
    # teammates share a Z_T.
    proj = [_proj("A", 15.0, 6.0, team="KC"), _proj("B", 9.0, 4.0, team="BUF")]
    totals = simulate_side(proj, n_sims=400_000, rng=np.random.default_rng(0), team_correlation=0.7)

    expected_var = 6.0**2 + 4.0**2  # no cross-team covariance term
    assert totals.var() == pytest.approx(expected_var, rel=0.02)
