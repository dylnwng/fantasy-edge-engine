"""Monte Carlo matchup simulator + FLEX optimizer (Phase 2). ESPN-only --
see roster/matchup.py's get_matchup_source() for why. Simulates the
user's actual current-week lineup against their actual opponent's actual
lineup, and recommends a FLEX swap if one meaningfully raises win
probability against that specific opponent.

Usage:
    python -m edge_engine.simulation.matchup_cli
    python -m edge_engine.simulation.matchup_cli --week 15
"""

from __future__ import annotations

import argparse

from edge_engine.roster.matchup import MatchupPlayer, get_matchup_source
from edge_engine.simulation.config import load_simulation_config
from edge_engine.simulation.flex_optimizer import find_best_flex_lineup
from edge_engine.simulation.monte_carlo import simulate_matchup
from edge_engine.simulation.projections import build_projections, make_projection_fn

_NON_STARTER_SLOTS = ("BE", "IR")


def _starters(lineup: list[MatchupPlayer]) -> list[MatchupPlayer]:
    return [mp for mp in lineup if mp.player.slot_position not in _NON_STARTER_SLOTS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--week", type=int, default=None,
        help="week to simulate (default: the live current week). Also useful for "
             "checking what the right call was on a past week.",
    )
    args = parser.parse_args()

    sim_config = load_simulation_config()
    source = get_matchup_source()
    try:
        snapshot = source.get_current_matchup(week=args.week)
    except Exception as e:
        # get_current_matchup() raises a clean ValueError for the case it
        # recognizes (e.g. a bye week in an odd-team league), but a QA
        # pass confirmed espn_api's own box_scores() can also raise a raw
        # KeyError deep inside its own parsing when asked about a week
        # with no real matchup data yet (e.g. before the season starts) --
        # not something worth enumerating every possible library-internal
        # exception for. Raised as a plain RuntimeError, not SystemExit:
        # weekly.py imports and calls main() directly and needs to catch
        # this with a normal `except Exception` to skip just this step --
        # SystemExit inherits from BaseException, not Exception, and
        # would otherwise escape that catch and kill the whole weekly
        # run instead of just skipping the matchup section (a real
        # regression caught by re-running weekly.py end to end after
        # this fix). The __main__ block below is the only place this
        # becomes a clean process exit, for direct CLI invocation.
        raise RuntimeError(f"Could not load the matchup for this week: {e}") from e

    print(f"Week {snapshot.week}: {snapshot.my_team_name} vs {snapshot.opponent_team_name}")
    print()

    season = source.get_league_config().season
    my_starters = _starters(snapshot.my_lineup)
    opponent_starters = _starters(snapshot.opponent_lineup)

    my_projections = build_projections(my_starters, season, snapshot.week, sim_config)
    opponent_projections = build_projections(opponent_starters, season, snapshot.week, sim_config)

    result = simulate_matchup(
        my_projections, opponent_projections, n_sims=sim_config.n_sims,
        distribution=sim_config.distribution, team_correlation=sim_config.team_correlation,
    )
    print(
        f"Win probability: {result.my_win_probability:.1%}  "
        f"(median {result.my_score_p50:.1f} vs {result.opponent_mean_score:.1f}, "
        f"range {result.my_score_p10:.1f}-{result.my_score_p90:.1f})"
    )

    project_fn = make_projection_fn(season, snapshot.week, sim_config)
    recommendation = find_best_flex_lineup(
        snapshot.my_lineup,
        opponent_projections,
        project_fn,
        source.get_league_config().lineup_slots,
        n_sims=sim_config.n_sims,
        win_prob_improvement_threshold=sim_config.win_prob_improvement_threshold,
        distribution=sim_config.distribution, team_correlation=sim_config.team_correlation,
    )

    print()
    if recommendation.improved:
        swap_out = ", ".join(p.name for p in recommendation.swapped_out) or "(none)"
        swap_in = ", ".join(p.name for p in recommendation.swapped_in) or "(none)"
        print(f"Recommended FLEX swap: bench {swap_out} -> start {swap_in}")
        print(
            f"  Win probability: {recommendation.current_lineup_win_probability:.1%} -> "
            f"{recommendation.best_lineup_win_probability:.1%}"
        )
    else:
        print("Current FLEX lineup is already the best option against this week's opponent.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Only converted to a clean process exit here, at the direct-CLI
        # boundary -- main() itself raises plain exceptions so weekly.py
        # (which imports and calls main() directly) can catch and skip
        # just this step instead of the whole run dying.
        raise SystemExit(str(e)) from e
