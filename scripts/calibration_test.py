"""Calibration test: does the matchup simulator's predicted win probability
actually track real outcomes, across the WHOLE league's real 2024 season
(not just one team), with the "already final" deterministic shortcut
stripped out so this genuinely exercises the model/variance-based
probabilistic path -- not just a season that's over and thus degenerate
(without stripping "final", every player from a completed season
resolves deterministically and every simulated outcome collapses to
exactly 0% or 100%, which is *not* a test of the probabilistic engine).

Requires EDGE_ENGINE_ROSTER_SOURCE=espn and real ESPN credentials in
.env (see .env.example) -- this hits the live ESPN API once per week.

Usage:
    python scripts/calibration_test.py
"""

from __future__ import annotations

import dataclasses
import math

from dotenv import load_dotenv
load_dotenv()

from edge_engine.roster.bye_weeks import get_bye_weeks
from edge_engine.roster.espn_source import EspnRosterStateSource
from edge_engine.roster.matchup import MatchupPlayer
from edge_engine.simulation.config import load_simulation_config
from edge_engine.simulation.monte_carlo import simulate_matchup
from edge_engine.simulation.projections import make_projection_fn

SEASON = 2024
WEEKS = range(4, 18)  # the trend feature needs 3 played games (not 2) to be non-null -- week<4
                       # leaves only 2 games, one short; 2024 regular season ran through 17
NON_STARTER_SLOTS = ("BE", "IR")


def strip_final(mp: MatchupPlayer) -> MatchupPlayer:
    """Force the pre-game path even though this game (from 2024) has long
    since actually happened -- otherwise every player resolves via the
    deterministic 'final' tier and the simulation is not actually
    probabilistic at all."""
    return dataclasses.replace(mp, game_final=False, actual_points=0.0, on_bye=mp.on_bye)


def main() -> None:
    source = EspnRosterStateSource(year=SEASON)
    league = source._get_league()
    sim_config = load_simulation_config()
    byes = get_bye_weeks(SEASON)

    records = []
    for week in WEEKS:
        box_scores = league.box_scores(week=week)
        project_fn = make_projection_fn(SEASON, week, sim_config)

        for bs in box_scores:
            if bs.home_team is None or bs.away_team is None:
                continue  # bye week in an odd-team league

            home_lineup_raw = [source._map_matchup_player(p, byes) for p in bs.home_lineup]
            away_lineup_raw = [source._map_matchup_player(p, byes) for p in bs.away_lineup]

            home_starters_raw = [mp for mp in home_lineup_raw if mp.player.slot_position not in NON_STARTER_SLOTS]
            away_starters_raw = [mp for mp in away_lineup_raw if mp.player.slot_position not in NON_STARTER_SLOTS]

            actual_home = sum(mp.actual_points for mp in home_starters_raw)
            actual_away = sum(mp.actual_points for mp in away_starters_raw)
            if actual_home == actual_away:
                continue  # ties: exclude from win-probability calibration

            home_stripped = [strip_final(mp) for mp in home_starters_raw]
            away_stripped = [strip_final(mp) for mp in away_starters_raw]

            home_proj = [project_fn(mp) for mp in home_stripped]
            away_proj = [project_fn(mp) for mp in away_stripped]

            result = simulate_matchup(
                home_proj, away_proj, n_sims=sim_config.n_sims,
                distribution=sim_config.distribution, team_correlation=sim_config.team_correlation,
            )

            records.append({
                "week": week,
                "home_team": bs.home_team.team_name,
                "away_team": bs.away_team.team_name,
                "predicted_home_win_prob": result.my_win_probability,
                "actual_home_score": actual_home,
                "actual_away_score": actual_away,
                "home_actually_won": actual_home > actual_away,
            })

    print(f"Total matchup-observations: {len(records)}")
    print()

    # Overall accuracy: did the >50% side actually win?
    correct = sum(1 for r in records if (r["predicted_home_win_prob"] > 0.5) == r["home_actually_won"])
    print(f"Overall pick accuracy: {correct}/{len(records)} = {correct/len(records):.1%}")

    # Brier score (lower is better; 0.25 = always predicting 50/50)
    brier = sum((r["predicted_home_win_prob"] - float(r["home_actually_won"])) ** 2 for r in records) / len(records)
    print(f"Brier score: {brier:.4f}  (0.25 = no-skill baseline of always predicting 50/50, 0 = perfect)")

    # Log loss
    eps = 1e-6
    ll = -sum(
        math.log(min(max(r["predicted_home_win_prob"], eps), 1 - eps)) if r["home_actually_won"]
        else math.log(min(max(1 - r["predicted_home_win_prob"], eps), 1 - eps))
        for r in records
    ) / len(records)
    print(f"Log loss: {ll:.4f}  (0.693 = no-skill baseline of always predicting 50/50)")
    print()

    # Calibration table: bucket predicted probability into deciles, compare to actual win rate
    print("Calibration (predicted probability bucket -> actual win rate in that bucket):")
    buckets = [(i / 10, (i + 1) / 10) for i in range(10)]
    for lo, hi in buckets:
        bucket = [r for r in records if lo <= r["predicted_home_win_prob"] < hi or (hi == 1.0 and r["predicted_home_win_prob"] == 1.0)]
        if not bucket:
            continue
        actual_rate = sum(r["home_actually_won"] for r in bucket) / len(bucket)
        print(f"  [{lo:.1f}-{hi:.1f})  n={len(bucket):3d}  predicted avg={sum(r['predicted_home_win_prob'] for r in bucket)/len(bucket):.2f}  actual win rate={actual_rate:.2f}")


if __name__ == "__main__":
    main()
