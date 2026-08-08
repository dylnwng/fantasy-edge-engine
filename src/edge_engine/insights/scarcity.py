"""Positional surplus, and the number nobody computes by hand: the
bye-adjusted *worst week*.

"I'm fine at RB" is usually true for this week and false for week 9, when
two of your three are on bye simultaneously. Current-week surplus hides
that completely. This module reports both, and the worst-week figure is
the headline.

**A documented simplification.** The PRD asks for depth measured against
"the rostered pool in the league" -- what other managers are actually
holding. `RosterStateSource` doesn't expose that: it has your roster and
the free-agent pool, and nothing about other teams (deliberately -- see
roster/matchup.py on why opponent data is a separate, ESPN-only
Protocol). So "startable" is defined here against the league-wide
positional distribution instead: a player at or above the median
predicted score among all scored players at his position. Cruder than
the PRD's definition, available under both roster sources, and recorded
in data_gaps rather than passed off as the real thing.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from edge_engine.roster.models import Player

# Slot labels that aren't real starting requirements.
_NON_STARTING_SLOTS = {"BENCH", "IR"}


@dataclass(frozen=True)
class PositionScarcity:
    position: str
    startable_now: int
    required: int
    surplus_now: int
    worst_week: int | None
    worst_week_surplus: int | None


def _startable_threshold(scored: pd.DataFrame, position: str) -> float | None:
    pool = scored[scored["position"] == position]["predicted_score"]
    if pool.empty:
        return None
    return float(pool.median())


def compute_scarcity(
    rostered: list[Player],
    lineup_slots: dict[str, int],
    scored: pd.DataFrame,
    remaining_weeks: list[int],
) -> tuple[list[PositionScarcity], list[str]]:
    """Returns (per-position scarcity, data_gaps).

    `scored` is the model's current output (needs `player_id`, `position`,
    `predicted_score`). `remaining_weeks` drives the bye-adjusted scan; pass
    an empty list to skip it (and get `worst_week=None`), which is the
    correct behaviour when the schedule is incomplete rather than
    inventing a bye-free future."""
    gaps: list[str] = [
        "Positional depth is measured against the league-wide distribution at each "
        "position (startable = at or above the positional median predicted score), not "
        "against what other managers actually roster -- the roster-state interface "
        "doesn't expose other teams' rosters."
    ]

    scored_by_id = dict(zip(scored["player_id"], scored["predicted_score"]))
    results: list[PositionScarcity] = []

    positions = {p.position for p in rostered} | {
        slot for slot in lineup_slots if slot.upper() not in _NON_STARTING_SLOTS and slot != "FLEX"
    }

    unscored = [p.name for p in rostered if p.player_id not in scored_by_id]
    if unscored:
        gaps.append(
            f"{len(unscored)} rostered player(s) have no model score this week and are "
            f"counted as not-startable: {', '.join(sorted(unscored)[:5])}"
            + (" ..." if len(unscored) > 5 else "")
        )

    unscored_positions: list[str] = []

    for position in sorted(positions):
        threshold = _startable_threshold(scored, position)
        required = max(0, lineup_slots.get(position, 0))

        at_pos = [p for p in rostered if p.position == position]
        if threshold is None:
            startable = []
            unscored_positions.append(position)
        else:
            startable = [
                p for p in at_pos
                if p.player_id in scored_by_id and scored_by_id[p.player_id] >= threshold
            ]

        worst_week, worst_surplus = None, None
        for wk in remaining_weeks:
            # A player on bye that week simply isn't available -- not a
            # reduced contributor, an absent one.
            available = [p for p in startable if p.bye_week != wk]
            surplus = len(available) - required
            if worst_surplus is None or surplus < worst_surplus:
                worst_week, worst_surplus = wk, surplus

        results.append(
            PositionScarcity(
                position=position,
                startable_now=len(startable),
                required=required,
                surplus_now=len(startable) - required,
                worst_week=worst_week,
                worst_week_surplus=worst_surplus,
            )
        )

    if scored.empty:
        # Distinct from "this position isn't covered": the model returned
        # nothing at all, usually because the requested week is too early
        # in the season for any player to have a full trailing window.
        # Saying "the model does not score RB/WR/TE" here would be a
        # flatly false explanation of a timing problem.
        gaps.append(
            "The opportunity model returned no scores for this week at all (typically too "
            "early in the season for a full trailing window), so every startable count "
            "below is 0 and no surplus figure is meaningful."
        )
    elif unscored_positions:
        # Expected, not a malfunction: the opportunity model covers
        # pass-catchers and ball-carriers, so QB/K/DST have no scored pool
        # to set a startable threshold from (see EVALUATION.md's QB scope
        # gap). Stated once as a scope boundary rather than once per
        # position, which reads like three separate failures.
        gaps.append(
            f"The opportunity model does not score {', '.join(unscored_positions)}, so "
            "startable counts at those positions are 0 and their surplus is not meaningful."
        )

    return results, gaps


def remaining_weeks(current_week: int, byes_known: bool, final_week: int = 17) -> tuple[list[int], list[str]]:
    """Weeks left to scan for bye collisions. Returns an empty list plus a
    gap note when bye data is unavailable, rather than scanning a schedule
    it can't actually see -- same discipline as bye_weeks.py refusing to
    infer a bye from an incomplete schedule."""
    if not byes_known:
        return [], [
            "Bye weeks unavailable for this season (schedule not published or incomplete); "
            "bye-adjusted worst-week scarcity could not be computed."
        ]
    weeks = [w for w in range(max(1, current_week), final_week + 1)]
    if not weeks:
        return [], [
            f"Week {current_week} is at or past the end of the fantasy regular season "
            f"(week {final_week}); there are no remaining weeks to bye-adjust against, so "
            "worst-week scarcity shows n/a."
        ]
    return weeks, []
