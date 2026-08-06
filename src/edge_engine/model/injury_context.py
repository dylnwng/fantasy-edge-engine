"""Requirement 3b: injury context for a usage spike.

Usage data alone can't tell an earned role change from a backup's snap
share rising because the starter ahead of him got hurt. This module
doesn't try to resolve that judgment call -- it surfaces the coincidence
(a same-position teammate with meaningfully higher trailing usage who is
now on the injury report) so the user can weigh it themselves. It never
predicts when the injured starter returns; that's out of scope by design
(see the PRD's requirement 3b acceptance criteria).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from edge_engine.ingestion import raw

REPORTED_STATUSES = {"Questionable", "Doubtful", "Out"}
# Higher number = more likely to be a real, near-term absence.
_STATUS_SEVERITY = {"Out": 3, "Doubtful": 2, "Questionable": 1}

INJURY_COLUMNS = [
    "season",
    "week",
    "team",
    "gsis_id",
    "full_name",
    "position",
    "report_status",
    "report_primary_injury",
]


@dataclass(frozen=True)
class InjuryContext:
    has_injury_context: bool
    teammate_name: str | None
    teammate_status: str | None
    teammate_injury: str | None
    explanation: str
    # "worsening" | "stable" | "improving" | None (not enough reports to
    # compare). Defaulted so existing positional construction elsewhere
    # keeps working unchanged.
    blocker_trajectory: str | None = None


def blocker_status_trajectory(
    injuries: pd.DataFrame, gsis_id: str, season: int, week: int, lookback: int
) -> str | None:
    """Which direction the blocker's *official designation* has moved
    across their two most recent reports in the window: "improving"
    (Out -> Questionable), "worsening" (Questionable -> Out), or
    "stable".

    This is the single most decision-relevant thing requirement 3b was
    missing. Knowing a usage spike is injury-driven doesn't tell you
    whether to spend FAAB on it -- a backup whose blocker is trending
    Out -> Out has a window that's still open, while one whose blocker
    is trending Out -> Questionable has a window that's about to slam
    shut. Same pickup on paper, opposite bets.

    Deliberately does NOT infer recovery from *absence* from the injury
    report. load_injury_reports() only keeps rows carrying a game-status
    designation, so a player missing from a given week might be healthy
    OR might just have had a practice-only report that week -- treating
    absence as "cleared" would manufacture a confident signal out of
    missing data, the same failure mode already fixed in bye_weeks.py.
    Only weeks where the blocker actually carries a designation are
    compared.

    Reports an observed direction; never predicts a return date -- the
    PRD's requirement 3b explicitly scopes that out ("it surfaces the
    injury context, the user applies judgment")."""
    reports = injuries[
        (injuries["season"] == season)
        & (injuries["gsis_id"] == gsis_id)
        & (injuries["week"] <= week)
        & (injuries["week"] >= week - lookback)
    ].sort_values("week")

    if len(reports) < 2:
        return None

    latest = _STATUS_SEVERITY.get(reports["report_status"].iloc[-1])
    previous = _STATUS_SEVERITY.get(reports["report_status"].iloc[-2])
    if latest is None or previous is None:
        return None

    if latest < previous:
        return "improving"
    if latest > previous:
        return "worsening"
    return "stable"


_TRAJECTORY_NOTE = {
    "improving": (
        "Their designation has been improving week over week — the opportunity window "
        "may be closing sooner than the usage trend alone suggests."
    ),
    "worsening": (
        "Their designation has been worsening week over week — the opportunity window "
        "may stay open longer."
    ),
    "stable": "Their designation has been unchanged week over week.",
}


def load_injury_reports(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame:
    """Official weekly injury designations for the given seasons, regular
    season only, restricted to rows that actually carry a game-status
    (most injury-report rows are practice-only and carry no report_status)."""
    frames = [raw.fetch_injuries(season, force_refresh) for season in seasons]
    injuries = pd.concat(frames, ignore_index=True)
    injuries = injuries[injuries["game_type"] == "REG"]
    injuries = injuries[injuries["report_status"].isin(REPORTED_STATUSES)]
    return injuries[INJURY_COLUMNS].reset_index(drop=True)


def _teammates_ahead(
    player_week: pd.DataFrame,
    player_id: str,
    season: int,
    week: int,
    usage_gap: float,
    lookback: int,
) -> pd.Series:
    """gsis_ids of same-team, same-position players whose trailing snap_pct
    (over the `lookback` weeks strictly before `week`) beats the
    candidate's by at least `usage_gap`. Empty if the candidate has no
    row for (season, week) or no teammate clears the gap."""
    candidate_row = player_week[
        (player_week["season"] == season)
        & (player_week["player_id"] == player_id)
        & (player_week["week"] == week)
    ]
    if candidate_row.empty:
        return pd.Series(dtype=float)

    team = candidate_row["team"].iloc[0]
    position = candidate_row["position"].iloc[0]

    prior = player_week[
        (player_week["season"] == season)
        & (player_week["team"] == team)
        & (player_week["position"] == position)
        & (player_week["week"] < week)
        & (player_week["week"] >= week - lookback)
    ]
    trailing_usage = prior.groupby("player_id")["snap_pct"].mean()
    candidate_usage = trailing_usage.get(player_id, 0.0)
    if pd.isna(candidate_usage):
        candidate_usage = 0.0

    ahead = trailing_usage[trailing_usage.index != player_id]
    return ahead[(ahead - candidate_usage) >= usage_gap]


def get_injury_context(
    player_week: pd.DataFrame,
    injuries: pd.DataFrame,
    player_id: str,
    season: int,
    week: int,
    usage_gap: float = 0.15,
    lookback: int = 2,
) -> InjuryContext:
    """Does this player's usage spike at (season, week) coincide with an
    injury designation for a same-position teammate who, before this
    week, had meaningfully higher usage than they did?"""
    ahead_ids = _teammates_ahead(player_week, player_id, season, week, usage_gap, lookback)
    if ahead_ids.empty:
        return InjuryContext(False, None, None, None, "")

    window_injuries = injuries[
        (injuries["season"] == season)
        & (injuries["week"] <= week)
        & (injuries["week"] >= week - lookback)
        & (injuries["gsis_id"].isin(ahead_ids.index))
    ]
    if window_injuries.empty:
        return InjuryContext(False, None, None, None, "")

    ranked = window_injuries.assign(
        severity=window_injuries["report_status"].map(_STATUS_SEVERITY)
    ).sort_values(["severity", "week"], ascending=[False, False])
    top = ranked.iloc[0]

    injury_note = f", {top['report_primary_injury']}" if pd.notna(top["report_primary_injury"]) else ""
    explanation = (
        f"Usage spike coincides with {top['full_name']} "
        f"({top['report_status']}{injury_note}) at the same position, "
        f"who had the higher usage share as of week {int(top['week'])} — "
        "may be an injury-driven opportunity rather than an earned role change."
    )

    trajectory = blocker_status_trajectory(injuries, top["gsis_id"], season, week, lookback)
    if trajectory:
        explanation = f"{explanation} {_TRAJECTORY_NOTE[trajectory]}"

    return InjuryContext(
        has_injury_context=True,
        teammate_name=top["full_name"],
        teammate_status=top["report_status"],
        teammate_injury=top["report_primary_injury"] if pd.notna(top["report_primary_injury"]) else None,
        explanation=explanation,
        blocker_trajectory=trajectory,
    )
