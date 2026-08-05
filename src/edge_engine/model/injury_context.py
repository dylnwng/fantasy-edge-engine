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
    return InjuryContext(
        has_injury_context=True,
        teammate_name=top["full_name"],
        teammate_status=top["report_status"],
        teammate_injury=top["report_primary_injury"] if pd.notna(top["report_primary_injury"]) else None,
        explanation=explanation,
    )
