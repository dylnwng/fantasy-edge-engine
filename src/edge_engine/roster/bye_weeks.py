"""Team bye weeks, computed from the published season schedule rather than
entered manually — they're public and don't change once released.
"""

from __future__ import annotations

from functools import lru_cache

from edge_engine.roster import nflverse_ref


@lru_cache(maxsize=None)
def get_bye_weeks(season: int, force_refresh: bool = False) -> dict[str, int]:
    """team_abbr -> bye week number, for one season."""
    sched = nflverse_ref.fetch_schedules(season, force_refresh)
    reg = sched[sched["game_type"] == "REG"]

    all_weeks = set(reg["week"].unique())
    teams = set(reg["home_team"]) | set(reg["away_team"])

    byes: dict[str, int] = {}
    for team in teams:
        played = set(reg.loc[(reg["home_team"] == team) | (reg["away_team"] == team), "week"])
        missing = all_weeks - played
        if missing:
            byes[team] = min(missing)
    return byes
