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
        # A real NFL team has exactly one bye per season, once the full
        # schedule is published -- so exactly one missing week is the
        # only case worth trusting. More than one missing week means the
        # schedule data for this team is incomplete (a partial scrape, or
        # a season whose schedule hasn't been fully published yet), not
        # that the team has multiple byes; `min(missing)` used to pick an
        # arbitrary one of those weeks and report it as THE bye,
        # fabricating a specific, wrong answer from data the function
        # doesn't actually have. Better to have no answer than a
        # confidently wrong one -- downstream (roster_fit.py) already
        # treats a missing bye_week as "no collision possible", so
        # omitting the team here degrades safely.
        if len(missing) == 1:
            byes[team] = min(missing)
    return byes
