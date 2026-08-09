"""Team bye weeks, computed from the published season schedule rather than
entered manually — they're public and don't change once released.
"""

from __future__ import annotations

import logging

from edge_engine.roster import nflverse_ref

logger = logging.getLogger(__name__)


def get_bye_weeks(season: int, force_refresh: bool = False) -> dict[str, int]:
    """team_abbr -> bye week number, for one season.

    Deliberately not @lru_cache'd (an earlier version was): the schedule
    fetch below already caches at the file level via nflverse_ref's
    cached_fetch, so an in-memory cache here bought nothing but a real
    staleness risk -- a long-running process (the Streamlit dashboard)
    would keep serving a schedule correction or a since-force_refresh'd
    update from before its first call in that process's life,
    indefinitely. Every call site here invokes this once per method call
    (not per player), so recomputing this small groupby each time is
    trivial -- there was no real performance case for the in-memory
    cache to begin with."""
    try:
        sched = nflverse_ref.fetch_schedules(season, force_refresh)
    except RuntimeError as e:
        # The schedule fetch can fail for reasons that have nothing to do
        # with this code: nflverse (or habitatring.com, which
        # nfl_data_py's import_schedules actually reads) being down or
        # rate-limiting, no connectivity, or a season whose schedule
        # genuinely hasn't been published yet.
        #
        # Whatever the cause, the result is the SAME epistemic state as
        # the incomplete-schedule case handled below -- we don't know the
        # byes -- and it degrades the same way: every caller reads this
        # with byes.get(team), and roster_fit already treats a missing
        # bye_week as "no collision possible". Raising instead meant one
        # transient fetch failure took down the entire Streamlit
        # dashboard with a raw traceback and blocked every roster read in
        # both roster sources, over optional context that by design never
        # moves predicted_score.
        #
        # Surfaced as a warning rather than swallowed, matching how
        # weekly.py already turns an unreachable nflverse into a clear
        # message instead of a traceback. Narrowly scoped to RuntimeError,
        # which is what nflverse_cache.cached_fetch raises for any fetch
        # failure -- a genuine bug in the parsing below still propagates.
        logger.warning(
            "No bye weeks available for %s (%s) -- continuing without them. "
            "Bye-week collision notes will be absent until nflverse publishes "
            "the schedule.",
            season, e,
        )
        return {}

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
