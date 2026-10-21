"""How unresolved players get reported.

Unresolved players are always surfaced, never silently dropped -- that's
a standing requirement. But *what to tell the user about them* depends
entirely on where the roster data came from, and getting it wrong sends
people hunting for a bug that doesn't exist.

With the manual CSV source the user typed the names, so "check the
spelling" is actionable. With the live ESPN connector it is not: the
names came from ESPN's own API, there is no spelling of the user's to
correct, and most are practice-squad or deep-bench players nflverse
carries no usage data for at all. A real week-7 run reported 20 of them
and told the user to go check an export that does not exist.
"""

from __future__ import annotations

import os

from edge_engine.roster.models import Player

# ESPN returns no team for some free agents, which guarantees the
# team-disambiguation step can't run. A gap in the source data, not
# something the user can fix from either source.
_NO_TEAM = {None, "", "None", "FA"}


def active_roster_source() -> str:
    return os.environ.get("EDGE_ENGINE_ROSTER_SOURCE", "manual").lower()


def unresolved_headline(
    unresolved: list[Player], kind: str = "free agent", source: str | None = None
) -> str:
    """One line introducing the unresolved list, phrased for the source
    actually in use."""
    source = source or active_roster_source()
    count = len(unresolved)
    no_team = sum(1 for p in unresolved if p.team in _NO_TEAM)

    if source == "espn":
        head = (
            f"{count} {kind}(s) couldn't be matched to usage data and are excluded above. "
            "Nothing for you to fix here — the names come straight from ESPN, and most are "
            "practice-squad or deep-bench players nflverse has no usage data for:"
        )
    else:
        head = (
            f"{count} {kind}(s) did not resolve and are excluded above — "
            "check the spelling and team in your CSV:"
        )

    if no_team:
        head += (
            f"\n  ({no_team} of these arrived with no team, so they can't be told apart "
            "from a same-named player. That's a gap in the source data.)"
        )
    return head
