"""Official, non-injury roster availability -- suspended, physically
unable to perform, commissioner exempt list, reserve. Deliberately
separate from the injury-report system (model/injury_context.py,
requirement 3b): that's sourced from the NFL's official weekly injury
reports (Questionable/Doubtful/Out); this is sourced from nflverse's
player reference table's own `status` field, which carries the
roster-transaction codes injury reports don't cover at all.

Reuses nflverse_ref.fetch_players() -- already cached, already fetched
for player-name resolution (player_lookup.py) -- rather than adding a
new network dependency; this just reads one more column off data
already being pulled.

Known limitation, stated up front: this is a live snapshot, not
season/week-indexed. It answers "what is this player's status right
now," not "what was it in week N" -- fine for the live, current-week
use this whole project is built around, but not something to backtest
against past weeks with.
"""

from __future__ import annotations

import pandas as pd

from edge_engine.roster import nflverse_ref

# nflverse's official roster-status codes that mean "not available to
# play for reasons the injury report doesn't cover" -- excludes ACT
# (active, nothing to flag), and excludes RET/CUT/DEV/etc., which mean
# "not rostered by any NFL team at all" rather than "rostered but
# unavailable" (a free agent pool built from real ESPN data wouldn't
# surface a retired/released player as available in the first place;
# this dict is only about players still on an NFL roster but sidelined).
_FLAGGED_STATUSES: dict[str, str] = {
    "SUS": "Suspended",
    "PUP": "Physically Unable to Perform list",
    "EXE": "Commissioner Exempt List",
    "RES": "Reserve list",
}


def _filter_flagged(players: pd.DataFrame) -> dict[str, tuple[str, str]]:
    """Pure filtering logic, separate from the network-fetching wrapper
    below, so it's directly unit-testable against a synthetic players
    frame -- same pattern as opponent_defense.py's _unpivot_schedule.

    nflverse's player table is a *lifetime* reference (every player
    ever, 25,000+ rows) -- `status` isn't re-scraped forever once a
    player is no longer relevant, so most SUS/PUP/EXE/RES rows are
    stale history from years ago, not a current status. Confirmed
    directly against real data: of ~3,400 rows with a flagged status
    code, `last_season` spanned back to 2017; only a small fraction
    were from the most recent season nflverse knows about. Filtered to
    exactly that most-recent `last_season` value (self-adjusting as
    nflverse's data updates, rather than a year hardcoded in this file
    that would go stale)."""
    most_recent_season = players["last_season"].max()
    flagged = players[
        players["status"].isin(_FLAGGED_STATUSES)
        & players["gsis_id"].notna()
        & (players["last_season"] == most_recent_season)
    ]
    return {row.gsis_id: (row.status, _FLAGGED_STATUSES[row.status]) for row in flagged.itertuples()}


def get_flagged_roster_statuses(force_refresh: bool = False) -> dict[str, tuple[str, str]]:
    """player_id -> (raw_status_code, human_label), only for players
    whose current nflverse roster status means they're rostered by an
    NFL team but not available to play for a non-injury-report reason.
    A player with a normal/active status, or any code not in
    _FLAGGED_STATUSES, is simply absent from this dict -- never a
    default/guessed flag."""
    return _filter_flagged(nflverse_ref.fetch_players(force_refresh))


def roster_status_note(player_id: str | None, statuses: dict[str, tuple[str, str]]) -> str | None:
    """None if the player isn't flagged (the common case) or has no
    resolved player_id; otherwise a short, ready-to-display note."""
    if not player_id or player_id not in statuses:
        return None
    code, label = statuses[player_id]
    return f"Officially listed as {label} ({code}) — may not be eligible to play."
