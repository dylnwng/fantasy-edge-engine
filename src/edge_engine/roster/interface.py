"""The roster-state interface. Everything downstream (ranking, roster-fit
re-ranking) must depend only on this Protocol, never on manual_source.py's
file formats directly. That's what makes the manual CSV/YAML source
swappable for a live ESPN connector later without touching model or
ranking code.
"""

from __future__ import annotations

from typing import Protocol

from edge_engine.roster.models import LeagueConfig, Player, RosterMeta


class RosterStateSource(Protocol):
    def get_rostered_players(self) -> list[Player]: ...
    def get_free_agents(self) -> list[Player]: ...
    def get_league_config(self) -> LeagueConfig: ...
    def get_roster_meta(self) -> RosterMeta: ...


def get_default_source() -> RosterStateSource:
    """The active roster-state source. Swap this to change data sources
    everywhere at once (e.g. to a live ESPN connector in a later phase)."""
    from edge_engine.roster.manual_source import ManualRosterStateSource

    return ManualRosterStateSource()
