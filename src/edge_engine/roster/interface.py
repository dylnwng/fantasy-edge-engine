"""The roster-state interface. Everything downstream (ranking, roster-fit
re-ranking) must depend only on this Protocol, never on manual_source.py's
file formats directly. That's what makes the manual CSV/YAML source
swappable for a live ESPN connector later without touching model or
ranking code.
"""

from __future__ import annotations

import os
from typing import Protocol

from dotenv import load_dotenv

from edge_engine.roster.models import LeagueConfig, Player, RosterMeta

# Loads ESPN_* / EDGE_ENGINE_ROSTER_SOURCE from a gitignored .env at the
# project root, if present. Safe to call repeatedly; a no-op if there's
# no .env or the vars are already set in the real environment.
load_dotenv()


class RosterStateSource(Protocol):
    def get_rostered_players(self) -> list[Player]: ...
    def get_free_agents(self) -> list[Player]: ...
    def get_league_config(self) -> LeagueConfig: ...
    def get_roster_meta(self) -> RosterMeta: ...


def get_default_source() -> RosterStateSource:
    """The active roster-state source, chosen by EDGE_ENGINE_ROSTER_SOURCE
    ("manual", the default, or "espn"). This is the one place that
    decision gets made -- everything downstream only ever depends on the
    RosterStateSource protocol above, never on which implementation is
    active, so flipping this env var is the entire migration."""
    source = os.environ.get("EDGE_ENGINE_ROSTER_SOURCE", "manual").lower()

    if source == "espn":
        from edge_engine.roster.espn_source import EspnRosterStateSource

        return EspnRosterStateSource()

    if source != "manual":
        raise ValueError(f"Unknown EDGE_ENGINE_ROSTER_SOURCE={source!r} (expected 'manual' or 'espn')")

    from edge_engine.roster.manual_source import ManualRosterStateSource

    return ManualRosterStateSource()
