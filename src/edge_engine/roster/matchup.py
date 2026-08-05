"""Opponent-matchup data -- deliberately NOT part of RosterStateSource.

RosterStateSource (interface.py) is a contract ManualRosterStateSource must
fully satisfy, and there is no manual-CSV equivalent of "my opponent's
roster" -- hand-maintaining an opponent's roster weekly has no payoff, so
it will never be built. Bolting an opponent method onto RosterStateSource
would force ManualRosterStateSource to either fake support for it or
raise NotImplementedError, silently breaking the "every source fully
implements this Protocol" guarantee. A second, smaller Protocol that only
EspnRosterStateSource satisfies keeps that guarantee intact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from edge_engine.roster.interface import get_default_source
from edge_engine.roster.models import Player


@dataclass(frozen=True)
class MatchupPlayer:
    player: Player  # includes slot_position (this week's starter/bench slot)
    espn_projected_points: float  # ESPN's own projection -- fallback/reference only, never primary
    actual_points: float  # 0 until the game is final
    game_final: bool
    on_bye: bool


@dataclass(frozen=True)
class MatchupSnapshot:
    week: int
    my_team_name: str
    opponent_team_name: str
    my_lineup: list[MatchupPlayer]  # starters + bench, this week
    opponent_lineup: list[MatchupPlayer]  # starters + bench, this week


class MatchupDataSource(Protocol):
    def get_current_matchup(self, week: int | None = None) -> MatchupSnapshot: ...


def get_matchup_source() -> MatchupDataSource:
    """The matchup simulator is ESPN-only by product decision. Reuses
    get_default_source() so EDGE_ENGINE_ROSTER_SOURCE stays the one
    switch that decides sourcing, and fails loudly here -- not via an
    AttributeError deep inside simulation code -- if the active source
    can't supply it."""
    source = get_default_source()
    if not hasattr(source, "get_current_matchup"):
        raise RuntimeError(
            "Matchup simulation requires EDGE_ENGINE_ROSTER_SOURCE=espn -- "
            "opponent rosters have no manual-CSV equivalent. Current source: "
            f"{type(source).__name__}."
        )
    return source  # type: ignore[return-value]
