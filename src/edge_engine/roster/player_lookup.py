"""Resolves a manually-entered (name, position, team) roster row to an
nflverse gsis_id, so the usage-data pipeline's player_week table (also
keyed by gsis_id) can be joined against roster state later.

Team abbreviations are compared by team *name* rather than abbreviation,
so a stale/alternate abbreviation the user types (e.g. "LAR" or "WAS")
still matches nflverse's current one (e.g. "LA", "WSH") without the user
needing to know which alias nflverse currently prefers.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from edge_engine.roster import nflverse_ref


def _normalize_name(name: str) -> str:
    name = name.lower().strip()
    name = re.sub(r"[.'’]", "", name)
    name = re.sub(r"\s+(jr|sr|ii|iii|iv|v)$", "", name)
    name = re.sub(r"\s+", " ", name)
    return name


@dataclass
class PlayerLookup:
    _players: pd.DataFrame
    _team_to_name: dict[str, str]

    @classmethod
    def build(cls, force_refresh: bool = False) -> "PlayerLookup":
        players = nflverse_ref.fetch_players(force_refresh).dropna(subset=["gsis_id"]).copy()
        players["_norm_name"] = players["display_name"].fillna("").map(_normalize_name)

        team_desc = nflverse_ref.fetch_team_desc(force_refresh)
        team_to_name = dict(zip(team_desc["team_abbr"], team_desc["team_name"]))

        return cls(players, team_to_name)

    def resolve(self, name: str, position: str, team: str) -> tuple[str | None, str, str]:
        """Returns (gsis_id or None, match_status, note)."""
        norm = _normalize_name(name)
        candidates = self._players[
            (self._players["_norm_name"] == norm) & (self._players["position"] == position)
        ]

        if len(candidates) == 0:
            return None, "unmatched", f"no player found for name={name!r} position={position!r}"

        if len(candidates) == 1:
            return candidates.iloc[0]["gsis_id"], "matched", ""

        target_team_name = self._team_to_name.get(team.upper())
        candidates = candidates.copy()
        candidates["_team_name"] = candidates["latest_team"].map(self._team_to_name)
        team_matches = (
            candidates[candidates["_team_name"] == target_team_name]
            if target_team_name
            else candidates.iloc[0:0]
        )

        if len(team_matches) == 1:
            return team_matches.iloc[0]["gsis_id"], "matched", ""

        return (
            None,
            "ambiguous",
            f"{len(candidates)} players match name={name!r} position={position!r}; "
            f"team={team!r} didn't disambiguate",
        )
