"""Resolves a manually-entered (name, position, team) roster row to an
nflverse gsis_id, so the usage-data pipeline's player_week table (also
keyed by gsis_id) can be joined against roster state later.

Team abbreviations are compared by team *name* rather than abbreviation,
so a stale/alternate abbreviation the user types (e.g. "LAR" or "WAS")
still matches nflverse's current one (e.g. "LA", "WSH") without the user
needing to know which alias nflverse currently prefers. That comparison
runs through `_TEAM_ABBR_ALIASES` on *both* sides, because nflverse's own
tables disagree with each other.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from edge_engine.roster import nflverse_ref


# nflverse is not internally consistent about team abbreviations: its
# players table says "AZ" while its own team_desc table only carries
# "ARI". An abbreviation missing from team_desc silently maps to NaN,
# which used to make every same-name pair on that team unresolvable --
# Marvin Harrison Jr. (AZ) could never be told apart from his father
# (IND), so the #6-round WR simply fell off the draft board untagged.
# Normalize both sides through this map before the team_desc lookup.
_TEAM_ABBR_ALIASES = {"AZ": "ARI"}


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

    def _team_name(self, abbr: str) -> str | None:
        if not isinstance(abbr, str):
            return None
        abbr = abbr.upper()
        return self._team_to_name.get(_TEAM_ABBR_ALIASES.get(abbr, abbr))

    def _by_team(self, candidates: pd.DataFrame, team: str) -> pd.DataFrame:
        target = self._team_name(team)
        if target is None:
            return candidates.iloc[0:0]
        return candidates[candidates["latest_team"].map(self._team_name) == target]

    def resolve(self, name: str, position: str, team: str) -> tuple[str | None, str, str]:
        """Returns (gsis_id or None, match_status, note)."""
        norm = _normalize_name(name)
        by_name = self._players[self._players["_norm_name"] == norm]
        candidates = by_name[by_name["position"] == position]

        if len(candidates) == 0:
            # nflverse lists a player at his *defensive* or depth-chart
            # position, which for a two-way player is not the one he's
            # drafted at -- Travis Hunter is "CB" there and a WR in every
            # fantasy league. Name plus team is already a strong key, so
            # fall back to it rather than dropping a real player. Still
            # requires a unique hit, and says what it did.
            fallback = self._by_team(by_name, team)
            if len(fallback) == 1:
                listed = fallback.iloc[0]["position"]
                return (
                    fallback.iloc[0]["gsis_id"],
                    "matched",
                    f"{name} is listed as {listed} by nflverse, matched on name and team",
                )
            return None, "unmatched", f"no player found for name={name!r} position={position!r}"

        if len(candidates) == 1:
            return candidates.iloc[0]["gsis_id"], "matched", ""

        team_matches = self._by_team(candidates, team)
        if len(team_matches) == 1:
            return team_matches.iloc[0]["gsis_id"], "matched", ""

        return (
            None,
            "ambiguous",
            f"{len(candidates)} players match name={name!r} position={position!r}; "
            f"team={team!r} didn't disambiguate",
        )
