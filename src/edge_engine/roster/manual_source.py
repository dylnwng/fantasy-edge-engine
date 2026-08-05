"""Manual-entry implementation of RosterStateSource: reads the CSV/YAML
files in data/roster_state/. This is the only place that knows those file
formats exist — everything else goes through interface.RosterStateSource.
"""

from __future__ import annotations

import csv
from pathlib import Path

import yaml

from edge_engine.paths import ROSTER_STATE_DIR
from edge_engine.roster.bye_weeks import get_bye_weeks
from edge_engine.roster.models import LeagueConfig, Player, RosterMeta, ScoringSettings, WaiverConfig
from edge_engine.roster.player_lookup import PlayerLookup


class ManualRosterStateSource:
    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or ROSTER_STATE_DIR
        self._lookup: PlayerLookup | None = None
        self._league_config: LeagueConfig | None = None

    def _get_lookup(self) -> PlayerLookup:
        if self._lookup is None:
            self._lookup = PlayerLookup.build()
        return self._lookup

    def _read_player_csv(self, filename: str) -> list[Player]:
        season = self.get_league_config().season
        byes = get_bye_weeks(season)
        lookup = self._get_lookup()

        players = []
        with open(self.base_dir / filename, newline="") as f:
            for row in csv.DictReader(f):
                name = row["name"].strip()
                position = row["position"].strip().upper()
                team = row["team"].strip().upper()
                gsis_id, status, note = lookup.resolve(name, position, team)
                players.append(
                    Player(
                        name=name,
                        position=position,
                        team=team,
                        player_id=gsis_id,
                        bye_week=byes.get(team),
                        match_status=status,
                        match_note=note,
                    )
                )
        return players

    def get_rostered_players(self) -> list[Player]:
        return self._read_player_csv("my_roster.csv")

    def get_free_agents(self) -> list[Player]:
        return self._read_player_csv("free_agents.csv")

    def get_league_config(self) -> LeagueConfig:
        if self._league_config is None:
            with open(self.base_dir / "league_config.yaml") as f:
                raw = yaml.safe_load(f)
            scoring = ScoringSettings(
                ppr_type=raw["scoring"]["ppr_type"], bonuses=raw["scoring"].get("bonuses") or {}
            )
            waivers = WaiverConfig(**raw["waivers"])
            self._league_config = LeagueConfig(
                season=raw["season"],
                scoring=scoring,
                lineup_slots=raw["lineup_slots"],
                waivers=waivers,
            )
        return self._league_config

    def get_roster_meta(self) -> RosterMeta:
        with open(self.base_dir / "roster_meta.yaml") as f:
            raw = yaml.safe_load(f)
        return RosterMeta(
            as_of_week=raw["as_of_week"],
            as_of_date=raw["as_of_date"],
            remaining_faab=raw["remaining_faab"],
        )
