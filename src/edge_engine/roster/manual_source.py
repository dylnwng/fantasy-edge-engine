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

        path = self.base_dir / filename
        try:
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
        except FileNotFoundError as e:
            raise RuntimeError(
                f"{path} doesn't exist -- see data/roster_state/README.md for the "
                "expected format, or point EDGE_ENGINE_ROSTER_SOURCE at 'espn' instead."
            ) from e

        players = []
        # DictReader rows are 0-indexed after the header, so the real CSV
        # line number is i+2 (1 for the header, 1 because humans count
        # from 1) -- makes a bad row easy to find by hand.
        for i, row in enumerate(rows):
            try:
                name = row["name"].strip()
                position = row["position"].strip().upper()
                team = row["team"].strip().upper()
            except KeyError as e:
                raise RuntimeError(
                    f"{path}: missing required column {e.args[0]!r} (row {i + 2}). "
                    "Expected columns: name, position, team -- see "
                    "data/roster_state/README.md."
                ) from e
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
            path = self.base_dir / "league_config.yaml"
            raw = _load_yaml_or_raise(path)
            try:
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
            except KeyError as e:
                raise RuntimeError(
                    f"{path}: missing required key {e.args[0]!r} -- see "
                    "data/roster_state/README.md for the expected format."
                ) from e
            except TypeError as e:
                raise RuntimeError(
                    f"{path}: {e} -- see data/roster_state/README.md for the expected "
                    "'waivers' fields (system, season_budget, min_bid, clear_day)."
                ) from e
        return self._league_config

    def get_roster_meta(self) -> RosterMeta:
        path = self.base_dir / "roster_meta.yaml"
        raw = _load_yaml_or_raise(path)
        try:
            return RosterMeta(
                as_of_week=raw["as_of_week"],
                as_of_date=raw["as_of_date"],
                remaining_faab=raw["remaining_faab"],
            )
        except KeyError as e:
            raise RuntimeError(
                f"{path}: missing required key {e.args[0]!r} -- see "
                "data/roster_state/README.md for the expected format."
            ) from e


def _load_yaml_or_raise(path: Path) -> dict:
    """A QA pass found that a missing file, or a YAML file left empty,
    both produced raw, unhelpful exceptions (FileNotFoundError with a
    bare path; yaml.safe_load("")'s None causing a downstream 'NoneType'
    object is not subscriptable, with no hint the file itself was the
    problem). Both are realistic first-run mistakes on this, the
    zero-credential default roster-state source -- worth a clear message
    each, not just letting them propagate raw."""
    try:
        with open(path) as f:
            raw = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"{path} doesn't exist -- see data/roster_state/README.md for the "
            "expected format, or point EDGE_ENGINE_ROSTER_SOURCE at 'espn' instead."
        ) from e
    if raw is None:
        raise RuntimeError(
            f"{path} is empty -- see data/roster_state/README.md for the expected format."
        )
    return raw
