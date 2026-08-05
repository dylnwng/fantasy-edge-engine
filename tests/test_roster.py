import pandas as pd
import yaml

from edge_engine.roster import manual_source
from edge_engine.roster.manual_source import ManualRosterStateSource
from edge_engine.roster.player_lookup import PlayerLookup


class _StubLookup:
    """Resolves by exact (name, position) from a small fixed table, so
    tests don't hit the network via PlayerLookup.build()."""

    def __init__(self, table: dict[tuple[str, str], str]):
        self.table = table

    def resolve(self, name, position, team):
        key = (name, position)
        if key in self.table:
            return self.table[key], "matched", ""
        return None, "unmatched", f"no player found for name={name!r} position={position!r}"


def _write_league_config(base_dir, season=2026):
    config = {
        "season": season,
        "scoring": {"ppr_type": "full_ppr", "bonuses": {"pass_td": 4}},
        "lineup_slots": {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
        "waivers": {
            "system": "FAAB",
            "season_budget": 100,
            "min_bid": 1,
            "clear_day": "Wednesday",
            "post_clear_rule": "first_come_first_available",
        },
    }
    (base_dir / "league_config.yaml").write_text(yaml.dump(config))


def _write_roster_meta(base_dir):
    meta = {"as_of_week": 3, "as_of_date": "2026-09-22", "remaining_faab": 87.5}
    (base_dir / "roster_meta.yaml").write_text(yaml.dump(meta))


def test_manual_source_resolves_players_and_bye_weeks(tmp_path, monkeypatch):
    _write_league_config(tmp_path)
    _write_roster_meta(tmp_path)
    (tmp_path / "my_roster.csv").write_text("name,position,team\nJosh Jacobs,RB,GB\n")
    (tmp_path / "free_agents.csv").write_text("name,position,team\n")

    monkeypatch.setattr(
        manual_source.PlayerLookup, "build", classmethod(lambda cls: _StubLookup({("Josh Jacobs", "RB"): "00-0035700"}))
    )
    monkeypatch.setattr(manual_source, "get_bye_weeks", lambda season: {"GB": 11})

    source = ManualRosterStateSource(base_dir=tmp_path)
    roster = source.get_rostered_players()

    assert len(roster) == 1
    player = roster[0]
    assert player.player_id == "00-0035700"
    assert player.match_status == "matched"
    assert player.bye_week == 11

    config = source.get_league_config()
    assert config.season == 2026
    assert config.scoring.ppr_type == "full_ppr"
    assert config.waivers.season_budget == 100

    meta = source.get_roster_meta()
    assert meta.as_of_week == 3
    assert meta.remaining_faab == 87.5


def test_unmatched_player_is_surfaced_not_dropped(tmp_path, monkeypatch):
    _write_league_config(tmp_path)
    _write_roster_meta(tmp_path)
    (tmp_path / "my_roster.csv").write_text("name,position,team\n")
    (tmp_path / "free_agents.csv").write_text("name,position,team\nTypo Playerr,RB,LA\n")

    monkeypatch.setattr(manual_source.PlayerLookup, "build", classmethod(lambda cls: _StubLookup({})))
    monkeypatch.setattr(manual_source, "get_bye_weeks", lambda season: {"LA": 11})

    source = ManualRosterStateSource(base_dir=tmp_path)
    free_agents = source.get_free_agents()

    assert len(free_agents) == 1
    assert free_agents[0].match_status == "unmatched"
    assert free_agents[0].player_id is None
    assert "Typo Playerr" in free_agents[0].match_note


def test_player_lookup_disambiguates_by_team():
    players = pd.DataFrame(
        {
            "display_name": ["Josh Allen", "Josh Allen"],
            "position": ["QB", "LB"],
            "latest_team": ["BUF", "JAX"],
            "gsis_id": ["00-1111", "00-2222"],
        }
    )
    players["_norm_name"] = players["display_name"].str.lower()
    team_to_name = {"BUF": "Buffalo Bills", "JAX": "Jacksonville Jaguars"}
    lookup = PlayerLookup(players, team_to_name)

    gsis_id, status, _ = lookup.resolve("Josh Allen", "QB", "BUF")
    assert gsis_id == "00-1111"
    assert status == "matched"
