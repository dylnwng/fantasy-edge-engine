import pandas as pd
import pytest
import yaml

from edge_engine.roster import manual_source
from edge_engine.roster.manual_source import ManualRosterStateSource
from edge_engine.roster.player_lookup import PlayerLookup, _normalize_name


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


def test_missing_csv_file_raises_clear_message(tmp_path, monkeypatch):
    # A first-time user who hasn't created free_agents.csv yet used to
    # get a bare FileNotFoundError with a raw temp-style path. Must point
    # at the documented format instead.
    _write_league_config(tmp_path)
    _write_roster_meta(tmp_path)
    (tmp_path / "my_roster.csv").write_text("name,position,team\n")
    # free_agents.csv deliberately not created

    monkeypatch.setattr(manual_source.PlayerLookup, "build", classmethod(lambda cls: _StubLookup({})))

    source = ManualRosterStateSource(base_dir=tmp_path)
    with pytest.raises(RuntimeError, match="README"):
        source.get_free_agents()


def test_csv_missing_required_column_raises_clear_message(tmp_path, monkeypatch):
    _write_league_config(tmp_path)
    _write_roster_meta(tmp_path)
    (tmp_path / "my_roster.csv").write_text("name,position,team\n")
    (tmp_path / "free_agents.csv").write_text("name,team\nBob Smith,KC\n")  # no "position" column

    monkeypatch.setattr(manual_source.PlayerLookup, "build", classmethod(lambda cls: _StubLookup({})))

    source = ManualRosterStateSource(base_dir=tmp_path)
    with pytest.raises(RuntimeError, match="position"):
        source.get_free_agents()


def test_empty_league_config_yaml_raises_clear_message(tmp_path):
    (tmp_path / "league_config.yaml").write_text("")
    _write_roster_meta(tmp_path)

    source = ManualRosterStateSource(base_dir=tmp_path)
    with pytest.raises(RuntimeError, match="empty"):
        source.get_league_config()


def test_league_config_missing_key_raises_clear_message(tmp_path):
    (tmp_path / "league_config.yaml").write_text(yaml.dump({"season": 2026}))  # no scoring/lineup_slots/waivers
    _write_roster_meta(tmp_path)

    source = ManualRosterStateSource(base_dir=tmp_path)
    with pytest.raises(RuntimeError, match="scoring"):
        source.get_league_config()


def test_league_config_unexpected_waiver_field_raises_clear_message(tmp_path):
    config = {
        "season": 2026,
        "scoring": {"ppr_type": "full_ppr"},
        "lineup_slots": {"QB": 1},
        "waivers": {
            "system": "FAAB", "season_budget": 100, "min_bid": 1,
            "clear_day": "Wednesday", "extra_field": "oops",
        },
    }
    (tmp_path / "league_config.yaml").write_text(yaml.dump(config))
    _write_roster_meta(tmp_path)

    source = ManualRosterStateSource(base_dir=tmp_path)
    with pytest.raises(RuntimeError, match="extra_field"):
        source.get_league_config()


def test_missing_roster_meta_file_raises_clear_message(tmp_path):
    _write_league_config(tmp_path)
    # roster_meta.yaml deliberately not created

    source = ManualRosterStateSource(base_dir=tmp_path)
    with pytest.raises(RuntimeError, match="README"):
        source.get_roster_meta()


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


def _lookup(players_rows, team_to_name):
    players = pd.DataFrame(players_rows)
    players["_norm_name"] = players["display_name"].map(_normalize_name)
    return PlayerLookup(players, team_to_name)


def test_suffix_collision_is_broken_by_team_despite_nflverse_abbr_mismatch():
    """nflverse's players table says "AZ"; its own team_desc table only
    has "ARI". The unknown abbr used to map to NaN, so the suffix-stripped
    "marvin harrison" stayed ambiguous against his father and the real
    6th-round WR silently fell off the draft board."""
    lookup = _lookup(
        [
            {"display_name": "Marvin Harrison", "position": "WR",
             "latest_team": "IND", "gsis_id": "00-dad"},
            {"display_name": "Marvin Harrison Jr.", "position": "WR",
             "latest_team": "AZ", "gsis_id": "00-son"},
        ],
        {"ARI": "Arizona Cardinals", "IND": "Indianapolis Colts"},  # note: no "AZ"
    )

    assert lookup.resolve("Marvin Harrison Jr.", "WR", "ARI")[0] == "00-son"
    assert lookup.resolve("Marvin Harrison", "WR", "IND")[0] == "00-dad"


def test_two_way_player_listed_at_his_defensive_position_still_resolves():
    """Travis Hunter is a CB in nflverse and a WR in every fantasy league.
    Name plus team is a strong enough key to keep him."""
    lookup = _lookup(
        [{"display_name": "Travis Hunter", "position": "CB",
          "latest_team": "JAX", "gsis_id": "00-hunter"}],
        {"JAX": "Jacksonville Jaguars"},
    )

    gsis_id, status, note = lookup.resolve("Travis Hunter", "WR", "JAX")
    assert (gsis_id, status) == ("00-hunter", "matched")
    assert "CB" in note  # the position difference is stated, not hidden


def test_position_fallback_will_not_grab_a_different_player_on_another_team():
    lookup = _lookup(
        [{"display_name": "Travis Hunter", "position": "CB",
          "latest_team": "JAX", "gsis_id": "00-hunter"}],
        {"JAX": "Jacksonville Jaguars", "KC": "Kansas City Chiefs"},
    )

    assert lookup.resolve("Travis Hunter", "WR", "KC")[0] is None


def test_position_fallback_stays_out_of_the_way_when_the_name_is_shared():
    """Two same-name players on the same team at different positions is
    not something name+team can resolve -- it must decline, not guess."""
    lookup = _lookup(
        [
            {"display_name": "Sam Same", "position": "CB",
             "latest_team": "KC", "gsis_id": "00-a"},
            {"display_name": "Sam Same", "position": "LB",
             "latest_team": "KC", "gsis_id": "00-b"},
        ],
        {"KC": "Kansas City Chiefs"},
    )

    assert lookup.resolve("Sam Same", "WR", "KC")[0] is None
