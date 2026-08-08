import pandas as pd
import pytest

from edge_engine.ingestion import pbp_fallback
from edge_engine.ingestion.pbp_fallback import _attach_shares, _player_team_map, build_weekly_from_pbp

PBP_COLUMNS = [
    "season", "week", "season_type", "posteam", "defteam",
    "passer_player_id", "rusher_player_id", "receiver_player_id", "fumbled_1_player_id",
    "passing_yards", "pass_touchdown", "interception",
    "rushing_yards", "rush_touchdown",
    "receiving_yards", "complete_pass", "pass_attempt", "air_yards",
    "fumble_lost", "two_point_conv_result",
]


def _play(**overrides):
    row = {
        "season": 2025, "week": 1, "season_type": "REG", "posteam": "KC", "defteam": "LV",
        "passer_player_id": None, "rusher_player_id": None, "receiver_player_id": None,
        "fumbled_1_player_id": None,
        "passing_yards": 0.0, "pass_touchdown": 0, "interception": 0,
        "rushing_yards": 0.0, "rush_touchdown": 0,
        "receiving_yards": 0.0, "complete_pass": 0, "pass_attempt": 0, "air_yards": 0.0,
        "fumble_lost": 0, "two_point_conv_result": None,
    }
    row.update(overrides)
    return row


def _build(plays, monkeypatch):
    pbp = pd.DataFrame(plays, columns=PBP_COLUMNS)
    monkeypatch.setattr(pbp_fallback, "fetch_pbp_data", lambda season, force_refresh=False: pbp)
    # _attach_identity hits the network for weekly rosters; identity is
    # not what these tests are about.
    monkeypatch.setattr(
        pbp_fallback, "_attach_identity",
        lambda weekly, season: weekly.assign(player_name="Test Player", position="WR"),
    )
    return build_weekly_from_pbp(2025)


def test_receiving_stat_line_scores_correctly_in_ppr(monkeypatch):
    # One 20-yard TD catch: 20 rec yds (2.0) + TD (6.0) + reception (1.0
    # in PPR) = 9.0 PPR, 8.0 standard.
    out = _build(
        [_play(
            passer_player_id="QB1", receiver_player_id="WR1",
            receiving_yards=20.0, passing_yards=20.0, pass_touchdown=1,
            complete_pass=1, pass_attempt=1, air_yards=15.0,
        )],
        monkeypatch,
    )
    wr = out[out["player_id"] == "WR1"].iloc[0]
    assert wr["fantasy_points_ppr"] == pytest.approx(9.0)
    assert wr["fantasy_points"] == pytest.approx(8.0)


def test_the_same_touchdown_credits_passer_and_receiver_separately(monkeypatch):
    # One pass TD is a passing TD for the QB (4 pts) and a receiving TD
    # for the WR (6 pts) -- not one or the other.
    out = _build(
        [_play(
            passer_player_id="QB1", receiver_player_id="WR1",
            passing_yards=20.0, receiving_yards=20.0, pass_touchdown=1,
            complete_pass=1, pass_attempt=1,
        )],
        monkeypatch,
    ).set_index("player_id")

    # QB: 20 pass yds / 25 = 0.8, + 4 for the TD = 4.8
    assert out.loc["QB1", "fantasy_points_ppr"] == pytest.approx(4.8)
    assert out.loc["WR1", "fantasy_points_ppr"] == pytest.approx(9.0)


def test_interceptions_and_lost_fumbles_are_negative(monkeypatch):
    out = _build(
        [
            _play(passer_player_id="QB1", interception=1, pass_attempt=1),
            _play(rusher_player_id="RB1", fumbled_1_player_id="RB1", fumble_lost=1),
        ],
        monkeypatch,
    ).set_index("player_id")

    assert out.loc["QB1", "fantasy_points_ppr"] == pytest.approx(-2.0)
    assert out.loc["RB1", "fantasy_points_ppr"] == pytest.approx(-2.0)


def test_postseason_plays_are_excluded(monkeypatch):
    out = _build(
        [
            _play(receiver_player_id="WR1", receiving_yards=10.0, complete_pass=1, pass_attempt=1),
            _play(receiver_player_id="WR2", receiving_yards=99.0, complete_pass=1,
                  pass_attempt=1, season_type="POST"),
        ],
        monkeypatch,
    )
    assert set(out["player_id"]) == {"WR1"}


def test_target_share_is_against_the_players_own_team_total():
    weekly = pd.DataFrame(
        [
            {"season": 2025, "week": 1, "player_id": "A", "team": "KC", "targets": 3.0, "air_yards": 30.0},
            {"season": 2025, "week": 1, "player_id": "B", "team": "KC", "targets": 1.0, "air_yards": 10.0},
            {"season": 2025, "week": 1, "player_id": "C", "team": "LV", "targets": 5.0, "air_yards": 50.0},
        ]
    )
    out = _attach_shares(weekly).set_index("player_id")

    assert out.loc["A", "target_share"] == pytest.approx(0.75)  # 3 of KC's 4
    assert out.loc["B", "target_share"] == pytest.approx(0.25)
    assert out.loc["C", "target_share"] == pytest.approx(1.0)  # LV's only target


def test_share_is_null_not_zero_when_the_team_denominator_is_zero():
    # A share with no denominator is undefined, not 0 -- same convention
    # transform.py uses for red_zone_target_share.
    weekly = pd.DataFrame(
        [{"season": 2025, "week": 1, "player_id": "A", "team": "KC", "targets": 0.0, "air_yards": 0.0}]
    )
    out = _attach_shares(weekly)
    assert pd.isna(out.iloc[0]["target_share"])
    assert pd.isna(out.iloc[0]["air_yards_share"])


def test_player_team_map_resolves_team_and_opponent():
    pbp = pd.DataFrame(
        [_play(rusher_player_id="RB1"), _play(receiver_player_id="WR1")], columns=PBP_COLUMNS
    )
    out = _player_team_map(pbp).set_index("player_id")

    assert out.loc["RB1", "team"] == "KC"
    assert out.loc["RB1", "opponent_team"] == "LV"
    assert out.loc["WR1", "team"] == "KC"


def test_season_with_no_play_by_play_fails_with_a_clear_message(monkeypatch):
    # nfl_data_py returns an EMPTY frame (no columns) for a season it has
    # no data for rather than raising, so the reconstruction used to die
    # on a bare KeyError: 'season_type'. This is the normal state for a
    # season that hasn't been played yet.
    monkeypatch.setattr(
        pbp_fallback, "fetch_pbp_data", lambda season, force_refresh=False: pd.DataFrame()
    )
    with pytest.raises(RuntimeError, match="no play-by-play"):
        build_weekly_from_pbp(2099)


def test_a_player_with_no_targets_gets_null_share_not_zero():
    # nflverse never emits a 0.0 target_share -- only NaN or positive
    # (verified against real 2024: 659 of 664 QB rows are NaN, as are
    # 350 RB and 35 WR rows). This is load-bearing, not cosmetic:
    # features.py drops null-feature rows, and that is the mechanism
    # keeping quarterbacks out of a model whose receiving-usage
    # features carry no signal for them. Emitting 0.0 silently reversed
    # that documented scope boundary.
    weekly = pd.DataFrame(
        [
            {"season": 2025, "week": 1, "player_id": "QB1", "team": "KC", "targets": 0.0, "air_yards": 0.0},
            {"season": 2025, "week": 1, "player_id": "WR1", "team": "KC", "targets": 4.0, "air_yards": 40.0},
        ]
    )
    out = _attach_shares(weekly).set_index("player_id")

    assert pd.isna(out.loc["QB1", "target_share"])
    assert pd.isna(out.loc["QB1", "air_yards_share"])
    assert out.loc["WR1", "target_share"] == pytest.approx(1.0)
