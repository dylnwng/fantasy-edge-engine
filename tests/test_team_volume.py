import pandas as pd
import pytest

from edge_engine.model import team_volume
from edge_engine.model.features import USAGE_COLUMNS, build_features, feature_columns
from edge_engine.model.team_volume import (
    TEAM_VOLUME_COLUMNS,
    attach_team_volume,
    build_team_volume,
)

PBP_COLUMNS = ["season", "week", "season_type", "posteam", "pass_attempt", "rush_attempt"]


def _play(**overrides):
    row = {"season": 2025, "week": 1, "season_type": "REG", "posteam": "GB",
           "pass_attempt": 0, "rush_attempt": 0}
    row.update(overrides)
    return row


def _stub_pbp(monkeypatch, plays):
    df = pd.DataFrame(plays, columns=PBP_COLUMNS)
    monkeypatch.setattr(team_volume, "fetch_pbp_data", lambda season, force_refresh=False: df)


# ---- building team volume ----

def test_plays_are_pass_plus_rush_attempts(monkeypatch):
    _stub_pbp(monkeypatch, [_play(pass_attempt=1)] * 30 + [_play(rush_attempt=1)] * 25)

    out = build_team_volume(2025).set_index("team")

    assert out.loc["GB", "team_pass_attempts"] == 30
    assert out.loc["GB", "team_plays"] == 55


def test_volume_is_split_by_team_and_week(monkeypatch):
    _stub_pbp(monkeypatch, [
        *[_play(week=1, posteam="GB", pass_attempt=1)] * 10,
        *[_play(week=1, posteam="CHI", pass_attempt=1)] * 40,
        *[_play(week=2, posteam="GB", pass_attempt=1)] * 20,
    ])

    out = build_team_volume(2025).set_index(["team", "week"])

    assert out.loc[("GB", 1), "team_pass_attempts"] == 10
    assert out.loc[("CHI", 1), "team_pass_attempts"] == 40
    assert out.loc[("GB", 2), "team_pass_attempts"] == 20


def test_postseason_is_excluded(monkeypatch):
    _stub_pbp(monkeypatch, [
        _play(week=1, pass_attempt=1),
        _play(week=20, season_type="POST", pass_attempt=1),
    ])

    assert list(build_team_volume(2025)["week"]) == [1]


def test_plays_with_no_possessing_team_are_dropped(monkeypatch):
    # Kickoffs and timeouts carry a null posteam; counting them would
    # invent a team whose name is NaN.
    _stub_pbp(monkeypatch, [
        _play(pass_attempt=1),
        _play(posteam=None, pass_attempt=1),
    ])

    out = build_team_volume(2025)

    assert list(out["team"]) == ["GB"]
    assert out.iloc[0]["team_pass_attempts"] == 1


def test_null_attempt_flags_count_as_zero_not_nan(monkeypatch):
    _stub_pbp(monkeypatch, [_play(pass_attempt=1), _play(pass_attempt=None, rush_attempt=1)])

    out = build_team_volume(2025)

    assert out.iloc[0]["team_pass_attempts"] == 1
    assert out.iloc[0]["team_plays"] == 2


def test_a_season_with_no_play_by_play_raises_clearly(monkeypatch):
    monkeypatch.setattr(team_volume, "fetch_pbp_data", lambda s, force_refresh=False: pd.DataFrame())

    with pytest.raises(RuntimeError, match="no play-by-play"):
        build_team_volume(2099)


# ---- attaching ----

def _player_week(rows):
    return pd.DataFrame(rows, columns=["season", "week", "player_id", "team"])


def test_attach_matches_on_season_week_and_team():
    pw = _player_week([(2025, 1, "P1", "GB"), (2025, 1, "P2", "CHI")])
    volume = pd.DataFrame([
        {"season": 2025, "week": 1, "team": "GB", "team_plays": 60, "team_pass_attempts": 35},
        {"season": 2025, "week": 1, "team": "CHI", "team_plays": 50, "team_pass_attempts": 20},
    ])

    out = attach_team_volume(pw, volume).set_index("player_id")

    assert out.loc["P1", "team_plays"] == 60
    assert out.loc["P2", "team_pass_attempts"] == 20


def test_a_player_with_no_team_volume_gets_null_not_zero():
    # Zero would claim the offence ran no plays, which is never true and
    # would drag a trailing average toward a fabricated floor.
    pw = _player_week([(2025, 1, "P1", "GB"), (2025, 1, "P2", "XXX")])
    volume = pd.DataFrame([
        {"season": 2025, "week": 1, "team": "GB", "team_plays": 60, "team_pass_attempts": 35},
    ])

    out = attach_team_volume(pw, volume).set_index("player_id")

    assert pd.isna(out.loc["P2", "team_plays"])
    assert out.loc["P2", "team_plays"] != 0


def test_duplicate_team_rows_raise_instead_of_fanning_out_the_roster():
    # A duplicated key silently doubles every player-week on that team,
    # corrupting all downstream trailing windows.
    pw = _player_week([(2025, 1, "P1", "GB")])
    volume = pd.DataFrame([
        {"season": 2025, "week": 1, "team": "GB", "team_plays": 60, "team_pass_attempts": 35},
        {"season": 2025, "week": 1, "team": "GB", "team_plays": 61, "team_pass_attempts": 36},
    ])

    with pytest.raises(ValueError, match="duplicate"):
        attach_team_volume(pw, volume)


def test_attach_requires_the_join_keys():
    with pytest.raises(ValueError, match="missing"):
        attach_team_volume(pd.DataFrame({"season": [2025]}), pd.DataFrame())


# ---- the trailing contract the experiment depends on ----

def _usage_row(player_id, season, week, team_plays, **overrides):
    row = dict(
        season=season, week=week, player_id=player_id, position="RB", team="GB",
        snap_pct=0.5, target_share=0.2, air_yards_share=0.2,
        red_zone_touches=1, red_zone_target_share=0.2,
        team_plays=team_plays, team_pass_attempts=30.0,
    )
    row.update(overrides)
    return row


def test_team_volume_is_trailed_not_taken_from_the_predicted_week():
    # The leak this would otherwise introduce: handing the model the pace
    # of the very game it is predicting.
    rows = [_usage_row("P1", 2025, w, team_plays=float(v))
            for w, v in [(1, 50.0), (2, 60.0), (3, 999.0)]]
    extended = list(USAGE_COLUMNS) + list(TEAM_VOLUME_COLUMNS)

    features = build_features(pd.DataFrame(rows), pd.Series([10.0, 10.0, 10.0]),
                              window=2, usage_columns=extended)
    by_week = features.set_index("week")

    # Week 3's trailing average covers weeks 2-3... but the point is that
    # week 2's covers weeks 1-2 only, never week 3's spike.
    assert by_week.loc[2, "trailing_team_plays_avg"] == pytest.approx(55.0)


def test_extended_feature_columns_are_produced_and_named_consistently():
    extended = list(USAGE_COLUMNS) + list(TEAM_VOLUME_COLUMNS)
    rows = [_usage_row("P1", 2025, w, team_plays=60.0) for w in (1, 2, 3)]

    features = build_features(pd.DataFrame(rows), pd.Series([10.0, 10.0, 10.0]),
                              window=2, usage_columns=extended)

    for col in feature_columns(2, usage_columns=extended):
        assert col in features.columns


def test_the_shipped_feature_set_is_unchanged_by_the_parameterisation():
    # The whole point of defaulting: adding this experiment must not move
    # what the shipped model trains on.
    assert feature_columns(2) == feature_columns(2, usage_columns=USAGE_COLUMNS)
    assert not any("team_" in c for c in feature_columns(2))


def test_extended_set_is_a_strict_superset_of_the_shipped_one():
    extended = list(USAGE_COLUMNS) + list(TEAM_VOLUME_COLUMNS)

    shipped = set(feature_columns(2))
    with_volume = set(feature_columns(2, usage_columns=extended))

    assert shipped < with_volume
    assert with_volume - shipped == {
        "trailing_team_plays_avg", "trailing_team_plays_trend",
        "trailing_team_pass_attempts_avg", "trailing_team_pass_attempts_trend",
    }
