import pandas as pd
import pytest

from edge_engine.model import target_depth
from edge_engine.model.target_depth import (
    attach_target_depth,
    build_target_depth,
    trailing_adot,
)

PBP_COLUMNS = ["season", "week", "season_type", "pass_attempt", "receiver_player_id", "air_yards"]


def _play(**overrides):
    row = {"season": 2025, "week": 1, "season_type": "REG",
           "pass_attempt": 1, "receiver_player_id": "P1", "air_yards": 10.0}
    row.update(overrides)
    return row


def _stub_pbp(monkeypatch, plays):
    df = pd.DataFrame(plays, columns=PBP_COLUMNS)
    monkeypatch.setattr(target_depth, "fetch_pbp_data", lambda season, force_refresh=False: df)


def _depth(rows):
    """rows: (season, week, player_id, targets, air_yards)"""
    return pd.DataFrame(rows, columns=["season", "week", "player_id", "targets", "air_yards"])


def _by_week(out, player_id="P1"):
    return out[out["player_id"] == player_id].set_index("week")


# ---- building from play-by-play ----

def test_targets_and_air_yards_are_summed_per_player_week(monkeypatch):
    _stub_pbp(monkeypatch, [
        _play(air_yards=5.0), _play(air_yards=15.0), _play(air_yards=25.0),
    ])

    out = build_target_depth(2025).set_index("player_id")

    assert out.loc["P1", "targets"] == 3
    assert out.loc["P1", "air_yards"] == pytest.approx(45.0)


def test_a_pass_with_no_charted_air_yards_counts_as_neither_target_nor_yards(monkeypatch):
    # Numerator and denominator must come from the same plays; counting a
    # target whose depth was never charted silently deflates every aDOT.
    _stub_pbp(monkeypatch, [_play(air_yards=20.0), _play(air_yards=None)])

    out = build_target_depth(2025).set_index("player_id")

    assert out.loc["P1", "targets"] == 1
    assert out.loc["P1", "air_yards"] == pytest.approx(20.0)


def test_throwaways_with_no_receiver_are_excluded(monkeypatch):
    _stub_pbp(monkeypatch, [_play(), _play(receiver_player_id=None)])

    out = build_target_depth(2025)

    assert len(out) == 1
    assert out.iloc[0]["targets"] == 1


def test_rush_plays_are_not_targets(monkeypatch):
    _stub_pbp(monkeypatch, [_play(), _play(pass_attempt=0)])

    assert build_target_depth(2025).iloc[0]["targets"] == 1


def test_postseason_is_excluded(monkeypatch):
    _stub_pbp(monkeypatch, [_play(week=1), _play(week=20, season_type="POST")])

    assert list(build_target_depth(2025)["week"]) == [1]


def test_players_are_kept_separate(monkeypatch):
    _stub_pbp(monkeypatch, [
        _play(receiver_player_id="P1", air_yards=5.0),
        _play(receiver_player_id="P2", air_yards=30.0),
    ])

    out = build_target_depth(2025).set_index("player_id")

    assert out.loc["P1", "air_yards"] == pytest.approx(5.0)
    assert out.loc["P2", "air_yards"] == pytest.approx(30.0)


def test_a_season_with_no_play_by_play_raises_clearly(monkeypatch):
    monkeypatch.setattr(target_depth, "fetch_pbp_data", lambda s, force_refresh=False: pd.DataFrame())

    with pytest.raises(RuntimeError, match="no play-by-play"):
        build_target_depth(2099)


# ---- the ratio-of-sums construction ----

def test_trailing_adot_is_a_ratio_of_sums_not_a_mean_of_ratios():
    # Week 1: one target at 40 yards  -> weekly aDOT 40
    # Week 2: nine targets at 5 yards -> weekly aDOT 5
    # Mean of ratios would be 22.5, weighting one deep throw as heavily as
    # a full afternoon. Ratio of sums is (40+45)/(1+9) = 8.5.
    depth = _depth([(2025, 1, "P1", 1, 40.0), (2025, 2, "P1", 9, 45.0)])

    out = _by_week(trailing_adot(depth, window=2))

    assert out.loc[2, "trailing_adot"] == pytest.approx(8.5)
    assert out.loc[2, "trailing_adot"] != pytest.approx(22.5)


def test_a_zero_target_week_contributes_nothing_rather_than_a_zero_depth():
    depth = _depth([(2025, 1, "P1", 4, 40.0), (2025, 2, "P1", 0, 0.0)])

    out = _by_week(trailing_adot(depth, window=2))

    # Still 40/4 = 10, undiluted by a week he wasn't thrown to.
    assert out.loc[2, "trailing_adot"] == pytest.approx(10.0)


def test_no_targets_across_the_whole_window_is_undefined_not_zero():
    # A non-receiving back has no depth of target. Filling 0.0 would tell
    # the model he runs the shallowest route tree in football, which is a
    # claim about missing data rather than about football.
    depth = _depth([(2025, 1, "P1", 0, 0.0), (2025, 2, "P1", 0, 0.0)])

    out = _by_week(trailing_adot(depth, window=2))

    assert pd.isna(out.loc[2, "trailing_adot"])


def test_a_partial_window_is_undefined():
    depth = _depth([(2025, 1, "P1", 4, 40.0), (2025, 2, "P1", 4, 40.0)])

    out = _by_week(trailing_adot(depth, window=2))

    assert pd.isna(out.loc[1, "trailing_adot"])  # only one game so far
    assert out.loc[2, "trailing_adot"] == pytest.approx(10.0)


def test_the_window_does_not_cross_a_season_boundary():
    depth = _depth([
        (2024, 17, "P1", 5, 150.0),   # 30.0 aDOT
        (2025, 1, "P1", 5, 25.0),     # 5.0 aDOT
        (2025, 2, "P1", 5, 25.0),
    ])

    out = trailing_adot(depth, window=2)
    row = out[(out["season"] == 2025) & (out["week"] == 2)].iloc[0]

    assert row["trailing_adot"] == pytest.approx(5.0)  # 2024 not pulled in


def test_the_window_counts_played_games_not_calendar_weeks():
    # P1 has no row for week 2 at all. The window over his own played
    # games is weeks 1 and 3 -- a bye is skipped, matching features.py.
    depth = _depth([(2025, 1, "P1", 5, 50.0), (2025, 3, "P1", 5, 50.0)])

    out = _by_week(trailing_adot(depth, window=2))

    assert out.loc[3, "trailing_adot"] == pytest.approx(10.0)


def test_one_players_targets_are_not_credited_to_another():
    depth = _depth([
        (2025, 1, "P1", 5, 50.0), (2025, 2, "P1", 5, 50.0),
        (2025, 1, "P2", 5, 250.0), (2025, 2, "P2", 5, 250.0),
    ])

    out = trailing_adot(depth, window=2)

    assert _by_week(out, "P1").loc[2, "trailing_adot"] == pytest.approx(10.0)
    assert _by_week(out, "P2").loc[2, "trailing_adot"] == pytest.approx(50.0)


# ---- guards ----

def test_a_duplicate_key_raises_instead_of_double_counting():
    depth = _depth([
        (2025, 1, "P1", 5, 50.0), (2025, 1, "P1", 5, 50.0), (2025, 2, "P1", 5, 50.0),
    ])

    with pytest.raises(ValueError, match="duplicate"):
        trailing_adot(depth, window=2)


def test_a_nonsense_window_raises():
    with pytest.raises(ValueError, match="window"):
        trailing_adot(_depth([(2025, 1, "P1", 1, 1.0)]), window=0)


def test_missing_columns_are_named():
    with pytest.raises(ValueError, match="missing"):
        trailing_adot(pd.DataFrame({"season": [2025], "week": [1]}), window=2)


# ---- attaching ----

def test_attach_leaves_undefined_depth_null_rather_than_filling_it():
    features = pd.DataFrame([
        {"season": 2025, "week": 2, "player_id": "P1"},
        {"season": 2025, "week": 2, "player_id": "P2"},
    ])
    depth = pd.DataFrame([{"season": 2025, "week": 2, "player_id": "P1", "trailing_adot": 9.5}])

    out = attach_target_depth(features, depth).set_index("player_id")

    assert out.loc["P1", "trailing_adot"] == pytest.approx(9.5)
    assert pd.isna(out.loc["P2", "trailing_adot"])


def test_attach_refuses_to_fan_out_the_feature_table():
    features = pd.DataFrame([{"season": 2025, "week": 1, "player_id": "P1"}])
    depth = pd.DataFrame([
        {"season": 2025, "week": 1, "player_id": "P1", "trailing_adot": 9.0},
        {"season": 2025, "week": 1, "player_id": "P1", "trailing_adot": 11.0},
    ])

    with pytest.raises(ValueError, match="duplicate"):
        attach_target_depth(features, depth)
