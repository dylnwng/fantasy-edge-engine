import pandas as pd
import pytest

from edge_engine.model import qb_features
from edge_engine.model.qb_features import (
    MIN_ATTEMPTS,
    build_qb_features,
    build_qb_volume,
    qb_feature_columns,
    qb_usage_explanation,
)

PBP_COLUMNS = [
    "season", "week", "season_type", "posteam", "defteam",
    "passer_player_id", "rusher_player_id",
    "pass_attempt", "rush_attempt", "air_yards",
]


def _play(**overrides):
    row = {
        "season": 2025, "week": 1, "season_type": "REG", "posteam": "KC", "defteam": "LV",
        "passer_player_id": None, "rusher_player_id": None,
        "pass_attempt": 0, "rush_attempt": 0, "air_yards": 0.0,
    }
    row.update(overrides)
    return row


def _pbp(plays, monkeypatch):
    df = pd.DataFrame(plays, columns=PBP_COLUMNS)
    monkeypatch.setattr(qb_features, "fetch_pbp_data", lambda season, force_refresh=False: df)


def _dropbacks(qb, week, n, team="KC", air=8.0):
    return [
        _play(week=week, passer_player_id=qb, pass_attempt=1, air_yards=air, posteam=team)
        for _ in range(n)
    ]


def _volume_row(player_id, week, attempts=25.0, pass_share=0.9, rush_attempts=3.0,
                team_attempts=28.0, air_yards=200.0, season=2025, team="KC"):
    return {
        "season": season, "week": week, "player_id": player_id, "team": team,
        "attempts": attempts, "pass_share": pass_share, "rush_attempts": rush_attempts,
        "team_attempts": team_attempts, "air_yards": air_yards,
    }


# ---- volume construction ----

def test_mop_up_quarterback_weeks_are_excluded(monkeypatch):
    # A QB with a handful of attempts got hurt early or was cleaning up a
    # blowout; that isn't a fantasy-relevant starter week.
    _pbp(_dropbacks("STARTER", 1, 30) + _dropbacks("BACKUP", 1, MIN_ATTEMPTS - 1), monkeypatch)
    out = build_qb_volume(2025)

    assert set(out["player_id"]) == {"STARTER"}


def test_pass_share_is_against_the_players_own_team_volume(monkeypatch):
    # Two QBs splitting one team's passing work.
    _pbp(_dropbacks("QB1", 1, 30) + _dropbacks("QB2", 1, 10), monkeypatch)
    out = build_qb_volume(2025).set_index("player_id")

    assert out.loc["QB1", "team_attempts"] == 40
    assert out.loc["QB1", "pass_share"] == pytest.approx(0.75)
    assert out.loc["QB2", "pass_share"] == pytest.approx(0.25)


def test_rush_attempts_are_counted_for_running_quarterbacks(monkeypatch):
    plays = _dropbacks("QB1", 1, 25) + [
        _play(week=1, rusher_player_id="QB1", rush_attempt=1) for _ in range(7)
    ]
    _pbp(plays, monkeypatch)
    out = build_qb_volume(2025).set_index("player_id")

    assert out.loc["QB1", "rush_attempts"] == 7


def test_postseason_is_excluded(monkeypatch):
    _pbp(
        _dropbacks("QB1", 1, 25)
        + [_play(week=20, passer_player_id="QB2", pass_attempt=1, season_type="POST") for _ in range(30)],
        monkeypatch,
    )
    assert set(build_qb_volume(2025)["player_id"]) == {"QB1"}


def test_season_with_no_play_by_play_raises_clearly(monkeypatch):
    monkeypatch.setattr(qb_features, "fetch_pbp_data", lambda s, force_refresh=False: pd.DataFrame())
    with pytest.raises(RuntimeError, match="no play-by-play"):
        build_qb_volume(2099)


# ---- features ----

def _points(player_id, weeks, value=18.0, season=2025):
    return pd.DataFrame(
        [{"season": season, "week": w, "player_id": player_id, "points": value} for w in weeks]
    )


def test_label_is_the_next_played_game():
    volume = pd.DataFrame([_volume_row("QB1", w) for w in (1, 2, 3)])
    points = pd.DataFrame([
        {"season": 2025, "week": 1, "player_id": "QB1", "points": 10.0},
        {"season": 2025, "week": 2, "player_id": "QB1", "points": 20.0},
        {"season": 2025, "week": 3, "player_id": "QB1", "points": 30.0},
    ])
    out = build_qb_features(volume, points, window=2).set_index("week")

    assert out.loc[1, "label_next_week_points"] == 20.0
    assert out.loc[2, "label_next_week_points"] == 30.0
    assert pd.isna(out.loc[3, "label_next_week_points"])  # no next game


def test_trailing_window_does_not_cross_a_season_boundary():
    volume = pd.DataFrame([
        _volume_row("QB1", 17, attempts=50.0, season=2024),
        _volume_row("QB1", 18, attempts=50.0, season=2024),
        _volume_row("QB1", 1, attempts=10.0, season=2025),
        _volume_row("QB1", 2, attempts=10.0, season=2025),
    ])
    points = pd.concat([_points("QB1", [17, 18], season=2024), _points("QB1", [1, 2], season=2025)])
    out = build_qb_features(volume, points, window=2)

    row = out[(out["season"] == 2025) & (out["week"] == 2)].iloc[0]
    assert row["trailing_attempts_avg"] == pytest.approx(10.0)  # not pulled up by 2024


def test_duplicate_player_week_key_raises():
    volume = pd.DataFrame([_volume_row("QB1", 1), _volume_row("QB1", 1), _volume_row("QB1", 2)])
    points = _points("QB1", [1, 2])
    with pytest.raises(ValueError, match="duplicate"):
        build_qb_features(volume, points, window=2)


def test_feature_columns_contract_matches_produced_columns():
    volume = pd.DataFrame([_volume_row("QB1", w) for w in (1, 2, 3)])
    out = build_qb_features(volume, _points("QB1", [1, 2, 3]), window=2)
    for col in qb_feature_columns(2):
        assert col in out.columns


def test_snap_share_is_not_among_the_features():
    # Excluded on purpose: QB snap share is ~binary (median weekly move
    # 0.0 points), so it would contribute noise dressed as a feature.
    assert not any("snap" in c for c in qb_feature_columns())


# ---- explanation ----

def test_explanation_reports_real_weekly_values_not_derived_arithmetic():
    # Regression: an early version subtracted a raw single-game delta from
    # a rolling mean and reported a "before" the player never posted
    # ("pass attempts 50 -> 20"). It must quote actual weeks.
    volume = pd.DataFrame([
        _volume_row("QB1", 1, attempts=48.0),
        _volume_row("QB1", 2, attempts=30.0),
        _volume_row("QB1", 3, attempts=18.0),
    ])
    text = qb_usage_explanation(volume, "QB1", season=2025, week=3, window=2)

    assert "48" in text and "18" in text
    assert "down" in text


def test_explanation_says_so_when_nothing_moved():
    volume = pd.DataFrame([_volume_row("QB1", w, attempts=25.0) for w in (1, 2, 3)])
    text = qb_usage_explanation(volume, "QB1", season=2025, week=3, window=2)
    assert "steady" in text


def test_explanation_needs_two_games():
    volume = pd.DataFrame([_volume_row("QB1", 1)])
    text = qb_usage_explanation(volume, "QB1", season=2025, week=1, window=2)
    assert "Not enough games" in text


def test_explanation_picks_the_metric_that_moved_furthest_past_its_noise_floor():
    # Rush attempts move by 9 (floor 2) = 4.5x; attempts move by 5
    # (floor 4) = 1.25x. Rushing should win despite the smaller raw number.
    volume = pd.DataFrame([
        _volume_row("QB1", 1, attempts=25.0, rush_attempts=1.0),
        _volume_row("QB1", 2, attempts=30.0, rush_attempts=10.0),
    ])
    text = qb_usage_explanation(volume, "QB1", season=2025, week=2, window=2)
    assert "rush attempts" in text
