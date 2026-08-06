import pandas as pd
import pytest

from edge_engine.model.features import build_features
from edge_engine.model.scoring import compute_fantasy_points, compute_points_for_seasons
from edge_engine.roster.models import ScoringSettings


def _weekly_row(**overrides):
    row = {
        "fantasy_points": 10.0,
        "fantasy_points_ppr": 14.0,  # +4 receptions worth of PPR
        "passing_tds": 0,
        "rushing_tds": 0,
        "receiving_tds": 0,
        "interceptions": 0,
        "receiving_fumbles_lost": 0,
        "rushing_fumbles_lost": 0,
        "sack_fumbles_lost": 0,
    }
    row.update(overrides)
    return row


def test_compute_fantasy_points_standard_and_ppr():
    weekly = pd.DataFrame([_weekly_row()])

    standard = compute_fantasy_points(weekly, ScoringSettings(ppr_type="standard"))
    full_ppr = compute_fantasy_points(weekly, ScoringSettings(ppr_type="full_ppr"))
    half_ppr = compute_fantasy_points(weekly, ScoringSettings(ppr_type="half_ppr"))

    assert standard.iloc[0] == 10.0
    assert full_ppr.iloc[0] == 14.0
    assert half_ppr.iloc[0] == 12.0  # exactly halfway between standard and full PPR


def test_compute_fantasy_points_supported_bonus_adds_points():
    weekly = pd.DataFrame([_weekly_row(rushing_tds=2)])
    scoring = ScoringSettings(ppr_type="full_ppr", bonuses={"rush_td": 6})

    points = compute_fantasy_points(weekly, scoring)

    assert points.iloc[0] == 14.0 + 2 * 6


def test_compute_fantasy_points_unsupported_bonus_warns_not_crashes():
    weekly = pd.DataFrame([_weekly_row()])
    scoring = ScoringSettings(ppr_type="full_ppr", bonuses={"bonus_100_rush_yards": 3})

    with pytest.warns(UserWarning, match="bonus_100_rush_yards"):
        points = compute_fantasy_points(weekly, scoring)

    assert points.iloc[0] == 14.0  # unsupported bonus is skipped, not silently guessed at


def _player_week_row(player_id, season, week, **overrides):
    row = {
        "season": season,
        "week": week,
        "player_id": player_id,
        "position": "RB",
        "team": "GB",
        "snap_pct": 0.5,
        "target_share": 0.1,
        "air_yards_share": 0.1,
        "red_zone_touches": 1,
        "red_zone_target_share": 0.1,
    }
    row.update(overrides)
    return row


def test_early_weeks_have_no_trailing_signal_by_design():
    rows = [_player_week_row("P1", 2023, w, snap_pct=0.5) for w in [1, 2, 3, 4]]
    player_week = pd.DataFrame(rows)
    points = pd.Series([10.0, 10.0, 10.0, 10.0])

    features = build_features(player_week, points, window=2)
    by_week = features.set_index("week")

    # window=2: week 1 has only one game so far, not a full trailing window.
    assert pd.isna(by_week.loc[1, "trailing_snap_pct_avg"])
    # Week 2 is the first week with a full window (games 1-2) -> real signal,
    # which is what feeds the *prediction for week 3* ("useful from week 3
    # onward" in the PRD refers to which week gets a usable prediction, not
    # which week's own trailing feature is first populated).
    assert by_week.loc[2, "trailing_snap_pct_avg"] == pytest.approx(0.5)
    assert by_week.loc[3, "trailing_snap_pct_avg"] == pytest.approx(0.5)


def test_trailing_window_does_not_cross_season_boundary():
    rows = [
        _player_week_row("P1", 2022, 17, snap_pct=0.9),
        _player_week_row("P1", 2022, 18, snap_pct=0.9),
        _player_week_row("P1", 2023, 1, snap_pct=0.1),
        _player_week_row("P1", 2023, 2, snap_pct=0.1),
    ]
    player_week = pd.DataFrame(rows)
    points = pd.Series([10.0, 10.0, 1.0, 1.0])

    features = build_features(player_week, points, window=2)
    row_2023_wk2 = features[(features["season"] == 2023) & (features["week"] == 2)].iloc[0]

    # Should only average 2023's own two games (0.1), not pull in 2022's.
    assert row_2023_wk2["trailing_snap_pct_avg"] == pytest.approx(0.1)


def test_label_is_next_games_points_within_same_player_season():
    rows = [_player_week_row("P1", 2023, w) for w in [1, 2, 3]]
    player_week = pd.DataFrame(rows)
    points = pd.Series([5.0, 8.0, 20.0])

    features = build_features(player_week, points, window=2)
    by_week = features.set_index("week")

    assert by_week.loc[1, "label_next_week_points"] == 8.0
    assert by_week.loc[2, "label_next_week_points"] == 20.0
    # Last game of the season has no "next week" to label.
    assert pd.isna(by_week.loc[3, "label_next_week_points"])


def test_duplicate_player_week_key_raises_instead_of_corrupting():
    # A duplicate (player_id, season, week) row would otherwise be
    # silently treated as a second real game by the rolling-window
    # logic -- fabricating a trailing average from the duplicate, and
    # pointing label_next_week_points at the duplicate itself rather
    # than the real next game. Must fail loudly instead.
    rows = [
        _player_week_row("P1", 2023, 1, snap_pct=0.1),
        _player_week_row("P1", 2023, 1, snap_pct=0.1),  # exact duplicate key
        _player_week_row("P1", 2023, 2, snap_pct=0.9),
    ]
    player_week = pd.DataFrame(rows)
    points = pd.Series([5.0, 5.0, 30.0])

    with pytest.raises(ValueError, match="duplicate"):
        build_features(player_week, points, window=2)


def test_compute_points_for_seasons_empty_list_returns_empty_not_a_crash():
    # pd.concat([]) raises "No objects to concatenate" -- an empty
    # seasons list has an obvious correct answer (no points), and is a
    # real, reachable case: history.py's track_flagged_players() is a
    # public function callable with an empty seasons list directly.
    result = compute_points_for_seasons([], ScoringSettings(ppr_type="full_ppr"))
    assert list(result.columns) == ["season", "week", "player_id", "points"]
    assert len(result) == 0
