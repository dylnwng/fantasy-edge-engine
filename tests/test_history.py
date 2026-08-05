import pandas as pd
import pytest

from edge_engine.model.history import _attach_following_performance


def _points_row(player_id, season, week, points):
    return {"season": season, "week": week, "player_id": player_id, "points": points}


def test_following_performance_averages_up_to_three_future_games():
    points_table = pd.DataFrame(
        [
            _points_row("P1", 2023, 1, 5.0),
            _points_row("P1", 2023, 2, 10.0),
            _points_row("P1", 2023, 3, 20.0),
            _points_row("P1", 2023, 4, 30.0),
        ]
    )
    flagged = pd.DataFrame([{"season": 2023, "week": 1, "player_id": "P1"}])

    result = _attach_following_performance(flagged, points_table, window=3)

    assert result.loc[0, "following_games_played"] == 3
    assert result.loc[0, "following_avg_points"] == pytest.approx((10.0 + 20.0 + 30.0) / 3)


def test_following_performance_truncates_at_end_of_season():
    # Flag is at the second-to-last game of the season -- only 1 game left.
    points_table = pd.DataFrame(
        [
            _points_row("P1", 2023, 17, 8.0),
            _points_row("P1", 2023, 18, 12.0),
        ]
    )
    flagged = pd.DataFrame([{"season": 2023, "week": 17, "player_id": "P1"}])

    result = _attach_following_performance(flagged, points_table, window=3)

    assert result.loc[0, "following_games_played"] == 1
    assert result.loc[0, "following_avg_points"] == 12.0


def test_following_performance_does_not_cross_season_boundary():
    points_table = pd.DataFrame(
        [
            _points_row("P1", 2022, 18, 9.0),
            _points_row("P1", 2023, 1, 100.0),  # must not leak into 2022 wk18's following window
        ]
    )
    flagged = pd.DataFrame([{"season": 2022, "week": 18, "player_id": "P1"}])

    result = _attach_following_performance(flagged, points_table, window=3)

    assert result.loc[0, "following_games_played"] == 0


def test_zero_following_games_yields_nan_not_a_crash():
    points_table = pd.DataFrame([_points_row("P1", 2023, 18, 9.0)])
    flagged = pd.DataFrame([{"season": 2023, "week": 18, "player_id": "P1"}])

    result = _attach_following_performance(flagged, points_table, window=3)

    assert result.loc[0, "following_games_played"] == 0
    assert pd.isna(result.loc[0, "following_avg_points"])
