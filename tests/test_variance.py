import pandas as pd
import pytest

from edge_engine.model.variance import compute_trailing_points_std


def _row(player_id, season, week, points):
    return {"season": season, "week": week, "player_id": player_id, "points": points}


def test_std_below_min_games_is_nan():
    df = pd.DataFrame(
        [_row("P1", 2023, 1, 10.0), _row("P1", 2023, 2, 12.0)]
    )
    out = compute_trailing_points_std(df, window=6, min_games=3)
    assert out["trailing_points_std"].isna().all()


def test_std_matches_hand_calculation():
    df = pd.DataFrame(
        [
            _row("P1", 2023, 1, 10.0),
            _row("P1", 2023, 2, 12.0),
            _row("P1", 2023, 3, 8.0),
        ]
    )
    out = compute_trailing_points_std(df, window=6, min_games=3)
    # sample stdev (ddof=1) of [10, 12, 8]: mean=10, sq devs (0,4,4), /2=4, sqrt=2.0
    assert out.iloc[2]["trailing_points_std"] == pytest.approx(2.0)


def test_std_does_not_cross_season_boundary():
    df = pd.DataFrame(
        [
            _row("P1", 2022, 16, 10.0),
            _row("P1", 2022, 17, 10.0),
            _row("P1", 2022, 18, 10.0),
            _row("P1", 2023, 1, 100.0),  # must not leak into this season's early-week std
            _row("P1", 2023, 2, 100.0),
        ]
    )
    out = compute_trailing_points_std(df, window=6, min_games=3)
    row = out[(out["season"] == 2023) & (out["week"] == 2)].iloc[0]
    assert pd.isna(row["trailing_points_std"])  # only 2 games so far *this season*


def test_std_only_counts_played_games_not_calendar_weeks():
    # A bye week simply isn't a row -- window is over played games, matching
    # features.py's build_features convention.
    df = pd.DataFrame(
        [
            _row("P1", 2023, 1, 10.0),
            _row("P1", 2023, 2, 10.0),
            # week 3 bye -- no row
            _row("P1", 2023, 4, 10.0),
        ]
    )
    out = compute_trailing_points_std(df, window=6, min_games=3)
    row = out[(out["season"] == 2023) & (out["week"] == 4)].iloc[0]
    assert row["trailing_points_std"] == pytest.approx(0.0)  # 3 identical values -> std 0, not NaN
