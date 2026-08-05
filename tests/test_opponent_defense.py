import pandas as pd

from edge_engine.model.opponent_defense import (
    _unpivot_schedule,
    build_points_allowed_table,
    compute_dvp_feature,
)


def test_unpivot_schedule_gives_both_sides_a_row():
    sched = pd.DataFrame(
        [
            {"season": 2024, "week": 1, "game_type": "REG", "home_team": "KC", "away_team": "BAL"},
            {"season": 2024, "week": 1, "game_type": "POST", "home_team": "SF", "away_team": "DAL"},
        ]
    )
    result = _unpivot_schedule(sched)

    # Postseason excluded; regular season game produces exactly 2 rows,
    # one from each team's own perspective.
    assert len(result) == 2
    rows = result.set_index("team")
    assert rows.loc["KC", "opponent"] == "BAL"
    assert rows.loc["BAL", "opponent"] == "KC"


def _pw_row(player_id, week, position, opponent, snap_pct=0.5):
    return {"season": 2023, "week": week, "player_id": player_id, "position": position, "opponent": opponent}


def test_points_allowed_table_rolls_trailing_and_respects_season_boundary():
    # Team WAS's defense faces a WR in weeks 1-3 of 2023 who scores
    # 10, 20, 30 -- with window=2, week 3's trailing average should be
    # (10+20)/2=15 (weeks 1-2, strictly the two games *before* week 3
    # under this module's "as of this row, inclusive" convention means
    # week 2's own value covers weeks 1-2; week 3's covers weeks 2-3).
    player_week = pd.DataFrame(
        [
            _pw_row("P1", 1, "WR", "WAS"),
            _pw_row("P1", 2, "WR", "WAS"),
            _pw_row("P1", 3, "WR", "WAS"),
        ]
    )
    points = pd.Series([10.0, 20.0, 30.0])

    result = build_points_allowed_table(player_week, points, window=2)
    result = result.set_index("week")

    assert pd.isna(result.loc[1, "trailing_points_allowed_avg"])  # only 1 game, window=2 needs 2
    assert result.loc[2, "trailing_points_allowed_avg"] == 15.0  # (10+20)/2
    assert result.loc[3, "trailing_points_allowed_avg"] == 25.0  # (20+30)/2


def test_points_allowed_table_does_not_cross_season_boundary():
    player_week = pd.DataFrame(
        [
            {"season": 2022, "week": 18, "player_id": "P1", "position": "WR", "opponent": "WAS"},
            {"season": 2023, "week": 1, "player_id": "P1", "position": "WR", "opponent": "WAS"},
            {"season": 2023, "week": 2, "player_id": "P1", "position": "WR", "opponent": "WAS"},
        ]
    )
    points = pd.Series([50.0, 1.0, 2.0])  # a huge 2022 outlier that must not leak into 2023

    result = build_points_allowed_table(player_week, points, window=2)
    row_2023_wk2 = result[(result["season"] == 2023) & (result["week"] == 2)].iloc[0]

    assert row_2023_wk2["trailing_points_allowed_avg"] == 1.5  # (1+2)/2, not diluted by the 2022 50-point game


def _merged_row(player_id, week, position, team):
    return {"season": 2023, "week": week, "player_id": player_id, "position": position, "team": team}


def test_compute_dvp_feature_resolves_next_played_game_across_a_bye():
    # P1 plays weeks 1 and 3 (bye in week 2) -- the DvP feature attached
    # to week-1's row must reflect week 3's opponent (the *next played*
    # game), matching label_next_week_points' own bye-skipping convention,
    # not week 2's (which P1 never plays).
    merged = pd.DataFrame([_merged_row("P1", 1, "WR", "KC"), _merged_row("P1", 3, "WR", "KC")])
    schedule = pd.DataFrame(
        [
            {"season": 2023, "week": 1, "team": "KC", "opponent": "BAL"},
            {"season": 2023, "week": 3, "team": "KC", "opponent": "DEN"},  # P1's real next opponent
        ]
    )
    points_allowed = pd.DataFrame(
        [
            {"season": 2023, "week": 1, "team": "BAL", "position": "WR", "trailing_points_allowed_avg": 12.0},
            {"season": 2023, "week": 2, "team": "DEN", "position": "WR", "trailing_points_allowed_avg": 30.0},
        ]
    )

    result = compute_dvp_feature(merged, schedule, points_allowed, target_week=None)
    week1_row = result[result["week"] == 1].iloc[0]

    assert week1_row["dvp_points_allowed_avg"] == 30.0  # DEN's number, not BAL's


def test_compute_dvp_feature_prediction_mode_uses_explicit_target_week():
    merged = pd.DataFrame([_merged_row("P1", 5, "RB", "KC")])
    schedule = pd.DataFrame([{"season": 2023, "week": 6, "team": "KC", "opponent": "LV"}])
    points_allowed = pd.DataFrame(
        [{"season": 2023, "week": 4, "team": "LV", "position": "RB", "trailing_points_allowed_avg": 22.5}]
    )

    result = compute_dvp_feature(merged, schedule, points_allowed, target_week=6)

    assert len(result) == 1
    assert result.iloc[0]["dvp_points_allowed_avg"] == 22.5


def test_compute_dvp_feature_never_uses_a_same_or_future_week_value():
    # points_allowed only has an entry for week 6 itself (the week being
    # predicted) -- an as-of-strictly-before join must reject it, not
    # leak the very outcome the model is trying to predict.
    merged = pd.DataFrame([_merged_row("P1", 5, "RB", "KC")])
    schedule = pd.DataFrame([{"season": 2023, "week": 6, "team": "KC", "opponent": "LV"}])
    points_allowed = pd.DataFrame(
        [{"season": 2023, "week": 6, "team": "LV", "position": "RB", "trailing_points_allowed_avg": 99.0}]
    )

    result = compute_dvp_feature(merged, schedule, points_allowed, target_week=6)

    assert pd.isna(result.iloc[0]["dvp_points_allowed_avg"])
