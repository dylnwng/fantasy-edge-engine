import pandas as pd
import pytest

from edge_engine.model.persistence import (
    compute_persistence_features,
    persistence_feature_columns,
)


def _rows(player_id, snap_pcts, season=2023, start_week=1):
    return [
        {
            "season": season,
            "week": start_week + i,
            "player_id": player_id,
            "snap_pct": s,
            "target_share": 0.1,
        }
        for i, s in enumerate(snap_pcts)
    ]


def test_distinguishes_a_one_week_spike_from_a_sustained_climb():
    # The entire reason this module exists: both players end at 60% and
    # have an identical endpoint-delta trend, but one earned it over
    # three weeks and the other spiked once.
    df = pd.DataFrame(
        _rows("SPIKE", [0.20, 0.20, 0.20, 0.60]) + _rows("CLIMB", [0.20, 0.33, 0.46, 0.60])
    )
    out = compute_persistence_features(df, window=4).set_index(["player_id", "week"])

    assert out.loc[("CLIMB", 4), "snap_pct_rising_streak"] == 3
    assert out.loc[("SPIKE", 4), "snap_pct_rising_streak"] == 1
    assert out.loc[("CLIMB", 4), "snap_pct_pct_weeks_rising"] == pytest.approx(1.0)
    assert out.loc[("SPIKE", 4), "snap_pct_pct_weeks_rising"] == pytest.approx(1 / 3)


def test_streak_does_not_carry_across_a_season_boundary():
    df = pd.DataFrame(
        _rows("P1", [0.30, 0.90], season=2022, start_week=17)
        + _rows("P1", [0.95], season=2023, start_week=1)
    )
    out = compute_persistence_features(df, window=4).set_index(["season", "week"])

    # 2022 wk18 rose off wk17; 2023 wk1 must restart at 0 rather than
    # inheriting a streak from last December.
    assert out.loc[(2022, 18), "snap_pct_rising_streak"] == 1
    assert out.loc[(2023, 1), "snap_pct_rising_streak"] == 0


def test_pct_weeks_rising_is_nan_until_enough_real_deltas_exist():
    # A player's first game has no prior game, so its delta is genuinely
    # undefined -- it must not be silently counted as "not rising",
    # which would understate the earliest reportable value.
    df = pd.DataFrame(_rows("P1", [0.20, 0.33, 0.46, 0.60]))
    out = compute_persistence_features(df, window=4).set_index("week")

    assert pd.isna(out.loc[1, "snap_pct_pct_weeks_rising"])
    assert pd.isna(out.loc[3, "snap_pct_pct_weeks_rising"])  # only 2 real deltas so far
    assert out.loc[4, "snap_pct_pct_weeks_rising"] == pytest.approx(1.0)


def test_flat_usage_produces_no_streak():
    df = pd.DataFrame(_rows("P1", [0.5, 0.5, 0.5, 0.5]))
    out = compute_persistence_features(df, window=4)
    assert (out["snap_pct_rising_streak"] == 0).all()
    assert out["snap_pct_pct_weeks_rising"].dropna().eq(0.0).all()


def test_duplicate_player_week_key_raises_instead_of_inflating_a_streak():
    # A duplicated row would be counted as another real game and inflate
    # the streak -- precisely the signal this module sells.
    df = pd.DataFrame(_rows("P1", [0.2, 0.4]) + _rows("P1", [0.4], start_week=2))
    with pytest.raises(ValueError, match="duplicate"):
        compute_persistence_features(df, window=4)


def test_window_below_two_is_rejected():
    df = pd.DataFrame(_rows("P1", [0.2, 0.4]))
    with pytest.raises(ValueError, match="window must be >= 2"):
        compute_persistence_features(df, window=1)


def test_missing_usage_column_is_rejected_with_a_clear_message():
    df = pd.DataFrame([{"season": 2023, "week": 1, "player_id": "P1", "snap_pct": 0.5}])
    with pytest.raises(ValueError, match="target_share"):
        compute_persistence_features(df)


def test_feature_columns_contract_matches_produced_columns():
    df = pd.DataFrame(_rows("P1", [0.2, 0.4, 0.6, 0.8]))
    out = compute_persistence_features(df, window=4)
    for col in persistence_feature_columns():
        assert col in out.columns
