import pandas as pd
import pytest

from edge_engine.ranking.output import confidence_tier
from edge_engine.ranking.usage_trend import get_usage_trend, usage_trend_explanation


@pytest.mark.parametrize(
    "predicted,baseline,expected",
    [
        (10.0, 10.0, "Low"),  # margin 0
        (12.9, 10.0, "Low"),  # margin 2.9, just under flag_margin
        (13.0, 10.0, "Medium"),  # margin exactly flag_margin
        (15.9, 10.0, "Medium"),  # margin 5.9, just under 2x flag_margin
        (16.0, 10.0, "High"),  # margin exactly 2x flag_margin
        (25.0, 10.0, "High"),
    ],
)
def test_confidence_tier_thresholds(predicted, baseline, expected):
    assert confidence_tier(predicted, baseline, flag_margin=3.0) == expected


def _pw_row(week, snap_pct, target_share, red_zone_touches):
    return {
        "season": 2023,
        "week": week,
        "player_id": "P1",
        "snap_pct": snap_pct,
        "target_share": target_share,
        "red_zone_touches": red_zone_touches,
    }


def test_usage_trend_cites_biggest_percent_mover():
    player_week = pd.DataFrame(
        [
            _pw_row(1, 0.30, 0.10, 1),
            _pw_row(2, 0.65, 0.12, 1),  # snap share jumped, target share barely moved
        ]
    )
    explanation = usage_trend_explanation(player_week, "P1", 2023, 2, window=2)

    assert "snap share" in explanation
    assert "30%" in explanation and "65%" in explanation
    assert "target share" not in explanation


def test_usage_trend_falls_back_to_red_zone_touches_when_shares_are_flat():
    player_week = pd.DataFrame(
        [
            _pw_row(1, 0.40, 0.10, 0),
            _pw_row(2, 0.41, 0.11, 3),  # shares flat, red zone touches jumped
        ]
    )
    explanation = usage_trend_explanation(player_week, "P1", 2023, 2, window=2)

    assert "red zone touches" in explanation
    assert "0" in explanation and "3" in explanation


def test_usage_trend_handles_limited_history_without_crashing():
    player_week = pd.DataFrame([_pw_row(1, 0.40, 0.10, 0)])
    explanation = usage_trend_explanation(player_week, "P1", 2023, 1, window=2)

    assert explanation  # must always be non-empty, per requirement 4's acceptance criteria
    assert "Limited" in explanation


def test_usage_trend_steady_case_is_still_non_empty():
    player_week = pd.DataFrame(
        [
            _pw_row(1, 0.40, 0.10, 1),
            _pw_row(2, 0.41, 0.11, 1),  # nothing moved meaningfully
        ]
    )
    explanation = usage_trend_explanation(player_week, "P1", 2023, 2, window=2)

    assert explanation
    assert "steady" in explanation


def test_get_usage_trend_exposes_raw_series_for_visualization():
    player_week = pd.DataFrame(
        [
            _pw_row(1, 0.30, 0.10, 1),
            _pw_row(2, 0.65, 0.12, 1),
        ]
    )
    trend = get_usage_trend(player_week, "P1", 2023, 2, window=2)

    assert trend.insufficient_history is False
    assert trend.moved_meaningfully is True
    assert trend.label == "snap share"
    assert trend.values == [0.30, 0.65]
    assert trend.weeks == [1, 2]
    assert trend.is_percentage is True
    assert trend.n_weeks_span == 1


def test_get_usage_trend_insufficient_history_has_no_series():
    player_week = pd.DataFrame([_pw_row(1, 0.40, 0.10, 0)])
    trend = get_usage_trend(player_week, "P1", 2023, 1, window=2)

    assert trend.insufficient_history is True
    assert trend.values == []
