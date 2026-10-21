import pandas as pd
import pytest

from edge_engine.model.labels import attach_forward_label, forward_window_label


def _points(rows):
    return pd.DataFrame(rows, columns=["season", "week", "player_id", "points"])


def _by_week(out, player_id="P1"):
    return out[out["player_id"] == player_id].set_index("week")


# ---- the forward window ----

def test_label_is_the_mean_of_the_next_three_games():
    points = _points([
        (2025, 1, "P1", 10.0),
        (2025, 2, "P1", 12.0),
        (2025, 3, "P1", 18.0),
        (2025, 4, "P1", 24.0),
    ])

    out = _by_week(forward_window_label(points, horizon=3))

    assert out.loc[1, "label_forward_points"] == pytest.approx((12 + 18 + 24) / 3)
    assert out.loc[1, "forward_games"] == 3


def test_the_window_never_crosses_a_season_boundary():
    points = _points([
        (2024, 16, "P1", 5.0), (2024, 17, "P1", 5.0),
        (2025, 1, "P1", 40.0), (2025, 2, "P1", 40.0), (2025, 3, "P1", 40.0),
    ])

    out = forward_window_label(points, horizon=3)
    row_2024_wk16 = out[(out["season"] == 2024) & (out["week"] == 16)].iloc[0]

    # Only one 2024 game follows; next season's 40s must not be pulled in.
    assert row_2024_wk16["forward_games"] == 1
    assert pd.isna(row_2024_wk16["label_forward_points"])  # full window required


def test_the_window_counts_played_games_not_calendar_weeks():
    # P1 misses week 3 entirely. The window is weeks 2, 4, 5 -- a bye is
    # skipped, never treated as a zero, matching features.py's trailing
    # window convention.
    points = _points([
        (2025, 1, "P1", 10.0), (2025, 2, "P1", 20.0),
        (2025, 4, "P1", 20.0), (2025, 5, "P1", 20.0),
    ])

    out = _by_week(forward_window_label(points, horizon=3))

    assert out.loc[1, "forward_games"] == 3
    assert out.loc[1, "label_forward_points"] == pytest.approx(20.0)


def test_one_players_future_is_never_credited_to_another():
    points = _points([
        (2025, 1, "P1", 10.0), (2025, 2, "P1", 11.0), (2025, 3, "P1", 11.0), (2025, 4, "P1", 11.0),
        (2025, 1, "P2", 10.0), (2025, 2, "P2", 99.0), (2025, 3, "P2", 99.0), (2025, 4, "P2", 99.0),
    ])

    out = _by_week(forward_window_label(points, horizon=3), "P1")

    assert out.loc[1, "label_forward_points"] == pytest.approx(11.0)


# ---- partial windows: the two intended behaviours ----

def test_a_partial_window_is_null_when_a_full_window_is_required():
    # For TRAINING: a label averaged over one game and one averaged over
    # three are different quantities, and mixing them teaches the model
    # that the target's variance changes arbitrarily near week 17.
    points = _points([(2025, 16, "P1", 10.0), (2025, 17, "P1", 30.0)])

    out = _by_week(forward_window_label(points, horizon=3, require_full_window=True))

    assert out.loc[16, "forward_games"] == 1
    assert pd.isna(out.loc[16, "label_forward_points"])


def test_a_partial_window_is_graded_on_what_exists_when_not_required():
    # For GRADING past flags, as history.py does: discarding a real
    # outcome is worse than judging it on a shorter window.
    points = _points([(2025, 16, "P1", 10.0), (2025, 17, "P1", 30.0)])

    out = _by_week(forward_window_label(points, horizon=3, require_full_window=False))

    assert out.loc[16, "forward_games"] == 1
    assert out.loc[16, "label_forward_points"] == pytest.approx(30.0)


def test_the_final_game_of_a_season_has_no_forward_window_either_way():
    points = _points([(2025, 16, "P1", 10.0), (2025, 17, "P1", 30.0)])

    for require in (True, False):
        out = _by_week(forward_window_label(points, horizon=3, require_full_window=require))
        assert out.loc[17, "forward_games"] == 0
        assert pd.isna(out.loc[17, "label_forward_points"])


# ---- guards ----

def test_a_duplicate_key_raises_instead_of_fabricating_a_label():
    # A duplicated row shifts the window onto the duplicate rather than
    # the real next game -- same corruption build_features refuses.
    points = _points([
        (2025, 1, "P1", 10.0), (2025, 1, "P1", 10.0),
        (2025, 2, "P1", 20.0), (2025, 3, "P1", 20.0), (2025, 4, "P1", 20.0),
    ])

    with pytest.raises(ValueError, match="duplicate"):
        forward_window_label(points, horizon=3)


def test_a_nonsense_horizon_raises():
    points = _points([(2025, 1, "P1", 10.0)])

    with pytest.raises(ValueError, match="horizon"):
        forward_window_label(points, horizon=0)


def test_a_points_table_missing_columns_names_what_is_missing():
    with pytest.raises(ValueError, match="missing"):
        forward_window_label(pd.DataFrame({"season": [2025], "week": [1]}), horizon=3)


def test_horizon_of_one_reproduces_the_next_game():
    points = _points([(2025, 1, "P1", 10.0), (2025, 2, "P1", 22.0)])

    out = _by_week(forward_window_label(points, horizon=1))

    assert out.loc[1, "label_forward_points"] == pytest.approx(22.0)


# ---- attaching to a feature table ----

def _features(rows):
    return pd.DataFrame(rows, columns=["season", "week", "player_id", "trailing_points_avg"])


def test_attach_keeps_every_feature_row_and_nulls_ungradeable_labels():
    # Left join on purpose: rows must not vanish mid-pipeline: the caller
    # decides what to drop.
    features = _features([(2025, w, "P1", 10.0) for w in (1, 2, 3, 4)])
    points = _points([(2025, w, "P1", 10.0) for w in (1, 2, 3, 4)])

    out = attach_forward_label(features, points, horizon=3)

    assert len(out) == len(features)
    assert "label_forward_points" in out.columns
    # Week 1 has weeks 2-4 following; weeks 2+ do not have a full window.
    by_week = out.set_index("week")
    assert by_week.loc[1, "label_forward_points"] == pytest.approx(10.0)
    assert pd.isna(by_week.loc[3, "label_forward_points"])


def test_attach_does_not_disturb_existing_feature_columns():
    features = _features([(2025, w, "P1", float(w)) for w in (1, 2, 3, 4)])
    points = _points([(2025, w, "P1", 10.0) for w in (1, 2, 3, 4)])

    out = attach_forward_label(features, points, horizon=3)

    assert list(out["trailing_points_avg"]) == [1.0, 2.0, 3.0, 4.0]
