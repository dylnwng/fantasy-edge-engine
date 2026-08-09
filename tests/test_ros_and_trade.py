import pandas as pd
import pytest

from edge_engine.draft.board import BoardPlayer
from edge_engine.draft.server import ServerState, _value_bar, manual_pick
from edge_engine.draft.tracker import DraftState
from edge_engine.trade.compare import TradeComparison, compare_trade
from edge_engine.trade.ros import build_ros_table, intervals_overlap, positional_prior, project


def _row(player_id, week, points, position="RB", season=2025):
    return {
        "season": season, "week": week, "player_id": player_id, "position": position,
        "player_name": player_id, "team": "KC", "snap_pct": 0.6, "target_share": 0.2,
        "air_yards_share": 0.2, "red_zone_touches": 2, "points": points,
    }


def _frame(rows):
    df = pd.DataFrame(rows)
    return df, df["points"]


# ---- ROS table construction ----

def test_features_and_target_never_overlap():
    # Features come from weeks <= origin, the target from weeks > origin.
    # Any overlap would leak the answer into the inputs.
    rows = [_row("P1", w, points=float(w)) for w in range(1, 11)]
    df, points = _frame(rows)

    table = build_ros_table(df, points, origin_week=5)
    r = table.iloc[0]

    assert r["games_played"] == 5
    assert r["ppg_to_date"] == pytest.approx(3.0)   # mean of 1..5
    assert r["ros_ppg"] == pytest.approx(8.0)       # mean of 6..10
    assert r["ros_games"] == 5


def test_empty_when_no_future_weeks_remain():
    rows = [_row("P1", w, points=10.0) for w in range(1, 5)]
    df, points = _frame(rows)
    assert build_ros_table(df, points, origin_week=10).empty


def test_duplicate_player_week_key_raises():
    rows = [_row("P1", 1, 10.0), _row("P1", 1, 10.0), _row("P1", 2, 10.0), _row("P1", 3, 10.0)]
    df, points = _frame(rows)
    with pytest.raises(ValueError, match="duplicate"):
        build_ros_table(df, points, origin_week=1)


# ---- shrinkage ----

def test_thin_samples_shrink_toward_the_prior_more_than_thick_ones():
    rows = (
        [_row("THIN", w, 30.0) for w in (1, 6, 7, 8)]          # 1 game before origin
        + [_row("THICK", w, 30.0) for w in range(1, 9)]         # 5 games before origin
        + [_row(f"F{i}", w, 5.0) for i in range(8) for w in range(1, 9)]  # pool
    )
    df, points = _frame(rows)
    table = build_ros_table(df, points, origin_week=5)
    prior = positional_prior(table)
    by_id = {p.player_id: p for p in project(table, prior)}

    # Both averaged 30 ppg, but the thin one is pulled further toward the
    # (much lower) positional prior.
    assert by_id["THIN"].shrinkage_weight < by_id["THICK"].shrinkage_weight
    assert by_id["THIN"].estimate < by_id["THICK"].estimate


def test_every_projection_carries_an_interval_containing_its_estimate():
    # The interval is a required field, never Optional -- a bare ROS point
    # estimate is the most likely route to overconfident output.
    rows = [_row(f"P{i}", w, float(5 + i)) for i in range(9) for w in range(1, 11)]
    df, points = _frame(rows)
    table = build_ros_table(df, points, origin_week=5)

    for p in project(table, positional_prior(table)):
        lo, hi = p.interval
        assert lo <= p.estimate <= hi
        assert hi > lo


def test_overlapping_intervals_are_detected_as_indistinguishable():
    rows = [_row(f"P{i}", w, float(10 + (i % 2))) for i in range(9) for w in range(1, 11)]
    df, points = _frame(rows)
    table = build_ros_table(df, points, origin_week=5)
    projections = project(table, positional_prior(table))

    # Near-identical players must read as indistinguishable, not ranked.
    assert intervals_overlap(projections[0], projections[1])


# ---- trade surfacer ----

def _trade_frame():
    rows = [_row(f"P{i}", w, float(5 + i), position="RB") for i in range(9) for w in range(1, 6)]
    return _frame(rows)


def test_trade_comparison_exposes_no_verdict_field():
    # §3.6: no verdict, no fairness score, no winner label. Enforced on
    # the type, not just the renderer.
    fields = set(TradeComparison.__dataclass_fields__)
    for forbidden in ("verdict", "recommendation", "fairness_score", "winner", "grade"):
        assert forbidden not in fields


def test_trade_comparison_summarises_both_sides_from_observed_data():
    df, points = _trade_frame()
    result = compare_trade(["P1"], ["P8"], df, points, season=2025, week=5, window=3)

    assert [p.name for p in result.outgoing.players] == ["P1"]
    assert [p.name for p in result.incoming.players] == ["P8"]
    assert result.incoming.players[0].ppg_to_date > result.outgoing.players[0].ppg_to_date


def test_unresolved_player_is_surfaced_not_dropped():
    df, points = _trade_frame()
    result = compare_trade(["Nobody At All"], ["P1"], df, points, season=2025, week=5, window=3)

    assert len(result.outgoing.players) == 1
    assert result.outgoing.players[0].signal == "no_data"
    assert any("did not resolve" in g for g in result.data_gaps)


def test_context_states_that_no_projection_is_offered():
    df, points = _trade_frame()
    result = compare_trade(["P1"], ["P2"], df, points, season=2025, week=5, window=3)
    joined = " ".join(result.context)

    assert "not a projection" in joined
    assert "No verdict" in joined


# ---- draft server state ----

def _bp(name, adp=1.0, position="RB", tier=1, low_confidence=False):
    return BoardPlayer(name=name, position=position, team="KC", adp=adp,
                       position_rank=1, tier=tier, player_id=name,
                       low_confidence=low_confidence)


def _state(board=None):
    return ServerState(
        draft=DraftState(board=board or [_bp("A", 1.0), _bp("B", 2.0)]),
        lineup_slots={"RB": 2, "BENCH": 5},
    )


def test_value_bar_is_a_length_between_zero_and_one():
    board = [_bp("Early", 1.0), _bp("Late", 100.0)]
    assert _value_bar(board[0], board) == 1.0
    assert _value_bar(board[1], board) == 0.0


def test_value_bar_handles_a_single_player_board_without_dividing_by_zero():
    board = [_bp("Only", 5.0)]
    assert _value_bar(board[0], board) == 1.0


def test_snapshot_reports_staleness_and_marks_it_past_the_threshold():
    state = _state()
    fresh = state.snapshot()
    assert fresh["stale"] is False

    state.last_poll_ok -= 30
    assert state.snapshot()["stale"] is True


def test_manual_pick_removes_the_player_and_can_mark_it_as_mine():
    state = _state()
    assert manual_pick(state, "A", mine=True) is True

    snap = state.snapshot()
    assert snap["picks_made"] == 1
    assert [p["name"] for p in snap["my_roster"]] == ["A"]
    assert "A" not in [p["name"] for p in snap["best_available"]]


def test_manual_pick_rejects_a_name_not_on_the_board():
    state = _state()
    assert manual_pick(state, "Not On Board") is False
    assert state.snapshot()["picks_made"] == 0


def test_snapshot_counts_down_unfilled_starting_slots():
    state = _state()
    manual_pick(state, "A", mine=True)
    assert state.snapshot()["unfilled"]["RB"] == 1
