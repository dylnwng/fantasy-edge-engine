import pandas as pd
import pytest

from edge_engine.insights.divergence import (
    MIN_POOL_SIZE,
    classify,
    compute_divergence,
)
from edge_engine.insights.exposure import (
    find_bye_collisions,
    injury_exposure_notes,
    team_stacking_notes,
)
from edge_engine.insights.report import build_report
from edge_engine.insights.scarcity import compute_scarcity, remaining_weeks
from edge_engine.model.injury_context import INJURY_COLUMNS
from edge_engine.roster.models import Player


def _pw_row(player_id, week, position="WR", snap_pct=0.5, target_share=0.2,
            air_yards_share=0.2, red_zone_touches=1, season=2025):
    return {
        "season": season, "week": week, "player_id": player_id, "position": position,
        "team": "KC", "snap_pct": snap_pct, "target_share": target_share,
        "air_yards_share": air_yards_share, "red_zone_touches": red_zone_touches,
    }


def _pool(n, week=3, position="WR", **overrides):
    """A positional pool big enough to z-score against."""
    return [_pw_row(f"POOL{i}", week, position=position, **overrides) for i in range(n)]


def _player(name, position="WR", player_id=None, bye_week=None, team="KC"):
    return Player(
        name=name, position=position, team=team, player_id=player_id or name,
        bye_week=bye_week, match_status="matched",
    )


# ---- divergence ----

def test_high_usage_low_points_reads_as_undervalued():
    rows = _pool(MIN_POOL_SIZE) + [
        _pw_row("STAR", 3, snap_pct=0.99, target_share=0.9, air_yards_share=0.9, red_zone_touches=9)
    ]
    df = pd.DataFrame(rows)
    # Everyone scores the same; only STAR's usage is elevated.
    points = pd.Series([10.0] * len(df))

    result, _ = compute_divergence(df, points, season=2025, week=3, window=1)
    star = next(d for d in result if d.player_id == "STAR")

    assert star.usage_z > 0
    assert star.divergence > 1.0
    assert star.signal == "undervalued"


def test_low_usage_high_points_reads_as_overvalued():
    rows = _pool(MIN_POOL_SIZE) + [
        _pw_row("LUCKY", 3, snap_pct=0.05, target_share=0.01, air_yards_share=0.01, red_zone_touches=0)
    ]
    df = pd.DataFrame(rows)
    points = pd.Series([10.0] * MIN_POOL_SIZE + [40.0])  # huge points, no usage

    result, _ = compute_divergence(df, points, season=2025, week=3, window=1)
    lucky = next(d for d in result if d.player_id == "LUCKY")

    assert lucky.divergence < -1.0
    assert lucky.signal == "overvalued"


def test_positions_are_pooled_separately():
    # A 60% snap share means different things at RB and WR; pooling them
    # together would smear the distributions.
    rows = _pool(MIN_POOL_SIZE, position="WR", snap_pct=0.9) + _pool(
        MIN_POOL_SIZE, position="RB", snap_pct=0.2
    )
    # Rename so ids don't collide across the two pools.
    for i, r in enumerate(rows):
        r["player_id"] = f"P{i}"
    df = pd.DataFrame(rows)
    points = pd.Series([10.0] * len(df))

    result, _ = compute_divergence(df, points, season=2025, week=3, window=1)

    # Within each pool everyone is identical, so every z-score is 0 --
    # which only holds if the pools were separated.
    assert all(abs(d.usage_z) < 1e-9 for d in result)


def test_small_positional_pool_is_skipped_and_recorded_as_a_gap():
    df = pd.DataFrame([_pw_row("A", 3), _pw_row("B", 3)])  # pool of 2
    points = pd.Series([10.0, 20.0])

    result, gaps = compute_divergence(df, points, season=2025, week=3, window=1)

    assert result == []
    assert any("pool too small" in g.lower() for g in gaps)


def test_missing_week_in_the_window_is_recorded_as_a_gap():
    rows = _pool(MIN_POOL_SIZE, week=3)  # window wants weeks 1-3, only 3 present
    df = pd.DataFrame(rows)
    points = pd.Series([10.0] * len(df))

    _, gaps = compute_divergence(df, points, season=2025, week=3, window=3)

    assert any("week" in g.lower() and "1" in g for g in gaps)


def test_duplicate_player_week_key_raises():
    rows = _pool(MIN_POOL_SIZE) + [_pw_row("DUP", 3), _pw_row("DUP", 3)]
    df = pd.DataFrame(rows)
    points = pd.Series([10.0] * len(df))

    with pytest.raises(ValueError, match="duplicate"):
        compute_divergence(df, points, season=2025, week=3, window=1)


def test_zero_variance_pool_yields_zero_not_nan():
    df = pd.DataFrame(_pool(MIN_POOL_SIZE))
    points = pd.Series([10.0] * MIN_POOL_SIZE)

    result, _ = compute_divergence(df, points, season=2025, week=3, window=1)

    assert len(result) == MIN_POOL_SIZE
    assert all(d.divergence == 0.0 and d.signal == "neutral" for d in result)


def test_classify_respects_the_neutral_band():
    assert classify(1.5) == "undervalued"
    assert classify(-1.5) == "overvalued"
    assert classify(0.9) == "neutral"
    assert classify(-1.0) == "neutral"  # boundary is inclusive-neutral


# ---- scarcity ----

def _scored(rows):
    return pd.DataFrame(rows, columns=["player_id", "position", "predicted_score"])


def test_surplus_counts_startable_players_against_required_slots():
    rostered = [_player("A", "RB"), _player("B", "RB"), _player("C", "RB")]
    scored = _scored([("A", "RB", 20.0), ("B", "RB", 15.0), ("C", "RB", 1.0),
                      ("X", "RB", 10.0), ("Y", "RB", 12.0)])

    result, _ = compute_scarcity(rostered, {"RB": 2}, scored, remaining_weeks=[])
    rb = next(r for r in result if r.position == "RB")

    # Median of the scored RB pool is 12.0; A and B clear it, C doesn't.
    assert rb.startable_now == 2
    assert rb.required == 2
    assert rb.surplus_now == 0


def test_bye_adjusted_worst_week_finds_the_collision_week():
    rostered = [_player("A", "RB", bye_week=9), _player("B", "RB", bye_week=9)]
    scored = _scored([("A", "RB", 20.0), ("B", "RB", 20.0), ("X", "RB", 5.0)])

    result, _ = compute_scarcity(rostered, {"RB": 2}, scored, remaining_weeks=[8, 9, 10])
    rb = next(r for r in result if r.position == "RB")

    assert rb.surplus_now == 0  # both startable now
    assert rb.worst_week == 9  # both on bye
    assert rb.worst_week_surplus == -2


def test_no_bye_data_skips_the_scan_rather_than_assuming_no_byes():
    weeks, gaps = remaining_weeks(current_week=5, byes_known=False)
    assert weeks == []
    assert any("bye" in g.lower() for g in gaps)


def test_empty_scored_frame_explains_itself_as_a_timing_problem():
    # Must NOT claim "the model does not score RB" -- that would be a
    # false explanation of an early-season trailing-window problem.
    rostered = [_player("A", "RB")]
    result, gaps = compute_scarcity(rostered, {"RB": 2}, _scored([]), remaining_weeks=[])

    assert any("no scores for this week at all" in g for g in gaps)
    assert not any("does not score" in g for g in gaps)


# ---- exposure ----

def test_bye_collision_requires_same_position():
    rostered = [
        _player("RB1", "RB", bye_week=9),
        _player("RB2", "RB", bye_week=9),
        _player("WR1", "WR", bye_week=9),  # same week, different position
    ]
    collisions = find_bye_collisions(rostered)

    assert len(collisions) == 1
    assert collisions[0].week == 9
    assert collisions[0].position == "RB"
    assert collisions[0].players == ("RB1", "RB2")


def test_team_stacking_note_quotes_the_measured_correlation():
    rostered = [_player(f"P{i}", "WR", team="KC") for i in range(3)]
    notes = team_stacking_notes(rostered)

    assert len(notes) == 1
    # Showing how weak the signal is IS the point of showing it.
    assert "0.0405" in notes[0]


def test_no_stacking_note_below_threshold():
    assert team_stacking_notes([_player("A", team="KC"), _player("B", team="KC")]) == []


def test_injury_exposure_counts_only_rostered_players():
    # Note the note uses the ROSTER's name for the player, not the injury
    # report's -- the manager recognises their own roster's spelling.
    rostered = [_player("Mine", "RB", player_id="MINE")]
    injuries = pd.DataFrame(
        [
            {"season": 2025, "week": 5, "team": "KC", "gsis_id": "MINE", "full_name": "Mine",
             "position": "RB", "report_status": "Questionable", "report_primary_injury": "Ankle"},
            {"season": 2025, "week": 5, "team": "KC", "gsis_id": "THEIRS", "full_name": "Theirs",
             "position": "RB", "report_status": "Out", "report_primary_injury": "Knee"},
        ],
        columns=INJURY_COLUMNS,
    )
    notes, _ = injury_exposure_notes(rostered, injuries, season=2025, week=5)

    assert len(notes) == 1
    assert "Questionable" in notes[0] and "Mine" in notes[0]
    assert "Theirs" not in notes[0]


# ---- report ----

def _report_inputs():
    rows = _pool(MIN_POOL_SIZE) + [
        _pw_row("MINE", 3, snap_pct=0.99, target_share=0.9, air_yards_share=0.9, red_zone_touches=9)
    ]
    df = pd.DataFrame(rows)
    points = pd.Series([10.0] * len(df))
    scored = _scored([("MINE", "WR", 20.0), ("POOL0", "WR", 5.0)])
    return df, points, scored


def test_report_context_is_always_text_never_numeric():
    df, points, scored = _report_inputs()
    report = build_report(
        rostered=[_player("Mine", "WR", player_id="MINE", bye_week=7)],
        player_week=df, points=points, scored=scored,
        lineup_slots={"WR": 2}, injuries=pd.DataFrame(columns=INJURY_COLUMNS),
        season=2025, week=3, window=1,
    )
    for p in report.players:
        for item in p.context:
            assert isinstance(item, str)


def test_report_flags_an_empty_roster_rather_than_showing_silent_zeros():
    df, points, scored = _report_inputs()
    report = build_report(
        rostered=[], player_week=df, points=points, scored=scored,
        lineup_slots={"WR": 2}, injuries=pd.DataFrame(columns=INJURY_COLUMNS),
        season=2025, week=3, window=1,
    )
    assert report.players == []
    assert any("0 rostered players" in g for g in report.data_gaps)


def test_report_records_a_gap_for_players_with_no_usage_data():
    df, points, scored = _report_inputs()
    report = build_report(
        rostered=[_player("Ghost", "WR", player_id="NOT_IN_DATA")],
        player_week=df, points=points, scored=scored,
        lineup_slots={"WR": 2}, injuries=pd.DataFrame(columns=INJURY_COLUMNS),
        season=2025, week=3, window=1,
    )
    assert report.players[0].signal == "no_data"
    assert report.players[0].divergence is None
    assert any("no usage data" in g for g in report.data_gaps)


def test_report_sorts_by_absolute_divergence_with_no_data_last():
    df, points, scored = _report_inputs()
    report = build_report(
        rostered=[
            _player("Ghost", "WR", player_id="NOT_IN_DATA"),
            _player("Mine", "WR", player_id="MINE"),
        ],
        player_week=df, points=points, scored=scored,
        lineup_slots={"WR": 2}, injuries=pd.DataFrame(columns=INJURY_COLUMNS),
        season=2025, week=3, window=1,
    )
    assert report.players[0].name == "Mine"
    assert report.players[-1].name == "Ghost"
