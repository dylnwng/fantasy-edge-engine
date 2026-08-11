import pandas as pd
import pytest

from edge_engine.model.injury_context import INJURY_COLUMNS, _teammates_ahead
from edge_engine.model.vacated_opportunity import (
    attach_vacated_opportunity,
    build_vacated_opportunity,
)


def _pw(rows):
    return pd.DataFrame(rows, columns=["season", "week", "player_id", "team", "position", "snap_pct"])


def _row(player_id, week, snap_pct, team="GB", position="RB", season=2025):
    return (season, week, player_id, team, position, snap_pct)


def _injuries(rows):
    """rows: (season, week, gsis_id, report_status)"""
    full = [
        {"season": s, "week": w, "team": "GB", "gsis_id": g, "full_name": g,
         "position": "RB", "report_status": st, "report_primary_injury": "Hamstring"}
        for s, w, g, st in rows
    ]
    return pd.DataFrame(full, columns=INJURY_COLUMNS)


def _lookup(out, player_id, week):
    row = out[(out["player_id"] == player_id) & (out["week"] == week)].iloc[0]
    return row["vacated_snap_share"], row["blocker_severity"]


# ---- the core behaviour ----

def test_an_injured_starter_ahead_vacates_his_snap_share():
    pw = _pw([
        _row("STARTER", 1, 0.80), _row("BACKUP", 1, 0.20),
        _row("STARTER", 2, 0.80), _row("BACKUP", 2, 0.20),
        _row("STARTER", 3, 0.80), _row("BACKUP", 3, 0.20),
    ])
    injuries = _injuries([(2025, 3, "STARTER", "Out")])

    out = build_vacated_opportunity(pw, injuries, usage_gap=0.15, lookback=2)
    vacated, severity = _lookup(out, "BACKUP", 3)

    assert vacated == pytest.approx(0.80)  # the starter's trailing share
    assert severity == 3  # Out


def test_a_healthy_starter_vacates_nothing():
    pw = _pw([
        _row("STARTER", w, 0.80) for w in (1, 2, 3)
    ] + [_row("BACKUP", w, 0.20) for w in (1, 2, 3)])

    out = build_vacated_opportunity(pw, _injuries([]), usage_gap=0.15, lookback=2)
    vacated, severity = _lookup(out, "BACKUP", 3)

    assert vacated == 0.0
    assert severity == 0


def test_an_injured_teammate_BEHIND_you_vacates_nothing():
    # A hurt player with less usage than you frees up no opportunity for
    # you -- the "ahead of" gate is the whole point.
    pw = _pw([
        _row("STARTER", w, 0.80) for w in (1, 2, 3)
    ] + [_row("BACKUP", w, 0.20) for w in (1, 2, 3)])
    injuries = _injuries([(2025, 3, "BACKUP", "Out")])

    out = build_vacated_opportunity(pw, injuries, usage_gap=0.15, lookback=2)
    vacated, _ = _lookup(out, "STARTER", 3)

    assert vacated == 0.0


def test_a_teammate_inside_the_usage_gap_does_not_count_as_ahead():
    # 0.30 vs 0.20 is a 0.10 gap, under the 0.15 threshold: that's a
    # timeshare, not a player blocking you.
    pw = _pw([
        _row("A", w, 0.30) for w in (1, 2, 3)
    ] + [_row("B", w, 0.20) for w in (1, 2, 3)])
    injuries = _injuries([(2025, 3, "A", "Out")])

    out = build_vacated_opportunity(pw, injuries, usage_gap=0.15, lookback=2)
    vacated, _ = _lookup(out, "B", 3)

    assert vacated == 0.0


def test_multiple_injured_blockers_sum_their_shares():
    pw = _pw([
        _row("RB1", w, 0.50) for w in (1, 2, 3)
    ] + [_row("RB2", w, 0.40) for w in (1, 2, 3)]
      + [_row("RB3", w, 0.10) for w in (1, 2, 3)])
    injuries = _injuries([(2025, 3, "RB1", "Questionable"), (2025, 3, "RB2", "Out")])

    out = build_vacated_opportunity(pw, injuries, usage_gap=0.15, lookback=2)
    vacated, severity = _lookup(out, "RB3", 3)

    assert vacated == pytest.approx(0.90)
    assert severity == 3  # the WORST designation among blockers, not the first


def test_severity_ranks_out_above_doubtful_above_questionable():
    seen = {}
    for status in ("Questionable", "Doubtful", "Out"):
        pw = _pw([_row("STARTER", w, 0.80) for w in (1, 2, 3)]
                 + [_row("BACKUP", w, 0.20) for w in (1, 2, 3)])
        out = build_vacated_opportunity(
            pw, _injuries([(2025, 3, "STARTER", status)]), usage_gap=0.15, lookback=2
        )
        seen[status] = _lookup(out, "BACKUP", 3)[1]

    assert seen["Out"] > seen["Doubtful"] > seen["Questionable"] > 0


# ---- scoping ----

def test_a_different_position_is_not_a_blocker():
    pw = _pw([_row("WR1", w, 0.90, position="WR") for w in (1, 2, 3)]
             + [_row("RB1", w, 0.20, position="RB") for w in (1, 2, 3)])
    injuries = _injuries([(2025, 3, "WR1", "Out")])

    out = build_vacated_opportunity(pw, injuries, usage_gap=0.15, lookback=2)
    vacated, _ = _lookup(out, "RB1", 3)

    assert vacated == 0.0


def test_a_different_team_is_not_a_blocker():
    pw = _pw([_row("OTHER", w, 0.90, team="CHI") for w in (1, 2, 3)]
             + [_row("MINE", w, 0.20, team="GB") for w in (1, 2, 3)])
    injuries = _injuries([(2025, 3, "OTHER", "Out")])

    out = build_vacated_opportunity(pw, injuries, usage_gap=0.15, lookback=2)
    vacated, _ = _lookup(out, "MINE", 3)

    assert vacated == 0.0


def test_a_different_season_is_not_a_blocker():
    pw = _pw([_row("STARTER", w, 0.80, season=2024) for w in (1, 2, 3)]
             + [_row("BACKUP", w, 0.20, season=2025) for w in (1, 2, 3)])
    injuries = _injuries([(2024, 3, "STARTER", "Out")])

    out = build_vacated_opportunity(pw, injuries, usage_gap=0.15, lookback=2)
    row = out[(out["player_id"] == "BACKUP") & (out["season"] == 2025) & (out["week"] == 3)].iloc[0]

    assert row["vacated_snap_share"] == 0.0


# ---- no leakage ----

def test_an_injury_reported_after_the_feature_week_is_not_used():
    # Week 5's designation must not appear in week 3's features -- it
    # doesn't exist yet when the model predicts week 4.
    pw = _pw([_row("STARTER", w, 0.80) for w in (1, 2, 3)]
             + [_row("BACKUP", w, 0.20) for w in (1, 2, 3)])
    injuries = _injuries([(2025, 5, "STARTER", "Out")])

    out = build_vacated_opportunity(pw, injuries, usage_gap=0.15, lookback=2)
    vacated, severity = _lookup(out, "BACKUP", 3)

    assert vacated == 0.0
    assert severity == 0


def test_an_injury_older_than_the_lookback_window_expires():
    pw = _pw([_row("STARTER", w, 0.80) for w in (1, 2, 3, 4, 5, 6)]
             + [_row("BACKUP", w, 0.20) for w in (1, 2, 3, 4, 5, 6)])
    injuries = _injuries([(2025, 1, "STARTER", "Out")])

    out = build_vacated_opportunity(pw, injuries, usage_gap=0.15, lookback=2)

    assert _lookup(out, "BACKUP", 3)[0] == pytest.approx(0.80)  # weeks 1..3 window
    assert _lookup(out, "BACKUP", 6)[0] == 0.0  # long expired


def test_trailing_usage_comes_from_weeks_strictly_before_the_feature_week():
    # The starter is a bit player until week 3, when he explodes. His
    # week-3 share must not inflate week 3's own vacated total.
    pw = _pw([
        _row("STARTER", 1, 0.20), _row("STARTER", 2, 0.20), _row("STARTER", 3, 0.99),
        _row("BACKUP", 1, 0.02), _row("BACKUP", 2, 0.02), _row("BACKUP", 3, 0.02),
    ])
    injuries = _injuries([(2025, 3, "STARTER", "Out")])

    out = build_vacated_opportunity(pw, injuries, usage_gap=0.15, lookback=2)
    vacated, _ = _lookup(out, "BACKUP", 3)

    assert vacated == pytest.approx(0.20)  # mean of weeks 1-2, not 0.99


# ---- agreement with the shipped per-row implementation ----

def test_blockers_match_injury_contexts_own_teammates_ahead():
    # The batch version exists only for speed; if it disagrees with the
    # shipped per-row logic it is measuring something else entirely.
    pw = _pw([
        _row("RB1", 1, 0.70), _row("RB2", 1, 0.45), _row("RB3", 1, 0.10),
        _row("RB1", 2, 0.75), _row("RB2", 2, 0.40), _row("RB3", 2, 0.12),
        _row("RB1", 3, 0.70), _row("RB2", 3, 0.40), _row("RB3", 3, 0.15),
    ])
    injuries = _injuries([(2025, 3, "RB1", "Out"), (2025, 3, "RB2", "Out")])
    out = build_vacated_opportunity(pw, injuries, usage_gap=0.15, lookback=2)

    for candidate in ("RB1", "RB2", "RB3"):
        ahead = _teammates_ahead(pw, candidate, season=2025, week=3, usage_gap=0.15, lookback=2)
        # Every blocker is injured in this fixture, so the reference sum
        # over "teammates ahead" is exactly what the batch should produce.
        expected = float(ahead.sum()) if len(ahead) else 0.0
        assert _lookup(out, candidate, 3)[0] == pytest.approx(expected), candidate


# ---- guards and attachment ----

def test_missing_columns_are_named():
    with pytest.raises(ValueError, match="missing"):
        build_vacated_opportunity(pd.DataFrame({"season": [2025]}), _injuries([]))


def test_nonsense_parameters_raise():
    pw = _pw([_row("P1", 1, 0.5)])
    with pytest.raises(ValueError, match="usage_gap"):
        build_vacated_opportunity(pw, _injuries([]), usage_gap=-0.1)
    with pytest.raises(ValueError, match="lookback"):
        build_vacated_opportunity(pw, _injuries([]), lookback=0)


def test_no_injury_data_at_all_yields_zeros_not_a_crash():
    pw = _pw([_row("P1", w, 0.5) for w in (1, 2, 3)])

    out = build_vacated_opportunity(pw, _injuries([]))

    assert (out["vacated_snap_share"] == 0.0).all()
    assert (out["blocker_severity"] == 0).all()


def test_attach_fills_absent_rows_with_zero_not_null():
    # "No injured blocker" is an observation, not missing data -- leaving
    # nulls would drop those rows from training for no reason.
    features = pd.DataFrame([
        {"season": 2025, "week": 1, "player_id": "P1"},
        {"season": 2025, "week": 1, "player_id": "P2"},
    ])
    vacated = pd.DataFrame([
        {"season": 2025, "week": 1, "player_id": "P1",
         "vacated_snap_share": 0.6, "blocker_severity": 3},
    ])

    out = attach_vacated_opportunity(features, vacated).set_index("player_id")

    assert out.loc["P1", "vacated_snap_share"] == pytest.approx(0.6)
    assert out.loc["P2", "vacated_snap_share"] == 0.0
    assert out.loc["P2", "blocker_severity"] == 0


def test_attach_refuses_to_fan_out_the_feature_table():
    features = pd.DataFrame([{"season": 2025, "week": 1, "player_id": "P1"}])
    vacated = pd.DataFrame([
        {"season": 2025, "week": 1, "player_id": "P1", "vacated_snap_share": 0.6, "blocker_severity": 3},
        {"season": 2025, "week": 1, "player_id": "P1", "vacated_snap_share": 0.7, "blocker_severity": 2},
    ])

    with pytest.raises(ValueError, match="duplicate"):
        attach_vacated_opportunity(features, vacated)
