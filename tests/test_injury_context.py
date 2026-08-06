import pandas as pd

from edge_engine.model.injury_context import get_injury_context


def _pw_row(player_id, week, team, position, snap_pct):
    return {
        "season": 2023,
        "week": week,
        "player_id": player_id,
        "team": team,
        "position": position,
        "snap_pct": snap_pct,
    }


def _injury_row(player_id, week, team, position, name, status, injury="Hamstring"):
    return {
        "season": 2023,
        "week": week,
        "team": team,
        "gsis_id": player_id,
        "full_name": name,
        "position": position,
        "report_status": status,
        "report_primary_injury": injury,
    }


def test_flags_when_ahead_teammate_is_injured():
    # Starter had 0.8 snap share weeks 1-2, backup only 0.1 -> a real gap.
    # In week 3, the starter is out and the backup's usage would spike.
    player_week = pd.DataFrame(
        [
            _pw_row("STARTER", 1, "GB", "RB", 0.80),
            _pw_row("STARTER", 2, "GB", "RB", 0.80),
            _pw_row("BACKUP", 1, "GB", "RB", 0.10),
            _pw_row("BACKUP", 2, "GB", "RB", 0.10),
            _pw_row("BACKUP", 3, "GB", "RB", 0.85),
        ]
    )
    injuries = pd.DataFrame([_injury_row("STARTER", 3, "GB", "RB", "Josh Jacobs", "Out")])

    ctx = get_injury_context(player_week, injuries, "BACKUP", 2023, 3, usage_gap=0.15, lookback=2)

    assert ctx.has_injury_context is True
    assert ctx.teammate_name == "Josh Jacobs"
    assert ctx.teammate_status == "Out"
    assert "Josh Jacobs" in ctx.explanation
    assert "Out" in ctx.explanation


def test_no_context_when_ahead_teammate_is_healthy():
    player_week = pd.DataFrame(
        [
            _pw_row("STARTER", 1, "GB", "RB", 0.80),
            _pw_row("STARTER", 2, "GB", "RB", 0.80),
            _pw_row("BACKUP", 1, "GB", "RB", 0.10),
            _pw_row("BACKUP", 2, "GB", "RB", 0.10),
            _pw_row("BACKUP", 3, "GB", "RB", 0.15),
        ]
    )
    injuries = pd.DataFrame(columns=[
        "season", "week", "team", "gsis_id", "full_name", "position",
        "report_status", "report_primary_injury",
    ])

    ctx = get_injury_context(player_week, injuries, "BACKUP", 2023, 3, usage_gap=0.15, lookback=2)

    assert ctx.has_injury_context is False
    assert ctx.explanation == ""


def test_no_context_when_candidate_already_had_the_usage_lead():
    # Candidate was already the lead back before this week -- there's no
    # teammate "ahead of him," so an injury to a lesser-used teammate
    # shouldn't be surfaced as context for his own role.
    player_week = pd.DataFrame(
        [
            _pw_row("LEAD", 1, "GB", "RB", 0.80),
            _pw_row("LEAD", 2, "GB", "RB", 0.80),
            _pw_row("LEAD", 3, "GB", "RB", 0.85),
            _pw_row("BENCH", 1, "GB", "RB", 0.10),
            _pw_row("BENCH", 2, "GB", "RB", 0.10),
        ]
    )
    injuries = pd.DataFrame([_injury_row("BENCH", 3, "GB", "RB", "Third Stringer", "Out")])

    ctx = get_injury_context(player_week, injuries, "LEAD", 2023, 3, usage_gap=0.15, lookback=2)

    assert ctx.has_injury_context is False


def test_prefers_most_severe_status_among_multiple_ahead_teammates():
    player_week = pd.DataFrame(
        [
            _pw_row("WR1", 1, "MIA", "WR", 0.90),
            _pw_row("WR1", 2, "MIA", "WR", 0.90),
            _pw_row("WR2", 1, "MIA", "WR", 0.70),
            _pw_row("WR2", 2, "MIA", "WR", 0.70),
            _pw_row("WR3", 1, "MIA", "WR", 0.10),
            _pw_row("WR3", 2, "MIA", "WR", 0.10),
            _pw_row("WR3", 3, "MIA", "WR", 0.60),
        ]
    )
    injuries = pd.DataFrame(
        [
            _injury_row("WR1", 3, "MIA", "WR", "Tyreek Hill", "Questionable", "Wrist"),
            _injury_row("WR2", 3, "MIA", "WR", "Jaylen Waddle", "Out", "Ankle"),
        ]
    )

    ctx = get_injury_context(player_week, injuries, "WR3", 2023, 3, usage_gap=0.15, lookback=2)

    assert ctx.has_injury_context is True
    assert ctx.teammate_name == "Jaylen Waddle"  # Out outranks Questionable
    assert ctx.teammate_status == "Out"


def _blocker_reports(statuses_by_week):
    return pd.DataFrame(
        [_injury_row("STARTER", wk, "GB", "RB", "Star Starter", s) for wk, s in statuses_by_week.items()]
    )


def test_blocker_trajectory_improving_means_the_window_is_closing():
    from edge_engine.model.injury_context import blocker_status_trajectory

    injuries = _blocker_reports({2: "Out", 3: "Questionable"})
    assert blocker_status_trajectory(injuries, "STARTER", 2023, 3, lookback=2) == "improving"


def test_blocker_trajectory_worsening_means_the_window_is_staying_open():
    from edge_engine.model.injury_context import blocker_status_trajectory

    injuries = _blocker_reports({2: "Questionable", 3: "Out"})
    assert blocker_status_trajectory(injuries, "STARTER", 2023, 3, lookback=2) == "worsening"


def test_blocker_trajectory_stable_when_designation_unchanged():
    from edge_engine.model.injury_context import blocker_status_trajectory

    injuries = _blocker_reports({2: "Out", 3: "Out"})
    assert blocker_status_trajectory(injuries, "STARTER", 2023, 3, lookback=2) == "stable"


def test_blocker_trajectory_needs_two_reports_to_have_a_direction():
    from edge_engine.model.injury_context import blocker_status_trajectory

    injuries = _blocker_reports({3: "Out"})
    assert blocker_status_trajectory(injuries, "STARTER", 2023, 3, lookback=2) is None


def test_blocker_trajectory_does_not_infer_recovery_from_absence():
    # load_injury_reports() only keeps rows carrying a game-status
    # designation, so a week where the blocker simply doesn't appear
    # might be a practice-only report rather than a clean bill of
    # health. Absence must never be scored as "improving".
    from edge_engine.model.injury_context import blocker_status_trajectory

    injuries = _blocker_reports({2: "Out"})  # nothing at all in week 3
    assert blocker_status_trajectory(injuries, "STARTER", 2023, 3, lookback=2) is None


def test_trajectory_is_surfaced_in_the_explanation_and_field():
    player_week = pd.DataFrame(
        [
            _pw_row("STARTER", 1, "GB", "RB", 0.80),
            _pw_row("STARTER", 2, "GB", "RB", 0.80),
            _pw_row("BACKUP", 1, "GB", "RB", 0.10),
            _pw_row("BACKUP", 2, "GB", "RB", 0.10),
            _pw_row("BACKUP", 3, "GB", "RB", 0.85),
        ]
    )
    injuries = _blocker_reports({2: "Out", 3: "Questionable"})

    context = get_injury_context(player_week, injuries, "BACKUP", 2023, 3)

    assert context.has_injury_context
    assert context.blocker_trajectory == "improving"
    assert "window may be closing" in context.explanation.replace("opportunity ", "")
