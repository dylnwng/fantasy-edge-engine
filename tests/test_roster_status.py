import pandas as pd

from edge_engine.roster.roster_status import _filter_flagged, roster_status_note


def _player(gsis_id, status, last_season):
    return {"gsis_id": gsis_id, "status": status, "last_season": last_season}


def test_flagged_status_is_captured_with_human_label():
    players = pd.DataFrame(
        [
            _player("P1", "SUS", 2024),
            _player("P2", "RES", 2024),
        ]
    )
    result = _filter_flagged(players)

    assert result["P1"] == ("SUS", "Suspended")
    assert result["P2"] == ("RES", "Reserve list")


def test_active_and_unrecognized_statuses_are_excluded():
    players = pd.DataFrame(
        [
            _player("P1", "ACT", 2024),
            _player("P2", "RET", 2024),
            _player("P3", "CUT", 2024),
        ]
    )
    result = _filter_flagged(players)

    assert result == {}


def test_stale_season_records_are_excluded():
    # P1's flagged status is from an old season -- nflverse's lifetime
    # reference table doesn't re-scrape status once a player is no
    # longer relevant, so a SUS code from years ago is stale history,
    # not a current status. Only the most-recent last_season counts.
    players = pd.DataFrame(
        [
            _player("P1", "SUS", 2017),
            _player("P2", "SUS", 2024),
        ]
    )
    result = _filter_flagged(players)

    assert "P1" not in result
    assert result["P2"] == ("SUS", "Suspended")


def test_missing_gsis_id_is_excluded():
    players = pd.DataFrame(
        [
            {"gsis_id": None, "status": "SUS", "last_season": 2024},
            _player("P1", "SUS", 2024),
        ]
    )
    result = _filter_flagged(players)

    assert list(result.keys()) == ["P1"]


def test_roster_status_note_is_none_for_unflagged_or_unresolved_player():
    statuses = {"P1": ("SUS", "Suspended")}

    assert roster_status_note("P2", statuses) is None
    assert roster_status_note(None, statuses) is None


def test_roster_status_note_formats_flagged_player():
    statuses = {"P1": ("SUS", "Suspended")}

    note = roster_status_note("P1", statuses)

    assert note == "Officially listed as Suspended (SUS) — may not be eligible to play."
