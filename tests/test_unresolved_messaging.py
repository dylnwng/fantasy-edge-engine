"""Advice about unresolved players has to match where the names came
from. A live week-7 ESPN run told the user to go check the spelling in
an export that does not exist -- the names had come from ESPN's API."""

from edge_engine.roster.models import Player
from edge_engine.roster.unresolved import unresolved_headline


def _p(name, team="CLE"):
    return Player(
        name=name, position="RB", team=team, player_id=None,
        match_status="unmatched", match_note="no player found",
    )


def test_espn_source_never_tells_the_user_to_check_their_spelling():
    head = unresolved_headline([_p("Audric Estime")], source="espn")
    assert "spelling" not in head.lower()
    assert "CSV" not in head
    assert "ESPN" in head


def test_manual_source_does_tell_the_user_to_check_their_spelling():
    head = unresolved_headline([_p("Typo Playerr")], source="manual")
    assert "spelling" in head.lower()
    assert "CSV" in head


def test_missing_team_is_called_out_as_a_source_gap_not_user_error():
    head = unresolved_headline([_p("Khari Blasingame", team=None)], source="espn")
    assert "no team" in head
    assert "source data" in head


def test_missing_team_count_only_counts_the_ones_actually_missing_it():
    players = [_p("Has Team", "CLE"), _p("No Team", None), _p("Also None", "None")]
    assert "(2 of these" in unresolved_headline(players, source="espn")
    assert "of these arrived with no team" not in unresolved_headline([_p("Has Team", "CLE")], source="espn")


def test_count_and_kind_are_reported():
    head = unresolved_headline([_p("A"), _p("B")], kind="player", source="manual")
    assert head.startswith("2 player(s)")
