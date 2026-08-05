from types import SimpleNamespace

from edge_engine.roster.espn_source import EspnRosterStateSource


def _team(team_id):
    return SimpleNamespace(team_id=team_id, team_name=f"Team{team_id}")


def _box_score(home_id, away_id, home_lineup=None, away_lineup=None):
    return SimpleNamespace(
        home_team=_team(home_id) if home_id is not None else None,
        away_team=_team(away_id) if away_id is not None else None,
        home_lineup=home_lineup or [],
        away_lineup=away_lineup or [],
    )


def test_select_my_side_finds_matchup_when_home():
    box_scores = [_box_score(1, 2), _box_score(3, 4)]
    result = EspnRosterStateSource._select_my_side(box_scores, my_team_id=3)
    assert result is not None
    my_team, my_lineup, opp_team, opp_lineup = result
    assert my_team.team_id == 3
    assert opp_team.team_id == 4


def test_select_my_side_finds_matchup_when_away():
    box_scores = [_box_score(1, 2), _box_score(3, 4)]
    result = EspnRosterStateSource._select_my_side(box_scores, my_team_id=4)
    assert result is not None
    my_team, my_lineup, opp_team, opp_lineup = result
    assert my_team.team_id == 4
    assert opp_team.team_id == 3


def test_select_my_side_returns_none_for_bye_week():
    # espn_api sets home_team/away_team to None for a bye matchup in an
    # odd-team league -- must not crash, must surface as "no opponent".
    box_scores = [_box_score(1, 2), _box_score(5, None)]
    result = EspnRosterStateSource._select_my_side(box_scores, my_team_id=5)
    assert result is None


def test_select_my_side_returns_none_when_team_not_found():
    box_scores = [_box_score(1, 2)]
    result = EspnRosterStateSource._select_my_side(box_scores, my_team_id=99)
    assert result is None


def _box_player(on_bye_week=False, game_played=100):
    return SimpleNamespace(on_bye_week=on_bye_week, game_played=game_played)


def test_resolve_game_final_true_when_game_played_and_not_bye():
    on_bye, game_final = EspnRosterStateSource._resolve_game_final(_box_player(on_bye_week=False, game_played=100))
    assert on_bye is False
    assert game_final is True


def test_resolve_game_final_false_when_game_not_yet_played():
    on_bye, game_final = EspnRosterStateSource._resolve_game_final(_box_player(on_bye_week=False, game_played=0))
    assert on_bye is False
    assert game_final is False


def test_resolve_game_final_bye_week_beats_default_game_played_true():
    # The actual gotcha: game_played defaults to 100 and is never
    # recomputed for a bye-week player. on_bye_week must win, or a bye
    # player would be misread as "game final."
    on_bye, game_final = EspnRosterStateSource._resolve_game_final(_box_player(on_bye_week=True, game_played=100))
    assert on_bye is True
    assert game_final is False
