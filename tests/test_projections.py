from edge_engine.roster.matchup import MatchupPlayer
from edge_engine.roster.models import Player
from edge_engine.simulation.config import SimulationConfig
from edge_engine.simulation.projections import resolve_projection

CONFIG = SimulationConfig(
    n_sims=1000, variance_window=6, min_games_for_variance=3,
    win_prob_improvement_threshold=0.01, untracked_position_std=5.0,
)


def _matchup_player(name, position, player_id="pid", espn_proj=10.0, actual=0.0, game_final=False, on_bye=False):
    player = Player(name=name, position=position, team="X", player_id=player_id)
    return MatchupPlayer(player=player, espn_projected_points=espn_proj, actual_points=actual, game_final=game_final, on_bye=on_bye)


def test_bye_tier_is_deterministic_zero():
    mp = _matchup_player("A", "RB", on_bye=True)
    proj = resolve_projection(mp, {}, {}, CONFIG)
    assert proj.source == "bye"
    assert proj.mean == 0.0
    assert proj.std == 0.0
    assert proj.note


def test_final_tier_uses_actual_points_deterministically():
    mp = _matchup_player("A", "RB", actual=27.4, game_final=True)
    # even if a model score also exists, "final" must win -- the game already happened
    scored = {"pid": (15.0, 4.0)}
    proj = resolve_projection(mp, scored, {}, CONFIG)
    assert proj.source == "final"
    assert proj.mean == 27.4
    assert proj.std == 0.0
    assert proj.note


def test_bye_flag_beats_default_game_played_true():
    # espn_api's BoxPlayer.game_played defaults to 100 and is never
    # recomputed for a bye-week player -- on_bye must win regardless.
    mp = _matchup_player("A", "RB", actual=0.0, game_final=True, on_bye=True)
    proj = resolve_projection(mp, {}, {}, CONFIG)
    assert proj.source == "bye"


def test_model_tier_uses_own_predicted_score_and_std():
    mp = _matchup_player("A", "RB")
    scored = {"pid": (18.5, 6.2)}
    proj = resolve_projection(mp, scored, {"RB": 99.0}, CONFIG)
    assert proj.source == "model"
    assert proj.mean == 18.5
    assert proj.std == 6.2
    assert proj.note == ""


def test_model_position_std_fallback_when_own_std_unavailable():
    mp = _matchup_player("A", "RB")
    scored = {"pid": (18.5, None)}  # model score exists, but <3 trailing games for a std
    proj = resolve_projection(mp, scored, {"RB": 7.5}, CONFIG)
    assert proj.source == "model_position_std_fallback"
    assert proj.mean == 18.5
    assert proj.std == 7.5
    assert proj.note


def test_espn_projection_only_for_unresolved_player():
    mp = _matchup_player("A", "WR", player_id=None, espn_proj=9.5)
    proj = resolve_projection(mp, {}, {"WR": 6.0}, CONFIG)
    assert proj.source == "espn_projection_only"
    assert proj.mean == 9.5
    assert proj.std == 6.0
    assert proj.note


def test_kicker_gets_zero_std_not_fabricated():
    mp = _matchup_player("A", "K", espn_proj=8.0)
    proj = resolve_projection(mp, {}, {"K": 3.5}, CONFIG)
    assert proj.source == "espn_projection_only"
    assert proj.std == 0.0  # not the position_avg_std -- no fabricated spread for K
    assert proj.note


def test_dst_gets_zero_std_not_fabricated():
    mp = _matchup_player("A", "D/ST", espn_proj=6.0)
    proj = resolve_projection(mp, {}, {}, CONFIG)
    assert proj.source == "espn_projection_only"
    assert proj.std == 0.0
