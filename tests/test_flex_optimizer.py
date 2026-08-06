import numpy as np
import pytest

from edge_engine.roster.matchup import MatchupPlayer
from edge_engine.roster.models import Player
from edge_engine.simulation.flex_optimizer import enumerate_flex_candidates, find_best_flex_lineup
from edge_engine.simulation.projections import PlayerProjection


def _mp(name, position, slot):
    player = Player(name=name, position=position, team="X", player_id=name, slot_position=slot)
    return MatchupPlayer(player=player, espn_projected_points=0.0, actual_points=0.0, game_final=False, on_bye=False)


def _proj(mp_obj, mean, std):
    p = mp_obj.player
    return PlayerProjection(p.player_id, p.name, p.position, p.team, mean, std, "model", "")


LINEUP_SLOTS_ONE_FLEX = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "K": 1, "DST": 1, "BENCH": 2}


def _basic_lineup():
    return [
        _mp("QB1", "QB", "QB"),
        _mp("RB1", "RB", "RB"),
        _mp("RB2", "RB", "RB"),
        _mp("WR1", "WR", "WR"),
        _mp("WR2", "WR", "WR"),
        _mp("TE1", "TE", "TE"),
        _mp("FlexNow", "RB", "FLEX"),
        _mp("BenchStud", "WR", "BE"),
        _mp("BenchWeak", "TE", "BE"),
        _mp("K1", "K", "K"),
        _mp("DST1", "D/ST", "DST"),
    ]


def test_enumerate_flex_candidates_only_considers_flex_eligible_bench():
    lineup = _basic_lineup()
    candidates = enumerate_flex_candidates(lineup, LINEUP_SLOTS_ONE_FLEX)

    # flex_pool = FlexNow (currently FLEX) + BenchStud + BenchWeak = 3 players,
    # choose 1 -> C(3,1) = 3 candidates. K1/DST1 must never appear as flex picks.
    assert len(candidates) == 3
    flex_names = {tuple(sorted(mp.player.name for mp in c if mp.player.slot_position in ("FLEX", "BE"))) for c in candidates}
    assert flex_names == {("FlexNow",), ("BenchStud",), ("BenchWeak",)}


def test_enumerate_flex_candidates_zero_flex_slots_returns_current_only():
    lineup = _basic_lineup()
    candidates = enumerate_flex_candidates(lineup, {"QB": 1, "RB": 2, "WR": 2, "TE": 1})
    assert len(candidates) == 1


def test_enumerate_flex_candidates_two_flex_slots_gives_correct_combination_count():
    lineup = _basic_lineup() + [_mp("BenchExtra", "RB", "BE")]
    # flex_pool now has 4 players (FlexNow, BenchStud, BenchWeak, BenchExtra), choose 2 -> C(4,2)=6
    candidates = enumerate_flex_candidates(lineup, {**LINEUP_SLOTS_ONE_FLEX, "FLEX": 2})
    assert len(candidates) == 6


def test_enumerate_flex_candidates_raises_clear_error_when_pool_too_small():
    # A 2-FLEX league with a thin bench (only 1 FLEX-eligible player) used
    # to leave find_best_flex_lineup() with an empty candidate list, which
    # crashed on scored[0] with a bare IndexError. Must fail clearly here
    # instead, before that ever happens.
    lineup = [
        _mp("QB1", "QB", "QB"),
        _mp("RB1", "RB", "RB"),
        _mp("WR1", "WR", "WR"),
        _mp("BenchRB", "RB", "BE"),  # only 1 FLEX-eligible bench player
        _mp("BenchK", "K", "BE"),  # not FLEX-eligible
    ]
    with pytest.raises(ValueError, match="2 FLEX slot"):
        enumerate_flex_candidates(lineup, {"QB": 1, "RB": 1, "WR": 1, "FLEX": 2})


def test_find_best_flex_lineup_raises_clear_error_when_pool_too_small():
    lineup = [
        _mp("QB1", "QB", "QB"),
        _mp("RB1", "RB", "RB"),
        _mp("WR1", "WR", "WR"),
        _mp("BenchRB", "RB", "BE"),
        _mp("BenchK", "K", "BE"),
    ]
    opp_proj = [_proj(_mp("Opp1", "RB", "RB"), 12.0, 3.0)]

    with pytest.raises(ValueError, match="2 FLEX slot"):
        find_best_flex_lineup(
            lineup, opp_proj, lambda mp: _proj(mp, 10.0, 3.0),
            {"QB": 1, "RB": 1, "WR": 1, "FLEX": 2},
            n_sims=100, rng=np.random.default_rng(0),
        )


def test_find_best_flex_lineup_picks_the_analytically_obvious_swap():
    lineup = _basic_lineup()
    projections = {
        "QB1": (18.0, 4.0), "RB1": (14.0, 4.0), "RB2": (13.0, 4.0),
        "WR1": (12.0, 4.0), "WR2": (11.0, 4.0), "TE1": (9.0, 3.0),
        "FlexNow": (5.0, 3.0),      # weak current flex starter
        "BenchStud": (25.0, 5.0),   # obviously much stronger bench option
        "BenchWeak": (3.0, 2.0),
        "K1": (7.0, 0.0), "DST1": (6.0, 0.0),
    }

    def project_fn(mp):
        mean, std = projections[mp.player.name]
        return _proj(mp, mean, std)

    opponent_projections = [PlayerProjection(f"o{i}", f"Opp{i}", "RB", "OPP", 10.0, 4.0, "model", "") for i in range(7)]

    rec = find_best_flex_lineup(
        lineup, opponent_projections, project_fn, LINEUP_SLOTS_ONE_FLEX,
        n_sims=20_000, rng=np.random.default_rng(1),
    )

    assert [p.name for p in rec.swapped_in] == ["BenchStud"]
    assert [p.name for p in rec.swapped_out] == ["FlexNow"]
    assert rec.improved is True
    assert rec.best_lineup_win_probability > rec.current_lineup_win_probability


def test_find_best_flex_lineup_no_improvement_when_current_is_already_best():
    lineup = _basic_lineup()
    projections = {
        "QB1": (18.0, 4.0), "RB1": (14.0, 4.0), "RB2": (13.0, 4.0),
        "WR1": (12.0, 4.0), "WR2": (11.0, 4.0), "TE1": (9.0, 3.0),
        "FlexNow": (25.0, 5.0),    # current flex starter is already the best option
        "BenchStud": (5.0, 3.0),
        "BenchWeak": (3.0, 2.0),
        "K1": (7.0, 0.0), "DST1": (6.0, 0.0),
    }

    def project_fn(mp):
        mean, std = projections[mp.player.name]
        return _proj(mp, mean, std)

    opponent_projections = [PlayerProjection(f"o{i}", f"Opp{i}", "RB", "OPP", 10.0, 4.0, "model", "") for i in range(7)]

    rec = find_best_flex_lineup(
        lineup, opponent_projections, project_fn, LINEUP_SLOTS_ONE_FLEX,
        n_sims=20_000, rng=np.random.default_rng(1),
    )

    assert rec.swapped_in == []
    assert rec.swapped_out == []
    assert rec.improved is False
