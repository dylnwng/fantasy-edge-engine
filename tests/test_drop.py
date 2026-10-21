import numpy as np
import pandas as pd
import pytest

from edge_engine.ranking.drop import (
    NO_DATA_REASON,
    DropCandidate,
    Swap,
    droppable,
    rank_drop_candidates,
    suggest_swaps,
)
from edge_engine.roster.models import LeagueConfig, Player, ScoringSettings, WaiverConfig

LINEUP = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "BENCH": 6}


def _league(lineup_slots=None):
    return LeagueConfig(
        season=2026,
        scoring=ScoringSettings(ppr_type="full_ppr"),
        lineup_slots=lineup_slots or LINEUP,
        waivers=WaiverConfig(system="FAAB", season_budget=100, min_bid=1, clear_day="Wednesday"),
    )


def _player(name, position, player_id=None):
    return Player(name=name, position=position, team="GB",
                  player_id=player_id if player_id is not None else name.lower(),
                  match_status="matched")


def _scored(rows):
    base = {
        "has_injury_context": False, "injury_explanation": "", "roster_status_note": None,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def _score_row(player_id, predicted, position="RB", **overrides):
    return {"player_id": player_id, "predicted_score": predicted, "position": position, **overrides}


# ---- ordering ----

def test_least_valuable_player_is_the_top_drop():
    rostered = [_player(f"RB{i}", "RB") for i in range(1, 5)]
    scored = _scored([
        _score_row("rb1", 20.0), _score_row("rb2", 5.0),
        _score_row("rb3", 15.0), _score_row("rb4", 12.0),
    ])

    ranked = rank_drop_candidates(rostered, scored, _league())

    assert ranked[0].player.name == "RB2"
    assert ranked[0].roster_value == pytest.approx(5.0 * ranked[0].scarcity_multiplier)


def test_depth_at_a_position_makes_its_players_easier_to_drop():
    # Equal projections, but five RBs against one TE. The surplus RB
    # should outrank the lone-ish TE as a drop.
    rostered = [_player(f"RB{i}", "RB") for i in range(1, 6)] + [
        _player("TE1", "TE"), _player("TE2", "TE")
    ]
    scored = _scored(
        [_score_row(f"rb{i}", 10.0) for i in range(1, 6)]
        + [_score_row("te1", 10.0, position="TE"), _score_row("te2", 10.0, position="TE")]
    )

    ranked = rank_drop_candidates(rostered, scored, _league())
    top = droppable(ranked)[0]

    assert top.player.position == "RB"


# ---- protection: structural ----

def test_the_only_quarterback_is_never_recommended():
    # Dropping him is wrong at any projection.
    rostered = [_player("QB1", "QB"), _player("RB1", "RB"), _player("RB2", "RB"), _player("RB3", "RB")]
    scored = _scored([
        _score_row("qb1", 0.1, position="QB"),  # by far the worst score
        _score_row("rb1", 30.0), _score_row("rb2", 30.0), _score_row("rb3", 30.0),
    ])

    ranked = rank_drop_candidates(rostered, scored, _league())
    qb = next(c for c in ranked if c.player.position == "QB")

    assert qb.is_protected
    assert "starting slot" in qb.protected_reason
    assert qb not in droppable(ranked)
    # Worst score in the league, yet never the recommendation.
    assert droppable(ranked)[0].player.position == "RB"


def test_a_position_at_exactly_its_starting_requirement_is_protected():
    # 2 RBs for 2 RB slots: dropping one leaves you unable to field a lineup.
    rostered = [_player("RB1", "RB"), _player("RB2", "RB")]
    scored = _scored([_score_row("rb1", 1.0), _score_row("rb2", 2.0)])

    ranked = rank_drop_candidates(rostered, scored, _league())

    assert all(c.is_protected for c in ranked)
    assert droppable(ranked) == []


def test_surplus_above_the_requirement_is_droppable():
    rostered = [_player(f"RB{i}", "RB") for i in range(1, 4)]  # 3 RBs, 2 slots
    scored = _scored([_score_row(f"rb{i}", float(i)) for i in range(1, 4)])

    ranked = rank_drop_candidates(rostered, scored, _league())

    assert len(droppable(ranked)) == 3
    assert droppable(ranked)[0].player.name == "RB1"  # lowest projection


def test_flex_does_not_count_toward_the_structural_requirement():
    # A fractional FLEX share is right for valuation but cannot express
    # "you must be able to field a starter here", so the hard rule uses
    # dedicated slots only. 2 TEs with 1 dedicated TE slot -> droppable.
    rostered = [_player("TE1", "TE"), _player("TE2", "TE")]
    scored = _scored([_score_row("te1", 5.0, position="TE"), _score_row("te2", 9.0, position="TE")])

    ranked = rank_drop_candidates(rostered, scored, _league())

    assert droppable(ranked)[0].player.name == "TE1"


# ---- protection: missing data ----

def test_a_player_with_no_model_row_is_never_recommended():
    # Recommending a drop on the basis of data you don't have would cost
    # a real player -- the same refusal bye_weeks.py makes.
    rostered = [_player(f"RB{i}", "RB") for i in range(1, 4)] + [_player("Rookie", "RB")]
    scored = _scored([_score_row(f"rb{i}", 20.0) for i in range(1, 4)])  # no row for Rookie

    ranked = rank_drop_candidates(rostered, scored, _league())
    rookie = next(c for c in ranked if c.player.name == "Rookie")

    assert rookie.is_protected
    assert rookie.protected_reason == NO_DATA_REASON
    assert rookie.predicted_score is None
    assert rookie.roster_value is None


def test_an_unresolved_player_with_no_id_is_never_recommended():
    rostered = [_player(f"RB{i}", "RB") for i in range(1, 4)] + [
        Player(name="Typo Playerr", position="RB", team="GB", player_id=None, match_status="unmatched")
    ]
    scored = _scored([_score_row(f"rb{i}", 20.0) for i in range(1, 4)])

    ranked = rank_drop_candidates(rostered, scored, _league())
    unresolved = next(c for c in ranked if c.player.name == "Typo Playerr")

    assert unresolved.is_protected
    assert unresolved.roster_value is None


def test_missing_score_is_none_not_zero():
    # 0.0 would sort to the top as "worthless" and get recommended.
    rostered = [_player(f"RB{i}", "RB") for i in range(1, 4)] + [_player("Unknown", "RB")]
    scored = _scored([_score_row(f"rb{i}", 20.0) for i in range(1, 4)])

    ranked = rank_drop_candidates(rostered, scored, _league())
    unknown = next(c for c in ranked if c.player.name == "Unknown")

    assert unknown.predicted_score is None
    assert unknown.roster_value is None


def test_protected_players_always_sort_last():
    rostered = [_player("QB1", "QB")] + [_player(f"RB{i}", "RB") for i in range(1, 4)]
    scored = _scored([
        _score_row("qb1", 0.0, position="QB"),
        _score_row("rb1", 1.0), _score_row("rb2", 2.0), _score_row("rb3", 3.0),
    ])

    ranked = rank_drop_candidates(rostered, scored, _league())

    protected_flags = [c.is_protected for c in ranked]
    assert protected_flags == sorted(protected_flags)  # False... then True...


def test_empty_scored_frame_protects_everyone_rather_than_crashing():
    # Early season: nobody has a full trailing window yet.
    rostered = [_player(f"RB{i}", "RB") for i in range(1, 5)]

    ranked = rank_drop_candidates(rostered, pd.DataFrame(), _league())

    assert len(ranked) == 4
    assert droppable(ranked) == []


# ---- context is surfaced, never scored ----

def test_roster_status_is_surfaced_without_changing_the_ranking():
    rostered = [_player(f"RB{i}", "RB") for i in range(1, 4)]
    scored = _scored([
        _score_row("rb1", 20.0, roster_status_note="Officially listed as Suspended (SUS)."),
        _score_row("rb2", 5.0),
        _score_row("rb3", 15.0),
    ])

    ranked = rank_drop_candidates(rostered, scored, _league())
    suspended = next(c for c in ranked if c.player.name == "RB1")

    assert "Suspended" in suspended.explanation
    # The suspension did NOT move him up the drop list -- RB2 still scores
    # lowest and is still the recommendation. The user decides.
    assert droppable(ranked)[0].player.name == "RB2"


def test_nan_roster_status_does_not_leak_the_string_nan():
    rostered = [_player(f"RB{i}", "RB") for i in range(1, 4)]
    scored = _scored([_score_row(f"rb{i}", float(i), roster_status_note=np.nan) for i in range(1, 4)])

    ranked = rank_drop_candidates(rostered, scored, _league())

    assert all("nan" not in c.explanation.lower() for c in ranked)


def test_injury_context_is_attached_when_present():
    rostered = [_player(f"RB{i}", "RB") for i in range(1, 4)]
    scored = _scored([
        _score_row("rb1", 5.0, has_injury_context=True, injury_explanation="Starter is Out."),
        _score_row("rb2", 20.0), _score_row("rb3", 20.0),
    ])

    ranked = rank_drop_candidates(rostered, scored, _league())

    assert "Starter is Out." in ranked[0].explanation


# ---- swaps ----

class _FakeAdd:
    def __init__(self, name, score):
        self.candidate = type("C", (), {"name": name})()
        self.roster_fit_score = score


def _drop(name, value):
    return DropCandidate(
        player=_player(name, "RB"), predicted_score=value, scarcity_multiplier=1.0,
        roster_value=value, rostered_count_at_position=4, position_need=2.3,
        protected_reason=None, explanation="",
    )


def test_a_swap_is_flagged_as_an_upgrade_only_when_the_add_scores_higher():
    swaps = suggest_swaps([_FakeAdd("Rising", 18.0)], [_drop("Fading", 6.0)])

    assert len(swaps) == 1
    assert swaps[0].gain == pytest.approx(12.0)
    assert swaps[0].is_upgrade


def test_a_swap_that_loses_value_is_reported_as_not_an_upgrade():
    swaps = suggest_swaps([_FakeAdd("Speculative", 4.0)], [_drop("Solid", 11.0)])

    assert not swaps[0].is_upgrade
    assert swaps[0].gain == pytest.approx(-7.0)


def test_swaps_never_pair_against_a_protected_player():
    protected = DropCandidate(
        player=_player("OnlyQB", "QB"), predicted_score=0.5, scarcity_multiplier=1.0,
        roster_value=0.5, rostered_count_at_position=1, position_need=1.0,
        protected_reason="dropping would leave 0 QB(s) for 1 starting slot(s)", explanation="",
    )

    swaps = suggest_swaps([_FakeAdd("Rising", 18.0)], [protected, _drop("Fading", 6.0)])

    assert [s.drop_name for s in swaps] == ["Fading"]


def test_no_swaps_when_there_is_nothing_safe_to_drop():
    protected = DropCandidate(
        player=_player("OnlyQB", "QB"), predicted_score=0.5, scarcity_multiplier=1.0,
        roster_value=0.5, rostered_count_at_position=1, position_need=1.0,
        protected_reason="protected", explanation="",
    )

    assert suggest_swaps([_FakeAdd("Rising", 18.0)], [protected]) == []


def test_swaps_are_capped_at_the_requested_limit():
    adds = [_FakeAdd(f"Add{i}", 20.0 - i) for i in range(5)]
    drops = [_drop(f"Drop{i}", float(i)) for i in range(5)]

    assert len(suggest_swaps(adds, drops, limit=2)) == 2
