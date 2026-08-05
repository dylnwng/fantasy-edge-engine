from edge_engine.ranking.output import RankedFreeAgent
from edge_engine.ranking.roster_fit import apply_roster_fit
from edge_engine.roster.models import LeagueConfig, Player, ScoringSettings, WaiverConfig

LINEUP_SLOTS = {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "DST": 1, "K": 1, "BENCH": 6}


def _league_config(lineup_slots=None) -> LeagueConfig:
    return LeagueConfig(
        season=2026,
        scoring=ScoringSettings(ppr_type="full_ppr"),
        lineup_slots=lineup_slots or LINEUP_SLOTS,
        waivers=WaiverConfig(system="FAAB", season_budget=100, min_bid=1, clear_day="Wednesday"),
    )


def _rostered_player(name, position, team="XXX", bye_week=None) -> Player:
    return Player(name=name, position=position, team=team, player_id=name, bye_week=bye_week, match_status="matched")


def _candidate(player_id, name, position, team, predicted_score) -> RankedFreeAgent:
    return RankedFreeAgent(
        player_id=player_id,
        name=name,
        position=position,
        team=team,
        season=2023,
        week=5,
        predicted_score=predicted_score,
        baseline_score=predicted_score - 5,  # arbitrary, not under test here
        confidence_tier="Medium",
        explanation="usage trend placeholder",
    )


def test_thin_position_outranks_deep_position_at_equal_opportunity_score():
    # Directly the PRD's requirement 5 acceptance criterion: 4 rostered RBs,
    # 1 rostered TE -> an equally-trending TE should rank above an
    # equally-trending RB once roster fit is applied.
    rostered = [
        _rostered_player("RB1", "RB"),
        _rostered_player("RB2", "RB"),
        _rostered_player("RB3", "RB"),
        _rostered_player("RB4", "RB"),
        _rostered_player("TE1", "TE"),
    ]
    candidates = [
        _candidate("cand_rb", "Candidate RB", "RB", "AAA", predicted_score=20.0),
        _candidate("cand_te", "Candidate TE", "TE", "BBB", predicted_score=20.0),
    ]

    results = apply_roster_fit(candidates, _league_config(), rostered, bye_weeks_by_player_id={})

    order = [r.candidate.player_id for r in results]
    assert order == ["cand_te", "cand_rb"]
    # And it's not a tie broken arbitrarily -- TE's final score is strictly higher.
    te_result = next(r for r in results if r.candidate.player_id == "cand_te")
    rb_result = next(r for r in results if r.candidate.player_id == "cand_rb")
    assert te_result.roster_fit_score > rb_result.roster_fit_score
    assert te_result.scarcity_multiplier > 1.0  # thin position -> boosted
    assert rb_result.scarcity_multiplier < 1.0  # deep position -> discounted


def test_bye_week_collision_is_explained_not_baked_into_score():
    rostered = [_rostered_player("Existing RB", "RB", bye_week=7)]
    candidate = _candidate("cand_rb", "Candidate RB", "RB", "AAA", predicted_score=20.0)

    with_collision = apply_roster_fit(
        [candidate], _league_config(), rostered, bye_weeks_by_player_id={"cand_rb": 7}
    )[0]
    without_collision = apply_roster_fit(
        [candidate], _league_config(), rostered, bye_weeks_by_player_id={"cand_rb": 9}
    )[0]

    assert with_collision.bye_week_collision is not None
    assert "week 7" in with_collision.bye_week_collision
    assert "Existing RB" in with_collision.bye_week_collision
    assert with_collision.bye_week_collision in with_collision.final_explanation

    assert without_collision.bye_week_collision is None

    # The collision is informational only -- it must not change the score.
    assert with_collision.roster_fit_score == without_collision.roster_fit_score


def test_pre_rerank_score_stays_visible_as_a_distinct_field():
    rostered = [_rostered_player("RB1", "RB")]
    candidate = _candidate("cand_rb", "Candidate RB", "RB", "AAA", predicted_score=20.0)

    result = apply_roster_fit([candidate], _league_config(), rostered, bye_weeks_by_player_id={})[0]

    # requirement 5's third acceptance criterion: the opportunity score
    # (pre-re-rank) must still be inspectable, separately from the final
    # roster-fit score, not overwritten or lost in the re-ranking step.
    assert result.candidate.predicted_score == 20.0
    assert result.roster_fit_score != result.candidate.predicted_score
