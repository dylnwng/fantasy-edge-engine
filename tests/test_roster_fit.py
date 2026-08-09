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


def test_negative_lineup_slots_does_not_invert_the_ranking():
    # A malformed league_config.yaml (e.g. a stray "-" typo) used to
    # produce a negative scarcity multiplier, silently flipping the sign
    # of roster_fit_score and inverting the ranking with no error.
    bad_config = _league_config(lineup_slots={"RB": -5})
    candidate = _candidate("cand_rb", "Candidate RB", "RB", "AAA", predicted_score=20.0)

    result = apply_roster_fit([candidate], bad_config, rostered=[], bye_weeks_by_player_id={})[0]

    assert result.scarcity_multiplier >= 0.0
    assert result.roster_fit_score >= 0.0


def _fit_candidate(name, tier, score):
    from edge_engine.ranking.roster_fit import RosterFitCandidate

    return RosterFitCandidate(
        candidate=RankedFreeAgent(
            player_id=name, name=name, position="RB", team="AAA", season=2023, week=5,
            predicted_score=score, baseline_score=score - 5, confidence_tier=tier,
            explanation="placeholder",
        ),
        roster_fit_score=score,
        scarcity_multiplier=1.0,
        position_need=1.0,
        rostered_count_at_position=0,
        bye_week_collision=None,
        final_explanation="placeholder",
    )


def test_default_view_shows_only_actionable_tiers():
    # A real live run produced 293 candidates: 5 High, 38 Medium, 249
    # Low. Printing all of them buries the handful worth a bid.
    from edge_engine.ranking.roster_fit import select_visible

    results = [
        _fit_candidate("A", "High", 30.0),
        _fit_candidate("B", "Medium", 20.0),
        _fit_candidate("C", "Low", 10.0),
        _fit_candidate("D", "Low", 9.0),
    ]
    visible, omitted = select_visible(results)

    assert [r.candidate.name for r in visible] == ["A", "B"]
    assert omitted == 2


def test_all_flag_shows_everything_and_omits_nothing():
    from edge_engine.ranking.roster_fit import select_visible

    results = [_fit_candidate("A", "High", 30.0), _fit_candidate("C", "Low", 10.0)]
    visible, omitted = select_visible(results, show_all=True)

    assert len(visible) == 2
    assert omitted == 0


def test_top_n_ignores_tier_and_reports_the_remainder():
    from edge_engine.ranking.roster_fit import select_visible

    results = [_fit_candidate(n, "Low", s) for n, s in [("A", 30.0), ("B", 20.0), ("C", 10.0)]]
    visible, omitted = select_visible(results, top=2)

    assert [r.candidate.name for r in visible] == ["A", "B"]
    assert omitted == 1


def test_no_actionable_candidates_still_reports_how_many_were_hidden():
    from edge_engine.ranking.roster_fit import select_visible

    results = [_fit_candidate("C", "Low", 10.0), _fit_candidate("D", "Low", 9.0)]
    visible, omitted = select_visible(results)

    assert visible == []
    assert omitted == 2


def test_startable_positions_excludes_positions_the_league_never_starts():
    # Bo Melton is carried as a WR by ESPN (so he lands in the free-agent
    # pool) but nflverse has him as a DB after Green Bay converted him to
    # cornerback. Without this filter a defensive back appears in a
    # standard league's waiver recommendations, which reads as a bug.
    from edge_engine.ranking.output import _startable_positions

    standard = _league_config(lineup_slots={"QB": 1, "RB": 2, "WR": 2, "TE": 1,
                                            "FLEX": 1, "K": 1, "DST": 1, "BENCH": 6})
    positions = _startable_positions(standard)

    assert "DB" not in positions and "LB" not in positions
    assert {"QB", "RB", "WR", "TE", "K", "DST"} <= positions
    assert "BENCH" not in positions and "FLEX" not in positions


def test_startable_positions_keeps_defensive_positions_for_an_idp_league():
    # Derived from the league's own slots, not hardcoded, so a league that
    # genuinely starts defensive backs keeps them.
    from edge_engine.ranking.output import _startable_positions

    idp = _league_config(lineup_slots={"QB": 1, "RB": 2, "WR": 2, "DB": 2, "LB": 2, "BENCH": 6})
    positions = _startable_positions(idp)

    assert "DB" in positions and "LB" in positions


def test_flex_slot_pulls_in_flex_eligible_positions():
    from edge_engine.ranking.output import _startable_positions

    only_flex = _league_config(lineup_slots={"QB": 1, "FLEX": 1, "BENCH": 6})
    assert {"RB", "WR", "TE"} <= _startable_positions(only_flex)
