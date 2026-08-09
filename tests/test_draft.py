import pytest

from edge_engine.draft.board import BoardPlayer, assign_tiers, build_board
from edge_engine.draft.market import ManualMarketPriceSource, MarketPrice
from edge_engine.draft.tracker import DraftState, Pick, unfilled_slots


def _price(name, position="RB", adp=1.0, player_id=None, team="KC", status="matched"):
    return MarketPrice(
        name=name, position=position, team=team, adp=adp,
        player_id=player_id if player_id is not None else name,
        match_status=status,
    )


def _bp(name, position="RB", adp=1.0, tier=1, rank=1, low_confidence=False):
    return BoardPlayer(
        name=name, position=position, team="KC", adp=adp,
        position_rank=rank, tier=tier, player_id=name, low_confidence=low_confidence,
    )


# ---- tiers ----

def test_tier_breaks_where_the_market_leaves_a_gap():
    prices = [
        _price("A", adp=1.0), _price("B", adp=3.0), _price("C", adp=5.0),   # tight cluster
        _price("D", adp=40.0), _price("E", adp=42.0),                        # after a big gap
    ]
    tiers = {name: tier for (p, _rank, tier) in assign_tiers(prices)["RB"] for name in [p.name]}

    assert tiers["A"] == tiers["B"] == tiers["C"] == 1
    assert tiers["D"] == tiers["E"] == 2


def test_uniformly_spaced_adp_produces_a_single_tier():
    # Consecutive integers never exceed the gap threshold. Worth pinning:
    # the bundled example ADP is uniformly spaced, so a board built on it
    # legitimately shows one tier per position rather than being broken.
    prices = [_price(f"P{i}", adp=float(i)) for i in range(1, 10)]
    tiers = {tier for (_p, _rank, tier) in assign_tiers(prices)["RB"]}
    assert tiers == {1}


def test_position_rank_is_per_position_and_adp_ordered():
    prices = [
        _price("RB2", position="RB", adp=5.0), _price("RB1", position="RB", adp=1.0),
        _price("WR1", position="WR", adp=3.0),
    ]
    result = assign_tiers(prices)
    rb_ranks = {p.name: rank for (p, rank, _t) in result["RB"]}

    assert rb_ranks == {"RB1": 1, "RB2": 2}
    assert [rank for (_p, rank, _t) in result["WR"]] == [1]


# ---- board ----

def test_positive_divergence_is_tagged_as_the_market_using_the_wrong_window():
    board, _ = build_board([_price("Riser", player_id="R1")], divergence_by_id={"R1": 2.0})
    assert any("usage ran ahead" in t for t in board[0].tags)
    assert not board[0].low_confidence


def test_negative_divergence_is_tagged_as_a_role_he_didnt_hold():
    board, _ = build_board([_price("Faller", player_id="F1")], divergence_by_id={"F1": -2.0})
    assert any("ADP may be pricing a role he didn't hold" in t for t in board[0].tags)


def test_divergence_inside_the_band_gets_no_tag():
    board, _ = build_board([_price("Steady", player_id="S1")], divergence_by_id={"S1": 0.4})
    assert board[0].tags == ()
    assert not board[0].low_confidence


def test_rookies_are_marked_low_confidence_and_never_look_like_veterans():
    board, _ = build_board(
        [_price("Rook", player_id="R9")], divergence_by_id={}, rookie_ids={"R9"}
    )
    assert board[0].low_confidence
    assert any("rookie" in t for t in board[0].tags)


def test_unresolved_players_stay_on_the_board_priced_at_adp():
    unresolved = MarketPrice(name="Ghost", position="WR", team="KC", adp=50.0, player_id=None)
    board, gaps = build_board([unresolved])

    assert len(board) == 1  # surfaced, not dropped
    assert board[0].low_confidence
    assert any("couldn't be matched" in g for g in gaps)


def test_kickers_and_defenses_are_not_reported_as_a_data_problem():
    """K and DST have no player-level usage data anywhere in nflverse, so an
    untagged kicker is the system working as designed. Telling the user to go
    check his spelling sends him hunting for a bug that does not exist."""
    prices = [
        MarketPrice(name="A Kicker", position="K", team="KC", adp=150.0, player_id=None),
        MarketPrice(name="A Defense", position="DST", team="SF", adp=140.0, player_id=None),
    ]
    board, gaps = build_board(prices)

    assert len(board) == 2  # still draftable, still on the board
    blob = " ".join(gaps)
    assert "expected" in blob
    assert "spelling" not in blob
    assert "couldn't be matched" not in blob


def test_a_real_unmatched_skill_player_is_still_reported_separately():
    """The expected K/DST case must not swallow the case worth looking into."""
    prices = [
        MarketPrice(name="A Kicker", position="K", team="KC", adp=150.0, player_id=None),
        MarketPrice(name="Ghost", position="WR", team="KC", adp=50.0, player_id=None),
    ]
    _, gaps = build_board(prices)

    assert any("expected" in g for g in gaps)
    assert any("1 player(s) couldn't be matched" in g for g in gaps)


def test_board_without_any_divergence_is_still_a_valid_adp_board():
    # 5-lite's whole premise: it degrades to well-organised market
    # pricing rather than to nothing.
    board, _ = build_board([_price("A", adp=1.0), _price("B", adp=2.0)])
    assert [b.name for b in board] == ["A", "B"]


def test_board_is_returned_in_adp_order_across_positions():
    board, _ = build_board(
        [_price("WR_early", position="WR", adp=1.0), _price("RB_late", position="RB", adp=9.0)]
    )
    assert [b.name for b in board] == ["WR_early", "RB_late"]


# ---- tracker ----

def test_applying_the_same_pick_twice_removes_only_one_player():
    # A poll against a real draft API will hand you duplicates.
    state = DraftState(board=[_bp("A"), _bp("B")])
    state = state.apply(Pick(1, 1, "A")).apply(Pick(1, 1, "A"))

    assert len(state.picks) == 1
    assert [b.name for b in state.available()] == ["B"]


def test_out_of_order_picks_reconcile_by_slot():
    state = DraftState(board=[_bp("A"), _bp("B"), _bp("C")])
    state = state.apply_all([Pick(1, 3, "C"), Pick(1, 1, "A")])

    assert [p.player_name for p in state.picks_in_order] == ["A", "C"]
    assert [b.name for b in state.available()] == ["B"]


def test_tiers_remaining_puts_the_thinnest_tier_first():
    board = [
        _bp("A", position="RB", tier=1), _bp("B", position="RB", tier=2),
        _bp("C", position="RB", tier=2), _bp("D", position="WR", tier=1),
        _bp("E", position="WR", tier=1), _bp("F", position="WR", tier=1),
    ]
    tiers = DraftState(board=board).tiers_remaining()

    assert (tiers[0].position, tiers[0].tier, tiers[0].remaining) == ("RB", 1, 1)
    assert tiers[-1].remaining == 3


def test_tier_counts_shrink_as_players_are_drafted():
    board = [_bp("A", tier=1), _bp("B", tier=1)]
    state = DraftState(board=board).apply(Pick(1, 1, "A"))

    rb_tier1 = next(t for t in state.tiers_remaining() if t.position == "RB" and t.tier == 1)
    assert rb_tier1.remaining == 1
    assert rb_tier1.names == ("B",)


def test_positional_run_is_detected_and_purely_descriptive():
    board = [_bp(f"W{i}", position="WR") for i in range(5)] + [_bp("R1", position="RB")]
    picks = [Pick(1, i + 1, n) for i, n in enumerate(["W0", "R1", "W1", "W2", "W3", "W4"])]
    state = DraftState(board=board).apply_all(picks)

    run = state.positional_run(lookback=6, threshold=4)
    assert run is not None and "WR" in run
    # No recommendation attached -- whether to chase or fade depends on
    # a roster the tool doesn't model.
    for word in ("should", "recommend", "take", "avoid"):
        assert word not in run.lower()


def test_no_run_reported_before_enough_picks_exist():
    state = DraftState(board=[_bp("A")]).apply(Pick(1, 1, "A"))
    assert state.positional_run(lookback=6) is None


def test_best_available_respects_position_filter():
    board = [_bp("RB1", position="RB", adp=1.0), _bp("WR1", position="WR", adp=2.0)]
    state = DraftState(board=board)

    assert [b.name for b in state.best_available(5, position="WR")] == ["WR1"]


def test_unfilled_slots_counts_down_as_you_draft():
    slots = {"RB": 2, "WR": 2, "FLEX": 1, "BENCH": 6}
    assert unfilled_slots([], slots) == {"RB": 2, "WR": 2, "FLEX": 1}

    after = unfilled_slots([_bp("A", position="RB")], slots)
    assert after["RB"] == 1
    assert "BENCH" not in after  # bench isn't a starting requirement


# ---- market source ----

def test_missing_adp_file_raises_a_clear_message(tmp_path):
    source = ManualMarketPriceSource(path=tmp_path / "nope.csv")
    with pytest.raises(RuntimeError, match="doesn't exist"):
        source.get_market_prices()


def test_missing_column_raises_a_clear_message(tmp_path):
    path = tmp_path / "adp.csv"
    path.write_text("name,position,team\nA,RB,KC\n")  # no adp column
    source = ManualMarketPriceSource(path=path, lookup=_StubLookup())

    with pytest.raises(RuntimeError, match="adp"):
        source.get_market_prices()


def test_non_numeric_adp_raises_a_clear_message(tmp_path):
    path = tmp_path / "adp.csv"
    path.write_text("name,position,team,adp\nA,RB,KC,early\n")
    source = ManualMarketPriceSource(path=path, lookup=_StubLookup())

    with pytest.raises(RuntimeError, match="isn't a number"):
        source.get_market_prices()


def test_empty_adp_file_raises_rather_than_yielding_an_empty_board(tmp_path):
    path = tmp_path / "adp.csv"
    path.write_text("name,position,team,adp\n")
    source = ManualMarketPriceSource(path=path, lookup=_StubLookup())

    with pytest.raises(RuntimeError, match="no rows"):
        source.get_market_prices()


class _StubLookup:
    def resolve(self, name, position, team):
        return name, "matched", ""


def test_two_players_sharing_a_name_are_removed_one_at_a_time():
    # The NFL routinely has two active players with the same name, and
    # ESPN's draft feed identifies picks BY NAME. A naive name-set filter
    # deleted both from the pool the moment either was taken -- silently
    # hiding an available player on draft night.
    board = [_bp("Michael Thomas", adp=10.0), _bp("Michael Thomas", adp=80.0), _bp("Other", adp=20.0)]
    state = DraftState(board=board)

    after_one = state.apply(Pick(1, 1, "Michael Thomas")).available()
    names_left = [(b.name, b.adp) for b in after_one]
    # The earlier-ADP one is assumed taken; the later one survives.
    assert ("Michael Thomas", 80.0) in names_left
    assert ("Michael Thomas", 10.0) not in names_left

    after_two = state.apply_all(
        [Pick(1, 1, "Michael Thomas"), Pick(1, 2, "Michael Thomas")]
    ).available()
    assert [b.name for b in after_two] == ["Other"]


# ---- live ADP source ----

def _ffc_payload(players):
    return {"status": "Success", "players": players}


def _ffc_source(tmp_path, payload, **kw):
    """A source wired to a fixture file instead of the network."""
    import json

    from edge_engine.draft.market import FantasyFootballCalculatorSource

    src = FantasyFootballCalculatorSource(
        year=2026, lookup=_StubLookup(), cache_dir=tmp_path, **kw
    )
    src._cache_path.parent.mkdir(parents=True, exist_ok=True)
    src._cache_path.write_text(json.dumps(payload))
    return src


def test_live_adp_normalises_position_labels(tmp_path):
    # FFC calls team defenses DEF and kickers PK; this project uses DST
    # and K. Without normalising, every kicker and defense falls out of
    # the board's position filter silently.
    payload = _ffc_payload([
        {"name": "Some Defense", "position": "DEF", "team": "SF", "adp": 120.0},
        {"name": "Some Kicker", "position": "PK", "team": "DAL", "adp": 140.0},
        {"name": "Some Back", "position": "RB", "team": "ATL", "adp": 2.0},
    ])
    prices = {p.name: p for p in _ffc_source(tmp_path, payload).get_market_prices()}

    assert prices["Some Defense"].position == "DST"
    assert prices["Some Kicker"].position == "K"
    assert prices["Some Back"].position == "RB"


def test_live_adp_snaps_to_a_league_size_the_service_publishes(tmp_path):
    # FFC only publishes 8/10/12/14-team boards; an 11-team league should
    # get the nearest rather than a 404.
    payload = _ffc_payload([{"name": "A", "position": "RB", "team": "SF", "adp": 1.0}])
    assert _ffc_source(tmp_path, payload, teams=11).teams in (10, 12)
    assert _ffc_source(tmp_path, payload, teams=13).teams in (12, 14)


def test_live_adp_maps_league_scoring_to_the_right_board(tmp_path):
    payload = _ffc_payload([{"name": "A", "position": "RB", "team": "SF", "adp": 1.0}])
    half = _ffc_source(tmp_path, payload, ppr_type="half_ppr")
    assert "half-ppr" in half._cache_path.name
    assert "half ppr" in half.describe()


def test_live_adp_rejects_a_scoring_type_the_service_has_no_board_for():
    from edge_engine.draft.market import FantasyFootballCalculatorSource

    with pytest.raises(RuntimeError, match="isn't available for scoring type"):
        FantasyFootballCalculatorSource(year=2026, ppr_type="double_ppr")


def test_live_adp_uses_cache_without_touching_the_network(tmp_path):
    # The cache write in _ffc_source is fresh, so no fetch should happen.
    payload = _ffc_payload([{"name": "Cached Guy", "position": "WR", "team": "KC", "adp": 5.0}])
    prices = _ffc_source(tmp_path, payload).get_market_prices()
    assert [p.name for p in prices] == ["Cached Guy"]
