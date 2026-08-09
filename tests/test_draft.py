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
    assert any("did not resolve" in g for g in gaps)


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
