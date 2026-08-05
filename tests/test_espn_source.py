import pytest

from edge_engine.roster.espn_source import (
    _bonuses_from_scoring_format,
    _map_lineup_slots,
    _normalize_team,
    _ppr_type_from_scoring_format,
)


def _scoring_item(stat_id, points):
    return {"id": stat_id, "points": points}


def test_ppr_type_detects_standard_half_and_full():
    assert _ppr_type_from_scoring_format([_scoring_item(53, 0)]) == "standard"
    assert _ppr_type_from_scoring_format([_scoring_item(53, 0.5)]) == "half_ppr"
    assert _ppr_type_from_scoring_format([_scoring_item(53, 1.0)]) == "full_ppr"


def test_ppr_type_missing_reception_item_defaults_to_standard():
    # No item with id=53 at all -- treat as 0 pts/reception, not a crash.
    assert _ppr_type_from_scoring_format([_scoring_item(4, 4.0)]) == "standard"


def test_ppr_type_raises_on_unrecognized_value_instead_of_guessing():
    # 1.5 PPR is a real (if less common) league format this code doesn't
    # support -- it must fail loudly, not silently mis-score every player.
    with pytest.raises(ValueError, match="1.5"):
        _ppr_type_from_scoring_format([_scoring_item(53, 1.5)])


def test_bonuses_are_deltas_from_nflverse_standard_baseline():
    # League matches nflverse's standard baseline exactly (4/6/6/-2/-2) --
    # no bonus should be recorded, since there's nothing extra to add.
    scoring_format = [
        _scoring_item(4, 4.0),  # pass TD
        _scoring_item(25, 6.0),  # rush TD
        _scoring_item(43, 6.0),  # rec TD
        _scoring_item(20, -2.0),  # INT
        _scoring_item(72, -2.0),  # fumble lost
    ]
    assert _bonuses_from_scoring_format(scoring_format) == {}


def test_bonuses_captures_real_deltas_only():
    scoring_format = [
        _scoring_item(4, 6.0),  # league pays 6 for pass TD, not nflverse's baseline 4
        _scoring_item(25, 6.0),  # matches baseline -- no bonus
        _scoring_item(20, -4.0),  # league docks double for INTs
    ]
    bonuses = _bonuses_from_scoring_format(scoring_format)
    assert bonuses == {"pass_td": 2.0, "interception": -2.0}


def test_map_lineup_slots_combines_flex_labels_and_renames():
    position_slot_counts = {
        "QB": 1,
        "RB": 2,
        "WR": 2,
        "TE": 1,
        "RB/WR/TE": 1,
        "RB/WR": 0,  # zero-count slots should be dropped, not shown as noise
        "D/ST": 1,
        "K": 1,
        "BE": 6,
        "IR": 1,
    }
    slots = _map_lineup_slots(position_slot_counts)

    assert slots == {
        "QB": 1,
        "RB": 2,
        "WR": 2,
        "TE": 1,
        "FLEX": 1,
        "DST": 1,
        "K": 1,
        "BENCH": 6,
        "IR": 1,
    }


def test_map_lineup_slots_sums_multiple_flex_style_slots():
    # A superflex-ish league with both a standard flex and an OP slot --
    # both should roll into one combined FLEX count.
    position_slot_counts = {"QB": 1, "RB/WR/TE": 1, "OP": 1}
    slots = _map_lineup_slots(position_slot_counts)
    assert slots["FLEX"] == 2


def test_normalize_team_maps_known_espn_alias():
    assert _normalize_team("WSH") == "WAS"
    assert _normalize_team("LAR") == "LAR"  # already nflverse-recognized, untouched
    assert _normalize_team("KC") == "KC"
