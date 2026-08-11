from types import SimpleNamespace

import numpy as np
import pandas as pd

from edge_engine.model.config import ModelConfig
from edge_engine.ranking import output
from edge_engine.ranking.output import _explanation_for, _join_sentences, _startable_positions
from edge_engine.roster.models import LeagueConfig, Player, ScoringSettings, WaiverConfig


def _config(trailing_window=2, flag_margin=3.0):
    return ModelConfig(
        train_seasons=[2023], validation_season=2024, trailing_window=trailing_window,
        flag_margin=flag_margin, injury_ahead_usage_gap=0.15, injury_lookback_weeks=2,
    )


def _league_config(lineup_slots=None):
    return LeagueConfig(
        season=2025,
        scoring=ScoringSettings(ppr_type="full_ppr"),
        lineup_slots=lineup_slots or {"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1, "BENCH": 6},
        waivers=WaiverConfig(system="FAAB", season_budget=100, min_bid=1, clear_day="Wednesday"),
    )


# ---- explanation assembly ----

def test_join_sentences_does_not_double_up_full_stops():
    # Regression: some fragments already end in "." and some don't, so a
    # naive ". ".join produced "...show a trend.. Usage spike...".
    text = _join_sentences(["usage is up.", "Injury context here", "Suspended."])

    assert ".." not in text
    assert text == "usage is up. Injury context here. Suspended."


def test_join_sentences_drops_empty_fragments():
    assert _join_sentences(["", "   ", "real text"]) == "real text."
    assert _join_sentences([]) == ""


def _row(**overrides):
    row = dict(
        player_id="P1", season=2025, week=3,
        has_injury_context=False, injury_explanation="",
        roster_status_note=np.nan,
    )
    row.update(overrides)
    return SimpleNamespace(**row)


def test_nan_roster_status_note_does_not_leak_the_string_nan(monkeypatch):
    # pandas coerces a mostly-None object column to NaN, and NaN is truthy
    # in Python -- a plain `if row.roster_status_note` appended the literal
    # "nan" to every single explanation.
    monkeypatch.setattr(output, "usage_trend_explanation", lambda *a, **k: "snap share 38% -> 64%")

    text = _explanation_for(pd.DataFrame(), _row(roster_status_note=np.nan), window=2)

    assert "nan" not in text.lower()
    assert text == "snap share 38% -> 64%."


def test_real_roster_status_note_is_appended(monkeypatch):
    monkeypatch.setattr(output, "usage_trend_explanation", lambda *a, **k: "snap share up")

    text = _explanation_for(
        pd.DataFrame(), _row(roster_status_note="Officially listed as Suspended (SUS)."), window=2
    )

    assert "Suspended" in text


def test_injury_explanation_is_appended_only_when_context_exists(monkeypatch):
    monkeypatch.setattr(output, "usage_trend_explanation", lambda *a, **k: "snap share up")

    with_ctx = _explanation_for(
        pd.DataFrame(), _row(has_injury_context=True, injury_explanation="Starter is Out"), window=2
    )
    without_ctx = _explanation_for(
        pd.DataFrame(), _row(has_injury_context=False, injury_explanation="Starter is Out"), window=2
    )

    assert "Starter is Out" in with_ctx
    assert "Starter is Out" not in without_ctx


def test_quarterbacks_use_their_own_usage_explanation_not_the_receiving_one(monkeypatch):
    # usage_trend_explanation reads snap/target share, which are null for a
    # passer -- it would report "usage steady" for every QB regardless.
    monkeypatch.setattr(output, "usage_trend_explanation", lambda *a, **k: "RECEIVING TREND")

    text = _explanation_for(
        pd.DataFrame(), _row(usage_explanation="pass attempts 30 -> 42 over 2 wks (up)"), window=2
    )

    assert "pass attempts 30 -> 42" in text
    assert "RECEIVING TREND" not in text


def test_blank_or_missing_qb_explanation_falls_back_to_the_receiving_trend(monkeypatch):
    monkeypatch.setattr(output, "usage_trend_explanation", lambda *a, **k: "RECEIVING TREND")

    for value in (None, np.nan, "", "   "):
        text = _explanation_for(pd.DataFrame(), _row(usage_explanation=value), window=2)
        assert "RECEIVING TREND" in text


# ---- startable positions ----

def test_flex_league_keeps_rb_wr_te_even_without_dedicated_slots():
    positions = _startable_positions(_league_config({"QB": 1, "FLEX": 2, "BENCH": 5}))

    assert {"RB", "WR", "TE"}.issubset(positions)
    assert "BENCH" not in positions


def test_zero_count_and_bench_slots_are_not_startable():
    positions = _startable_positions(_league_config({"QB": 1, "RB": 2, "TE": 0, "BENCH": 6, "IR": 1}))

    assert positions == {"QB", "RB"}


# ---- end-to-end ranking assembly ----

class _StubSource:
    def __init__(self, free_agents, lineup_slots=None):
        self._free_agents = free_agents
        self._lineup_slots = lineup_slots

    def get_free_agents(self):
        return self._free_agents

    def get_league_config(self):
        return _league_config(self._lineup_slots)


def _scored(rows):
    return pd.DataFrame(rows)


def _scored_row(player_id, position="RB", predicted=20.0, baseline=10.0, **overrides):
    row = dict(
        player_id=player_id, season=2025, week=3, position=position, team="GB",
        predicted_score=predicted, baseline_score=baseline, flagged=True,
        has_injury_context=False, injury_explanation="", roster_status_note=np.nan,
    )
    row.update(overrides)
    return row


def _patch_rankings(monkeypatch, tmp_path, source, scored):
    path = tmp_path / "player_week.parquet"
    pd.DataFrame([{"season": 2025, "week": 3, "player_id": "X", "snap_pct": 0.5}]).to_parquet(path)
    monkeypatch.setattr(output, "PLAYER_WEEK_PATH", path)
    monkeypatch.setattr(output, "get_default_source", lambda: source)
    monkeypatch.setattr(output, "score_latest_week", lambda config: scored)
    monkeypatch.setattr(output, "usage_trend_explanation", lambda *a, **k: "usage steady")


def test_only_free_agents_are_ranked_not_the_whole_scored_pool(tmp_path, monkeypatch):
    source = _StubSource([Player(name="Free Agent", position="RB", team="GB", player_id="FA1", match_status="matched")])
    scored = _scored([_scored_row("FA1"), _scored_row("ROSTERED_ELSEWHERE")])
    _patch_rankings(monkeypatch, tmp_path, source, scored)

    candidates, unresolved = output.build_free_agent_rankings(_config())

    assert [c.player_id for c in candidates] == ["FA1"]
    assert unresolved == []


def test_unresolved_free_agents_are_surfaced_not_silently_dropped(tmp_path, monkeypatch):
    source = _StubSource([
        Player(name="Good Name", position="RB", team="GB", player_id="FA1", match_status="matched"),
        Player(name="Typo Playerr", position="RB", team="GB", player_id=None, match_status="unmatched"),
    ])
    _patch_rankings(monkeypatch, tmp_path, source, _scored([_scored_row("FA1")]))

    candidates, unresolved = output.build_free_agent_rankings(_config())

    assert [c.player_id for c in candidates] == ["FA1"]
    assert [p.name for p in unresolved] == ["Typo Playerr"]


def test_positions_the_league_does_not_start_are_filtered_out(tmp_path, monkeypatch):
    # A player nflverse correctly lists as a DB can still sit in ESPN's WR
    # free-agent pool; a standard league shouldn't recommend him.
    source = _StubSource([
        Player(name="Real RB", position="RB", team="GB", player_id="RB1", match_status="matched"),
        Player(name="Converted DB", position="WR", team="GB", player_id="DB1", match_status="matched"),
    ])
    scored = _scored([_scored_row("RB1", position="RB"), _scored_row("DB1", position="DB")])
    _patch_rankings(monkeypatch, tmp_path, source, scored)

    candidates, _ = output.build_free_agent_rankings(_config())

    assert [c.player_id for c in candidates] == ["RB1"]


def test_candidates_are_sorted_by_predicted_score_with_tiers_assigned(tmp_path, monkeypatch):
    source = _StubSource([
        Player(name="Low", position="RB", team="GB", player_id="LOW", match_status="matched"),
        Player(name="High", position="RB", team="GB", player_id="HIGH", match_status="matched"),
    ])
    scored = _scored([
        _scored_row("LOW", predicted=11.0, baseline=10.0),   # margin 1 -> Low
        _scored_row("HIGH", predicted=20.0, baseline=10.0),  # margin 10 -> High
    ])
    _patch_rankings(monkeypatch, tmp_path, source, scored)

    candidates, _ = output.build_free_agent_rankings(_config(flag_margin=3.0))

    assert [c.player_id for c in candidates] == ["HIGH", "LOW"]
    assert candidates[0].confidence_tier == "High"
    assert candidates[1].confidence_tier == "Low"
    # The user-entered name is preserved, not the scored frame's id.
    assert candidates[0].name == "High"
