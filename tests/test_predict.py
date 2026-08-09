import numpy as np
import pandas as pd
import pytest

from edge_engine.model import predict
from edge_engine.model.config import ModelConfig
from edge_engine.model.injury_context import InjuryContext
from edge_engine.roster.models import LeagueConfig, ScoringSettings, WaiverConfig

INJURY_COLUMNS = ["season", "week", "team", "gsis_id", "full_name", "position", "report_status", "report_primary_injury"]


def _config(trailing_window=2, flag_margin=3.0):
    return ModelConfig(
        train_seasons=[2023],
        validation_season=2024,
        trailing_window=trailing_window,
        flag_margin=flag_margin,
        injury_ahead_usage_gap=0.15,
        injury_lookback_weeks=2,
    )


class _StubSource:
    def get_league_config(self):
        return LeagueConfig(
            season=2025,
            scoring=ScoringSettings(ppr_type="full_ppr"),
            lineup_slots={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
            waivers=WaiverConfig(system="FAAB", season_budget=100, min_bid=1, clear_day="Wednesday"),
        )


class _SpyModel:
    """Records the exact feature frame it was asked to score and returns a
    caller-supplied array, so tests can assert on what reached the model
    without a real trained xgboost artifact."""

    def __init__(self, scores):
        self.scores = scores
        self.seen = None

    def predict(self, X):
        self.seen = X.reset_index(drop=True)
        return np.asarray(self.scores, dtype=float)


def _usage_row(player_id, season, week, position="RB", team="GB", **overrides):
    row = dict(
        season=season, week=week, player_id=player_id, position=position, team=team,
        snap_pct=0.5, target_share=0.2, air_yards_share=0.2,
        red_zone_touches=1, red_zone_target_share=0.2,
    )
    row.update(overrides)
    return row


def _three_games(player_id, season=2025, **overrides):
    # window=2's trend feature is `value - value.shift(2)`, which is only
    # non-null from a player's THIRD played game onward -- score_as_of_week
    # drops any row missing a feature, trend included, so every fixture
    # needs 3 games to survive to a scoreable "latest" row.
    return [_usage_row(player_id, season, w, **overrides) for w in (1, 2, 3)]


def _write_player_week(tmp_path, rows, monkeypatch):
    path = tmp_path / "player_week.parquet"
    pd.DataFrame(rows).to_parquet(path)
    monkeypatch.setattr(predict, "PLAYER_WEEK_PATH", path)


def _stub_points(rows, value):
    return pd.DataFrame(
        [{"season": r["season"], "week": r["week"], "player_id": r["player_id"], "points": value} for r in rows]
    )


def _patch_common(monkeypatch, rows, model, points_value=10.0):
    monkeypatch.setattr(predict, "get_default_source", lambda: _StubSource())
    monkeypatch.setattr(predict, "compute_points_for_seasons", lambda seasons, scoring: _stub_points(rows, points_value))
    monkeypatch.setattr(predict, "load_model", lambda: model)
    monkeypatch.setattr(predict, "load_injury_reports", lambda seasons, force_refresh=False: pd.DataFrame(columns=INJURY_COLUMNS))
    monkeypatch.setattr(predict, "get_injury_context", lambda *a, **k: InjuryContext(False, None, None, None, ""))
    monkeypatch.setattr(predict, "get_flagged_roster_statuses", lambda: {})
    monkeypatch.setattr("edge_engine.model.predict_qb.score_qbs_as_of_week", lambda *a, **k: pd.DataFrame())


def test_players_without_a_full_trailing_window_are_excluded(tmp_path, monkeypatch):
    # Only one game played before target_week=2 -- window=2 needs at
    # least two just for the rolling average, let alone the trend.
    rows = [_usage_row("P1", 2025, 1)]
    _write_player_week(tmp_path, rows, monkeypatch)
    _patch_common(monkeypatch, rows, model=_SpyModel([]))

    result = predict.score_as_of_week(2025, target_week=2, config=_config())

    assert result.empty
    for col in ["predicted_score", "baseline_score", "flagged", "has_injury_context", "injury_explanation", "roster_status_note"]:
        assert col in result.columns


def test_only_games_strictly_before_target_week_feed_the_features(tmp_path, monkeypatch):
    # Regression guard for the exact bug the docstring calls out: an
    # earlier version scored off a player's single latest row in the
    # *entire* ingested file, leaking data from after target_week into a
    # retrospective "as of week N" prediction once later weeks exist.
    rows_truncated = [
        _usage_row("P1", 2025, 1, snap_pct=0.3),
        _usage_row("P1", 2025, 2, snap_pct=0.4),
        _usage_row("P1", 2025, 3, snap_pct=0.6),
    ]
    rows_with_future = rows_truncated + [
        _usage_row("P1", 2025, 4, snap_pct=0.9),  # ingested later, after the fact
        _usage_row("P1", 2025, 5, snap_pct=0.95),
    ]

    def _run(rows):
        _write_player_week(tmp_path, rows, monkeypatch)
        model = _SpyModel([42.0])
        _patch_common(monkeypatch, rows, model)
        result = predict.score_as_of_week(2025, target_week=4, config=_config())
        return result, model

    result_truncated, model_truncated = _run(rows_truncated)
    result_with_future, model_with_future = _run(rows_with_future)

    # Both must land on the same trailing snap_pct (mean of weeks 2-3 =
    # 0.5), never the week-4/5 rows that hadn't "happened yet" as of week 4.
    assert model_truncated.seen["trailing_snap_pct_avg"].iloc[0] == pytest.approx(0.5)
    assert model_with_future.seen["trailing_snap_pct_avg"].iloc[0] == pytest.approx(0.5)
    assert result_truncated.loc[0, "predicted_score"] == result_with_future.loc[0, "predicted_score"] == 42.0


def test_flagged_is_true_only_when_margin_clears_the_configured_threshold(tmp_path, monkeypatch):
    rows = _three_games("COLD") + _three_games("HOT")
    _write_player_week(tmp_path, rows, monkeypatch)
    # tail(1) rows come out sorted by player_id: COLD before HOT.
    model = _SpyModel([20.0, 11.0])
    _patch_common(monkeypatch, rows, model, points_value=10.0)  # baseline_score = 10.0 for both

    result = predict.score_as_of_week(2025, target_week=4, config=_config(flag_margin=3.0)).set_index("player_id")

    assert result.loc["COLD", "predicted_score"] == 20.0
    assert bool(result.loc["COLD", "flagged"]) is True  # margin 10 >= 3
    assert result.loc["HOT", "predicted_score"] == 11.0
    assert bool(result.loc["HOT", "flagged"]) is False  # margin 1 < 3


def test_injury_context_is_attached_per_player_not_broadcast(tmp_path, monkeypatch):
    rows = _three_games("P1") + _three_games("P2")
    _write_player_week(tmp_path, rows, monkeypatch)
    _patch_common(monkeypatch, rows, model=_SpyModel([10.0, 10.0]))

    def _ctx(player_week, injuries, player_id, season, week, usage_gap, lookback):
        if player_id == "P1":
            return InjuryContext(True, "Teammate", "Out", "Hamstring", "Usage spike coincides with Teammate (Out).")
        return InjuryContext(False, None, None, None, "")

    monkeypatch.setattr(predict, "get_injury_context", _ctx)

    result = predict.score_as_of_week(2025, target_week=4, config=_config()).set_index("player_id")

    assert bool(result.loc["P1", "has_injury_context"]) is True
    assert "Teammate" in result.loc["P1", "injury_explanation"]
    assert bool(result.loc["P2", "has_injury_context"]) is False
    assert result.loc["P2", "injury_explanation"] == ""


def test_roster_status_note_is_attached_from_flagged_statuses(tmp_path, monkeypatch):
    rows = _three_games("P1")
    _write_player_week(tmp_path, rows, monkeypatch)
    _patch_common(monkeypatch, rows, model=_SpyModel([10.0]))
    monkeypatch.setattr(predict, "get_flagged_roster_statuses", lambda: {"P1": ("SUS", "Suspended")})

    result = predict.score_as_of_week(2025, target_week=4, config=_config())

    assert "Suspended" in result.loc[0, "roster_status_note"]


def test_qb_rows_from_the_separate_model_are_unioned_in(tmp_path, monkeypatch):
    rows = _three_games("WR1", position="WR")
    _write_player_week(tmp_path, rows, monkeypatch)
    _patch_common(monkeypatch, rows, model=_SpyModel([10.0]))

    qb_row = pd.DataFrame([{
        "season": 2025, "week": 3, "player_id": "QB1", "position": "QB", "team": "KC",
        "trailing_points_avg": 18.0, "predicted_score": 25.0, "baseline_score": 18.0, "flagged": True,
        "has_injury_context": False, "injury_explanation": "", "roster_status_note": None,
        "usage_explanation": "Pass attempts up 30 -> 40.",
    }])
    monkeypatch.setattr("edge_engine.model.predict_qb.score_qbs_as_of_week", lambda *a, **k: qb_row)

    result = predict.score_as_of_week(2025, target_week=4, config=_config())

    assert set(result["player_id"]) == {"WR1", "QB1"}
    # QB1's predicted score (25.0) beats WR1's (10.0) -> sorted to the top.
    assert result.iloc[0]["player_id"] == "QB1"


def test_score_latest_week_derives_season_and_next_week_from_ingested_data(tmp_path, monkeypatch):
    rows = [
        _usage_row("P1", 2024, 1),
        _usage_row("P1", 2025, 1),
        _usage_row("P1", 2025, 5),  # the latest season's latest played week
    ]
    _write_player_week(tmp_path, rows, monkeypatch)

    calls = []

    def _stub_score_as_of_week(season, target_week, config):
        calls.append((season, target_week))
        return "SENTINEL"

    monkeypatch.setattr(predict, "score_as_of_week", _stub_score_as_of_week)

    result = predict.score_latest_week(_config())

    assert calls == [(2025, 6)]
    assert result == "SENTINEL"
