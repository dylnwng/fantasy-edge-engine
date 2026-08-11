from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from edge_engine.model import predict_qb
from edge_engine.model.config import ModelConfig
from edge_engine.model.injury_context import InjuryContext
from edge_engine.roster.models import LeagueConfig, ScoringSettings, WaiverConfig

INJURY_COLUMNS = ["season", "week", "team", "gsis_id", "full_name", "position", "report_status", "report_primary_injury"]


def _config(flag_margin=3.0):
    return ModelConfig(
        train_seasons=[2023], validation_season=2024, trailing_window=2, flag_margin=flag_margin,
        injury_ahead_usage_gap=0.15, injury_lookback_weeks=2,
    )


class _StubSource:
    def get_league_config(self):
        return LeagueConfig(
            season=2025,
            scoring=ScoringSettings(ppr_type="full_ppr"),
            lineup_slots={"QB": 1},
            waivers=WaiverConfig(system="FAAB", season_budget=100, min_bid=1, clear_day="Wednesday"),
        )


class _FakeRegressor:
    def __init__(self, scores):
        self._scores = scores

    def load_model(self, path):
        pass

    def predict(self, X):
        return np.asarray(self._scores, dtype=float)


def _volume_row(player_id, week, attempts=25.0, pass_share=0.8, rush_attempts=2.0,
                 team_attempts=30.0, air_yards=200.0, season=2025, team="KC"):
    return dict(
        season=season, week=week, player_id=player_id, team=team,
        attempts=attempts, pass_share=pass_share, rush_attempts=rush_attempts,
        team_attempts=team_attempts, air_yards=air_yards,
    )


def _three_games(player_id, **overrides):
    # window=2's trend feature is `value - value.shift(2)`, non-null only
    # from a player's THIRD played game onward -- score_qbs_as_of_week
    # drops any row missing a feature, so every fixture needs 3 games.
    return [_volume_row(player_id, w, **overrides) for w in (1, 2, 3)]


def _points_for(volume_rows, value=15.0):
    return pd.DataFrame([
        {"season": r["season"], "week": r["week"], "player_id": r["player_id"], "points": value}
        for r in volume_rows
    ])


def _patch_common(monkeypatch, volume_rows, scores):
    monkeypatch.setattr(predict_qb, "qb_model_available", lambda: True)
    monkeypatch.setattr(predict_qb, "build_qb_volume", lambda season: pd.DataFrame(volume_rows))
    monkeypatch.setattr(predict_qb, "get_default_source", lambda: _StubSource())
    monkeypatch.setattr(predict_qb, "compute_points_for_seasons", lambda seasons, scoring: _points_for(volume_rows))
    monkeypatch.setattr(predict_qb, "xgb", SimpleNamespace(XGBRegressor=lambda: _FakeRegressor(scores)))
    monkeypatch.setattr(predict_qb, "load_injury_reports", lambda seasons, force_refresh=False: pd.DataFrame(columns=INJURY_COLUMNS))
    monkeypatch.setattr(predict_qb, "get_injury_context", lambda *a, **k: InjuryContext(False, None, None, None, ""))
    monkeypatch.setattr(predict_qb, "get_flagged_roster_statuses", lambda: {})


def test_no_trained_qb_artifact_is_a_silent_no_op(monkeypatch):
    # Documented contract: absent models/qb_model.json, QB scoring must
    # degrade to the pre-QB behaviour, not crash.
    monkeypatch.setattr(predict_qb, "qb_model_available", lambda: False)

    result = predict_qb.score_qbs_as_of_week(2025, target_week=4, config=_config())

    assert result.empty
    assert list(result.columns) == predict_qb.OUTPUT_COLUMNS


def test_season_with_no_play_by_play_returns_empty_not_a_crash(monkeypatch):
    monkeypatch.setattr(predict_qb, "qb_model_available", lambda: True)

    def _raise(season):
        raise RuntimeError(f"nflverse has no play-by-play for {season}")

    monkeypatch.setattr(predict_qb, "build_qb_volume", _raise)

    result = predict_qb.score_qbs_as_of_week(2099, target_week=4, config=_config())

    assert result.empty
    assert list(result.columns) == predict_qb.OUTPUT_COLUMNS


def test_only_weeks_strictly_before_target_week_are_used(monkeypatch):
    volume_rows = _three_games("QB1") + [_volume_row("QB1", 4, attempts=99.0)]
    _patch_common(monkeypatch, volume_rows, scores=[30.0])

    result = predict_qb.score_qbs_as_of_week(2025, target_week=4, config=_config(), player_week=None)

    assert list(result["week"]) == [3]  # week 4 excluded even though it's in the fixture


def test_predicted_baseline_and_flagged_are_wired_through(monkeypatch):
    volume_rows = _three_games("QB1")
    _patch_common(monkeypatch, volume_rows, scores=[25.0])

    result = predict_qb.score_qbs_as_of_week(2025, target_week=4, config=_config(flag_margin=3.0), player_week=None)

    row = result.iloc[0]
    assert row["predicted_score"] == 25.0
    assert row["baseline_score"] == pytest.approx(15.0)  # trailing_points_avg from the stubbed points
    assert bool(row["flagged"]) is True  # margin 10 >= 3


def test_injury_context_is_skipped_without_a_player_week_frame(monkeypatch):
    volume_rows = _three_games("QB1")
    _patch_common(monkeypatch, volume_rows, scores=[20.0])
    called = []
    monkeypatch.setattr(
        predict_qb, "get_injury_context",
        lambda *a, **k: called.append(1) or InjuryContext(True, None, None, None, "x"),
    )

    result = predict_qb.score_qbs_as_of_week(2025, target_week=4, config=_config(), player_week=None)

    assert called == []
    assert bool(result.iloc[0]["has_injury_context"]) is False
    assert result.iloc[0]["injury_explanation"] == ""


def test_injury_context_is_attached_when_player_week_is_provided(monkeypatch):
    volume_rows = _three_games("QB1")
    _patch_common(monkeypatch, volume_rows, scores=[20.0])
    monkeypatch.setattr(
        predict_qb, "get_injury_context",
        lambda *a, **k: InjuryContext(True, "Backup", "Out", None, "Starter is out."),
    )
    player_week = pd.DataFrame([{"season": 2025, "week": 3, "player_id": "QB1", "position": "QB", "team": "KC"}])

    result = predict_qb.score_qbs_as_of_week(2025, target_week=4, config=_config(), player_week=player_week)

    assert bool(result.iloc[0]["has_injury_context"]) is True
    assert result.iloc[0]["injury_explanation"] == "Starter is out."


def test_roster_status_note_and_usage_explanation_are_attached(monkeypatch):
    volume_rows = _three_games("QB1")
    _patch_common(monkeypatch, volume_rows, scores=[20.0])
    monkeypatch.setattr(predict_qb, "get_flagged_roster_statuses", lambda: {"QB1": ("SUS", "Suspended")})

    result = predict_qb.score_qbs_as_of_week(2025, target_week=4, config=_config(), player_week=None)

    assert "Suspended" in result.iloc[0]["roster_status_note"]
    assert result.iloc[0]["usage_explanation"]  # non-empty: the real qb_usage_explanation ran
