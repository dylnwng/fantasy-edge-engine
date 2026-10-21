import numpy as np
import pandas as pd
import pytest

from edge_engine.model import history
from edge_engine.model.config import ModelConfig
from edge_engine.model.history import (
    HELD_OUT,
    IN_SAMPLE,
    LIVE,
    _attach_following_performance,
    season_provenance,
    summarize,
    track_flagged_players,
)
from edge_engine.roster.models import LeagueConfig, ScoringSettings, WaiverConfig


def _config(train_seasons=(2023, 2024), validation_season=2025, flag_margin=3.0):
    return ModelConfig(
        train_seasons=list(train_seasons),
        validation_season=validation_season,
        trailing_window=2,
        flag_margin=flag_margin,
        injury_ahead_usage_gap=0.15,
        injury_lookback_weeks=2,
    )


# ---- provenance: the difference between evidence and homework ----

def test_a_trained_on_season_is_flagged_as_in_sample():
    assert season_provenance(2024, _config()) == IN_SAMPLE


def test_the_validation_season_is_held_out_not_in_sample():
    assert season_provenance(2025, _config()) == HELD_OUT


def test_the_current_season_is_live_when_the_model_never_saw_it():
    # The normal weekly case: train through 2024, validate 2025, play
    # 2026. That last one is the only fully clean read.
    assert season_provenance(2026, _config()) == LIVE


def test_every_provenance_has_a_note_explaining_what_it_means():
    for provenance in (IN_SAMPLE, HELD_OUT, LIVE):
        assert history.provenance_note(provenance)


# ---- grading the following games ----

def _points(rows):
    return pd.DataFrame(rows, columns=["season", "week", "player_id", "points"])


def test_following_window_averages_the_next_three_games():
    points = _points([
        (2025, 1, "P1", 10.0),
        (2025, 2, "P1", 12.0),
        (2025, 3, "P1", 18.0),
        (2025, 4, "P1", 24.0),
    ])
    flagged = pd.DataFrame([{"season": 2025, "week": 1, "player_id": "P1"}])

    out = _attach_following_performance(flagged, points, window=3)

    assert out.loc[0, "following_games_played"] == 3
    assert out.loc[0, "following_avg_points"] == pytest.approx((12.0 + 18.0 + 24.0) / 3)


def test_a_partial_window_at_season_end_is_graded_on_what_exists():
    # Flagged in the second-to-last game: only one following game to
    # judge against. Grade it on that rather than discarding it or
    # averaging in zeros for games that were never played.
    points = _points([(2025, 16, "P1", 10.0), (2025, 17, "P1", 20.0)])
    flagged = pd.DataFrame([{"season": 2025, "week": 16, "player_id": "P1"}])

    out = _attach_following_performance(flagged, points, window=3)

    assert out.loc[0, "following_games_played"] == 1
    assert out.loc[0, "following_avg_points"] == pytest.approx(20.0)


def test_the_following_window_does_not_cross_a_season_boundary():
    points = _points([
        (2024, 17, "P1", 5.0),
        (2025, 1, "P1", 30.0),
        (2025, 2, "P1", 30.0),
    ])
    flagged = pd.DataFrame([{"season": 2024, "week": 17, "player_id": "P1"}])

    out = _attach_following_performance(flagged, points, window=3)

    # 2024's last game has no following games *within 2024* -- next
    # season's 30-point weeks must not be credited to it.
    assert out.loc[0, "following_games_played"] == 0


def test_the_window_counts_played_games_not_calendar_weeks():
    # P1 misses week 2 entirely (bye or inactive). The next *played*
    # game is week 3, which is what should be graded -- consistent with
    # how trailing features treat a missing week.
    points = _points([(2025, 1, "P1", 10.0), (2025, 3, "P1", 20.0)])
    flagged = pd.DataFrame([{"season": 2025, "week": 1, "player_id": "P1"}])

    out = _attach_following_performance(flagged, points, window=3)

    assert out.loc[0, "following_games_played"] == 1
    assert out.loc[0, "following_avg_points"] == pytest.approx(20.0)


def test_one_players_following_games_are_not_credited_to_another():
    points = _points([
        (2025, 1, "P1", 10.0), (2025, 2, "P1", 11.0),
        (2025, 1, "P2", 10.0), (2025, 2, "P2", 99.0),
    ])
    flagged = pd.DataFrame([{"season": 2025, "week": 1, "player_id": "P1"}])

    out = _attach_following_performance(flagged, points, window=3)

    assert out.loc[0, "following_avg_points"] == pytest.approx(11.0)


# ---- summarising ----

def test_summarize_of_nothing_reports_zero_not_a_fabricated_rate():
    summary = summarize(pd.DataFrame())

    assert summary["n_flagged"] == 0
    assert summary["hit_rate_3wk"] is None
    assert summary["avg_points_above_baseline"] is None


def test_summarize_computes_hit_rate_and_margin():
    tracked = pd.DataFrame([
        {"beat_baseline": True, "following_avg_points": 15.0, "baseline_score": 10.0},
        {"beat_baseline": False, "following_avg_points": 8.0, "baseline_score": 10.0},
    ])

    summary = summarize(tracked)

    assert summary["n_flagged"] == 2
    assert summary["hit_rate_3wk"] == pytest.approx(0.5)
    assert summary["avg_points_above_baseline"] == pytest.approx(1.5)  # (+5 and -2) / 2


# ---- end-to-end flag tracking ----

class _StubSource:
    def get_league_config(self):
        return LeagueConfig(
            season=2026,
            scoring=ScoringSettings(ppr_type="full_ppr"),
            lineup_slots={"QB": 1, "RB": 2, "WR": 2, "TE": 1, "FLEX": 1},
            waivers=WaiverConfig(system="FAAB", season_budget=100, min_bid=1, clear_day="Wednesday"),
        )


class _StubModel:
    def __init__(self, prediction):
        self.prediction = prediction

    def predict(self, X):
        return np.full(len(X), float(self.prediction))


def _usage_row(player_id, season, week, **overrides):
    row = dict(
        season=season, week=week, player_id=player_id, position="RB", team="GB",
        snap_pct=0.5, target_share=0.2, air_yards_share=0.2,
        red_zone_touches=1, red_zone_target_share=0.2,
    )
    row.update(overrides)
    return row


def _patch_tracking(monkeypatch, usage_rows, points_rows, prediction):
    monkeypatch.setattr(history, "_load_model", lambda: _StubModel(prediction))
    monkeypatch.setattr(history, "_ensure_seasons_ingested", lambda seasons: pd.DataFrame(usage_rows))
    monkeypatch.setattr(history, "get_default_source", lambda: _StubSource())
    monkeypatch.setattr(history, "compute_points_for_seasons", lambda seasons, scoring: _points(points_rows))


def test_only_players_clearing_the_margin_are_tracked(monkeypatch):
    usage = [_usage_row("P1", 2026, w) for w in (1, 2, 3, 4)]
    points = [(2026, w, "P1", 10.0) for w in (1, 2, 3, 4)]
    # Baseline is the trailing points average (10.0); predicting 10.5 is a
    # margin of 0.5, nowhere near the flag_margin of 3.0.
    _patch_tracking(monkeypatch, usage, points, prediction=10.5)

    tracked = track_flagged_players([2026], _config(flag_margin=3.0))

    assert tracked.empty


def test_a_flagged_player_is_graded_against_his_own_baseline(monkeypatch):
    usage = [_usage_row("P1", 2026, w) for w in (1, 2, 3, 4)]
    # Baseline settles at 10.0; the following games are much better.
    points = [(2026, 1, "P1", 10.0), (2026, 2, "P1", 10.0), (2026, 3, "P1", 30.0), (2026, 4, "P1", 30.0)]
    _patch_tracking(monkeypatch, usage, points, prediction=50.0)  # clears any margin

    tracked = track_flagged_players([2026], _config(flag_margin=3.0))

    assert not tracked.empty
    assert tracked["beat_baseline"].all()
    assert set(tracked.columns) >= {"season", "week", "player_id", "baseline_score", "beat_baseline"}


def test_flags_with_no_following_game_are_excluded_not_counted_as_misses(monkeypatch):
    # A flag on the final game of a season has nothing to grade against.
    # Counting it as a miss would understate the model with data that
    # doesn't exist -- the same "never fabricate from missing data"
    # discipline as bye_weeks.py.
    usage = [_usage_row("P1", 2026, w) for w in (1, 2, 3)]
    points = [(2026, w, "P1", 10.0) for w in (1, 2, 3)]
    _patch_tracking(monkeypatch, usage, points, prediction=50.0)

    tracked = track_flagged_players([2026], _config(flag_margin=3.0))

    # Whatever is tracked must have had something to grade it on.
    assert (tracked["following_games_played"] > 0).all()
    assert 3 not in set(tracked["week"])  # the last week has no following game
