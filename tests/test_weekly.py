from edge_engine.weekly import (
    refresh_current_season,
    report_flag_history,
    run_matchup_simulator_if_live,
)


def test_refresh_current_season_swallows_failure_and_reports_it(monkeypatch, capsys):
    def _raise(seasons, force_refresh):
        raise FileNotFoundError("nflverse has no data for 2026 yet")

    monkeypatch.setattr("edge_engine.weekly.ingest_seasons", _raise)

    refresh_current_season(2026)  # must not raise

    out = capsys.readouterr().out
    assert "Could not refresh 2026 data" in out
    assert "Continuing with whatever data is already ingested" in out


def test_refresh_current_season_force_refreshes_only_the_given_season(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "edge_engine.weekly.ingest_seasons",
        lambda seasons, force_refresh: calls.append((seasons, force_refresh)),
    )

    refresh_current_season(2026)

    assert calls == [([2026], True)]


class _NoMatchupSource:
    """Shaped like ManualRosterStateSource -- no get_current_matchup at all."""


def test_matchup_simulator_skipped_when_source_has_no_matchup_support(capsys):
    run_matchup_simulator_if_live(_NoMatchupSource())

    out = capsys.readouterr().out
    assert "Skipping matchup simulator" in out
    assert "EDGE_ENGINE_ROSTER_SOURCE=espn" in out


class _EspnLikeSource:
    def get_current_matchup(self, week=None):
        raise NotImplementedError  # never actually called in this test


def test_matchup_simulator_skipped_when_no_live_matchup_exists_yet(monkeypatch, capsys):
    def _raise_no_matchup():
        raise ValueError("No two-team matchup found for team_id=1 in week 0")

    monkeypatch.setattr("edge_engine.simulation.matchup_cli.main", _raise_no_matchup)

    run_matchup_simulator_if_live(_EspnLikeSource())

    out = capsys.readouterr().out
    assert "Skipping matchup simulator: No two-team matchup found" in out


def test_matchup_simulator_skipped_on_unanticipated_library_error(monkeypatch, capsys):
    # Confirmed against a real live offseason run (week 0): espn_api's own
    # box_scores() can raise a raw KeyError deep inside its own parsing
    # rather than one of our recognized exception types -- the catch here
    # must be broad enough to treat that the same as "no matchup yet",
    # not let it crash the whole weekly run with a bare traceback.
    def _raise_library_internal_error():
        raise KeyError("rosterForCurrentScoringPeriod")

    monkeypatch.setattr("edge_engine.simulation.matchup_cli.main", _raise_library_internal_error)

    run_matchup_simulator_if_live(_EspnLikeSource())  # must not raise

    out = capsys.readouterr().out
    assert "Skipping matchup simulator" in out


# ---- flag history step ----

def _history_config(train_seasons=(2023, 2024), validation_season=2025):
    from edge_engine.model.config import ModelConfig

    return ModelConfig(
        train_seasons=list(train_seasons), validation_season=validation_season,
        trailing_window=2, flag_margin=3.0,
        injury_ahead_usage_gap=0.15, injury_lookback_weeks=2,
    )


def test_flag_history_skipped_with_a_clear_message_when_untrained(monkeypatch, capsys):
    # A fresh checkout that hasn't run `train` must not lose its rankings
    # to a missing model file.
    monkeypatch.setattr("edge_engine.model.history.model_available", lambda: False)

    report_flag_history(2026)

    out = capsys.readouterr().out
    assert "Skipping flag history" in out
    assert "model.train" in out


def test_flag_history_swallows_a_tracking_failure(monkeypatch, capsys):
    monkeypatch.setattr("edge_engine.model.history.model_available", lambda: True)
    monkeypatch.setattr("edge_engine.weekly.load_model_config", lambda: _history_config())

    def _boom(seasons, config):
        raise ValueError("no ingested data for 2026")

    monkeypatch.setattr("edge_engine.model.history.track_flagged_players", _boom)

    report_flag_history(2026)  # must not raise

    assert "Skipping flag history: no ingested data for 2026" in capsys.readouterr().out


def test_flag_history_says_so_when_nothing_is_gradeable_yet(monkeypatch, capsys):
    import pandas as pd

    monkeypatch.setattr("edge_engine.model.history.model_available", lambda: True)
    monkeypatch.setattr("edge_engine.weekly.load_model_config", lambda: _history_config())
    monkeypatch.setattr("edge_engine.model.history.track_flagged_players", lambda s, c: pd.DataFrame())

    report_flag_history(2026)

    out = capsys.readouterr().out
    assert "No flags with a following game to grade against yet" in out


def test_flag_history_labels_the_live_season_as_fully_out_of_sample(monkeypatch, capsys):
    import pandas as pd

    monkeypatch.setattr("edge_engine.model.history.model_available", lambda: True)
    monkeypatch.setattr("edge_engine.weekly.load_model_config", lambda: _history_config())
    tracked = pd.DataFrame([
        {"beat_baseline": True, "following_avg_points": 15.0, "baseline_score": 10.0},
    ])
    monkeypatch.setattr("edge_engine.model.history.track_flagged_players", lambda s, c: tracked)

    report_flag_history(2026)  # neither trained on nor validated against

    out = capsys.readouterr().out
    assert "live" in out
    assert "100.0%" in out
    assert "+5.00" in out


def test_flag_history_warns_when_the_season_was_trained_on(monkeypatch, capsys):
    # Grading a training season is the model marking its own homework;
    # the number must not be presented as evidence.
    import pandas as pd

    monkeypatch.setattr("edge_engine.model.history.model_available", lambda: True)
    monkeypatch.setattr("edge_engine.weekly.load_model_config", lambda: _history_config())
    tracked = pd.DataFrame([
        {"beat_baseline": True, "following_avg_points": 15.0, "baseline_score": 10.0},
    ])
    monkeypatch.setattr("edge_engine.model.history.track_flagged_players", lambda s, c: tracked)

    report_flag_history(2024)  # in train_seasons

    out = capsys.readouterr().out
    assert "in-sample" in out
    assert "not evidence" in out.lower() or "not as evidence" in out.lower()
