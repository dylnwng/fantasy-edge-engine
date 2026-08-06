from edge_engine.weekly import refresh_current_season, run_matchup_simulator_if_live


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
