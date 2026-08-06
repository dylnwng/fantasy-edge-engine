import pytest

from edge_engine.simulation import matchup_cli


class _RaisesLibraryInternalError:
    def get_current_matchup(self, week=None):
        # Confirmed against a real live offseason run: espn_api's own
        # box_scores() can raise a raw KeyError deep inside its own
        # parsing (not the ValueError get_current_matchup() itself
        # raises for the case it recognizes) when asked about a week
        # with no real matchup data yet.
        raise KeyError("rosterForCurrentScoringPeriod")

    def get_league_config(self):
        raise AssertionError("should never be reached -- get_current_matchup fails first")


def test_main_raises_a_plain_exception_not_systemexit(monkeypatch):
    # main() itself must raise a normal exception (not SystemExit, which
    # inherits from BaseException) -- weekly.py imports and calls main()
    # directly, catching with a plain `except Exception` to skip just
    # this step. A first version of this fix raised SystemExit here
    # directly, which escaped that catch and killed the whole weekly.py
    # run instead of just skipping the matchup section -- caught by
    # re-running weekly.py end to end against the real live league.
    monkeypatch.setattr(matchup_cli, "load_simulation_config", lambda: None)
    monkeypatch.setattr(matchup_cli, "get_matchup_source", lambda: _RaisesLibraryInternalError())
    monkeypatch.setattr("sys.argv", ["matchup_cli.py"])

    with pytest.raises(RuntimeError, match="rosterForCurrentScoringPeriod"):
        matchup_cli.main()


def test_direct_invocation_exits_cleanly_instead_of_raw_traceback(monkeypatch):
    # The __main__ boundary is where main()'s plain exception becomes a
    # clean process exit, for direct `python -m ...` invocation.
    monkeypatch.setattr(matchup_cli, "load_simulation_config", lambda: None)
    monkeypatch.setattr(matchup_cli, "get_matchup_source", lambda: _RaisesLibraryInternalError())
    monkeypatch.setattr("sys.argv", ["matchup_cli.py"])

    with pytest.raises(SystemExit, match="rosterForCurrentScoringPeriod"):
        try:
            matchup_cli.main()
        except Exception as e:
            raise SystemExit(str(e)) from e
