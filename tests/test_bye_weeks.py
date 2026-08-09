import logging

import pandas as pd
import pytest

from edge_engine.roster import bye_weeks, nflverse_ref


def _sched(rows):
    return pd.DataFrame(rows, columns=["season", "week", "game_type", "home_team", "away_team"])


def test_real_bye_week_is_detected(monkeypatch):
    # AAA/BBB play every week except a real, single bye in week 9 (some
    # other team-pair covers week 9 league-wide, like a real NFL bye
    # week rotation) -- exactly the case get_bye_weeks should trust.
    rows = [{"season": 2024, "week": w, "game_type": "REG", "home_team": "AAA", "away_team": "BBB"}
            for w in range(1, 18) if w != 9]
    rows.append({"season": 2024, "week": 9, "game_type": "REG", "home_team": "CCC", "away_team": "DDD"})
    monkeypatch.setattr(nflverse_ref, "fetch_schedules", lambda season, force_refresh=False: _sched(rows))

    byes = bye_weeks.get_bye_weeks(2024)

    assert byes["AAA"] == 9
    assert byes["BBB"] == 9


def test_incomplete_schedule_data_does_not_fabricate_a_bye(monkeypatch):
    # CCC/DDD only appears in 1 game (week 9) -- genuinely incomplete
    # schedule data, not a real bye. This used to get reported as "bye
    # week 1" (min of every other missing week), a fabricated answer.
    rows = [{"season": 2024, "week": w, "game_type": "REG", "home_team": "AAA", "away_team": "BBB"}
            for w in range(1, 18) if w != 9]
    rows.append({"season": 2024, "week": 9, "game_type": "REG", "home_team": "CCC", "away_team": "DDD"})
    monkeypatch.setattr(nflverse_ref, "fetch_schedules", lambda season, force_refresh=False: _sched(rows))

    byes = bye_weeks.get_bye_weeks(2024)

    # CCC/DDD have 16 missing weeks (everything but week 9) -- ambiguous,
    # not a real bye -- so they must be omitted rather than assigned one.
    assert "CCC" not in byes
    assert "DDD" not in byes


def test_team_with_full_schedule_has_no_bye_entry(monkeypatch):
    rows = [{"season": 2024, "week": w, "game_type": "REG", "home_team": "AAA", "away_team": "BBB"}
            for w in range(1, 18)]
    monkeypatch.setattr(nflverse_ref, "fetch_schedules", lambda season, force_refresh=False: _sched(rows))

    byes = bye_weeks.get_bye_weeks(2024)

    assert "AAA" not in byes
    assert "BBB" not in byes


def test_a_schedule_correction_is_picked_up_on_the_next_call(monkeypatch):
    # get_bye_weeks used to be @lru_cache'd, so a long-running process
    # (the Streamlit dashboard) would keep serving a stale answer even
    # after the underlying schedule data changed (e.g. force_refresh
    # pulling a correction). No caching here now -- each call re-reads
    # whatever fetch_schedules currently returns.
    original_rows = [{"season": 2024, "week": w, "game_type": "REG", "home_team": "AAA", "away_team": "BBB"}
                      for w in range(1, 18) if w != 9]
    original_rows.append({"season": 2024, "week": 9, "game_type": "REG", "home_team": "CCC", "away_team": "DDD"})
    monkeypatch.setattr(nflverse_ref, "fetch_schedules", lambda season, force_refresh=False: _sched(original_rows))
    first = bye_weeks.get_bye_weeks(2024)
    assert first["AAA"] == 9

    # A correction moves AAA's bye from week 9 to week 10.
    corrected_rows = [{"season": 2024, "week": w, "game_type": "REG", "home_team": "AAA", "away_team": "BBB"}
                       for w in range(1, 18) if w != 10]
    corrected_rows.append({"season": 2024, "week": 10, "game_type": "REG", "home_team": "CCC", "away_team": "DDD"})
    monkeypatch.setattr(nflverse_ref, "fetch_schedules", lambda season, force_refresh=False: _sched(corrected_rows))
    second = bye_weeks.get_bye_weeks(2024)
    assert second["AAA"] == 10


def test_unfetchable_schedule_degrades_to_no_byes_instead_of_crashing(monkeypatch, caplog):
    # The schedule fetch can fail for reasons unrelated to this code --
    # nflverse or habitatring.com down or rate-limiting, no connectivity,
    # a season not yet published -- and cached_fetch surfaces all of them
    # as RuntimeError. That's the same "we don't know the byes" state as
    # the incomplete-schedule case above and must degrade the same way:
    # it used to propagate and take the whole Streamlit dashboard down
    # with a raw traceback, and blocked every roster read in both roster
    # sources, over context that never moves predicted_score.
    def _raise(season, force_refresh=False):
        raise RuntimeError(
            f"Could not fetch 'schedules_{season}' from nflverse: HTTP Error 403: Forbidden."
        )

    monkeypatch.setattr(nflverse_ref, "fetch_schedules", _raise)

    with caplog.at_level(logging.WARNING):
        byes = bye_weeks.get_bye_weeks(2026)

    assert byes == {}
    # Reported, never silent -- same discipline as weekly.py's handling
    # of the same external blocker.
    assert "2026" in caplog.text
    assert "bye" in caplog.text.lower()


def test_a_genuine_parsing_bug_is_not_swallowed_by_the_fetch_guard(monkeypatch):
    # The guard is scoped to RuntimeError (what cached_fetch raises for a
    # fetch failure) specifically so a real bug below it still surfaces.
    monkeypatch.setattr(
        nflverse_ref, "fetch_schedules",
        lambda season, force_refresh=False: _sched([]).drop(columns=["game_type"]),
    )

    with pytest.raises(KeyError):
        bye_weeks.get_bye_weeks(2024)
