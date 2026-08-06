import pandas as pd

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

    bye_weeks.get_bye_weeks.cache_clear()
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

    bye_weeks.get_bye_weeks.cache_clear()
    byes = bye_weeks.get_bye_weeks(2024)

    # CCC/DDD have 16 missing weeks (everything but week 9) -- ambiguous,
    # not a real bye -- so they must be omitted rather than assigned one.
    assert "CCC" not in byes
    assert "DDD" not in byes


def test_team_with_full_schedule_has_no_bye_entry(monkeypatch):
    rows = [{"season": 2024, "week": w, "game_type": "REG", "home_team": "AAA", "away_team": "BBB"}
            for w in range(1, 18)]
    monkeypatch.setattr(nflverse_ref, "fetch_schedules", lambda season, force_refresh=False: _sched(rows))

    bye_weeks.get_bye_weeks.cache_clear()
    byes = bye_weeks.get_bye_weeks(2024)

    assert "AAA" not in byes
    assert "BBB" not in byes
