"""Cached nflverse reference data used to resolve manually-entered roster
rows into real player IDs and bye weeks: the player ID table, team
abbreviation/alias table, and season schedule.
"""

from __future__ import annotations

import warnings

import pandas as pd

from edge_engine.nflverse_cache import cached_fetch

warnings.filterwarnings("ignore", module="nfl_data_py")


def fetch_players(force_refresh: bool = False) -> pd.DataFrame:
    import nfl_data_py as nfl

    return cached_fetch("players", lambda: nfl.import_players(), force_refresh)


def fetch_team_desc(force_refresh: bool = False) -> pd.DataFrame:
    import nfl_data_py as nfl

    return cached_fetch("team_desc", lambda: nfl.import_team_desc(), force_refresh)


def fetch_schedules(season: int, force_refresh: bool = False) -> pd.DataFrame:
    import nfl_data_py as nfl

    return cached_fetch(
        f"schedules_{season}", lambda: nfl.import_schedules([season]), force_refresh
    )
