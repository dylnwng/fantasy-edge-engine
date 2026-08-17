"""Cached nflverse reference data used to resolve manually-entered roster
rows into real player IDs and bye weeks: the player ID table, team
abbreviation/alias table, and season schedule.
"""

from __future__ import annotations

import nflreadpy as nfr
import pandas as pd

from edge_engine.nflverse_cache import cached_fetch


def fetch_players(force_refresh: bool = False) -> pd.DataFrame:
    return cached_fetch("players", lambda: nfr.load_players().to_pandas(), force_refresh)


def fetch_team_desc(force_refresh: bool = False) -> pd.DataFrame:
    return cached_fetch("team_desc", lambda: nfr.load_teams().to_pandas(), force_refresh)


def fetch_schedules(season: int, force_refresh: bool = False) -> pd.DataFrame:
    return cached_fetch(
        f"schedules_{season}", lambda: nfr.load_schedules(season).to_pandas(), force_refresh
    )
