"""Cached fetches from nfl_data_py.

Each function pulls one raw nflverse table and caches it to
data/raw/<name>.parquet so re-running the pipeline for an already-fetched
season doesn't hit the network again. Pass force_refresh=True to bypass
the cache (e.g. to pick up corrections nflverse pushed for a week that
already looked "final").
"""

from __future__ import annotations

import warnings

import pandas as pd

from edge_engine.nflverse_cache import cached_fetch

warnings.filterwarnings("ignore", module="nfl_data_py")


def fetch_weekly_data(season: int, force_refresh: bool = False) -> pd.DataFrame:
    import nfl_data_py as nfl

    return cached_fetch(
        f"weekly_data_{season}",
        lambda: nfl.import_weekly_data([season], downcast=True),
        force_refresh,
    )


def fetch_snap_counts(season: int, force_refresh: bool = False) -> pd.DataFrame:
    import nfl_data_py as nfl

    return cached_fetch(
        f"snap_counts_{season}", lambda: nfl.import_snap_counts([season]), force_refresh
    )


def fetch_pbp_data(season: int, force_refresh: bool = False) -> pd.DataFrame:
    import nfl_data_py as nfl

    return cached_fetch(
        f"pbp_data_{season}",
        lambda: nfl.import_pbp_data([season], downcast=True),
        force_refresh,
    )


def fetch_injuries(season: int, force_refresh: bool = False) -> pd.DataFrame:
    import nfl_data_py as nfl

    return cached_fetch(
        f"injuries_{season}", lambda: nfl.import_injuries([season]), force_refresh
    )


def fetch_id_crosswalk(force_refresh: bool = False) -> pd.DataFrame:
    """gsis_id <-> pfr_id crosswalk. Not season-scoped, cached under a fixed key."""
    import nfl_data_py as nfl

    return cached_fetch(
        "id_crosswalk",
        lambda: nfl.import_ids()[["gsis_id", "pfr_id"]].dropna().drop_duplicates(),
        force_refresh,
    )
