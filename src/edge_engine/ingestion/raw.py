"""Cached fetches from nfl_data_py.

Each function pulls one raw nflverse table and caches it to
data/raw/<name>.parquet so re-running the pipeline for an already-fetched
season doesn't hit the network again. Pass force_refresh=True to bypass
the cache (e.g. to pick up corrections nflverse pushed for a week that
already looked "final").
"""

from __future__ import annotations

import logging
import warnings

import pandas as pd

from edge_engine.nflverse_cache import cached_fetch

warnings.filterwarnings("ignore", module="nfl_data_py")

logger = logging.getLogger(__name__)


def fetch_weekly_data(season: int, force_refresh: bool = False) -> pd.DataFrame:
    import nfl_data_py as nfl

    return cached_fetch(
        f"weekly_data_{season}",
        lambda: nfl.import_weekly_data([season], downcast=True),
        force_refresh,
    )


def fetch_weekly_data_or_reconstruct(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """fetch_weekly_data(), falling back to rebuilding the same stat
    lines from play-by-play when nflverse hasn't published its
    pre-aggregated table for that season yet.

    This is a real, recurring situation rather than a hypothetical:
    nflverse publishes player_stats_<year> on a lag, so for much of a
    live season import_weekly_data() 404s while the play-by-play it's
    derived from is already complete. Without this, the model can't see
    the season currently being played -- the only season a waiver tool
    actually cares about.

    The reconstruction is gated on reproducing seasons where both
    sources exist (0.999 correlation on fantasy points, exact on
    air-yards share -- see pbp_fallback.validate_against_weekly_data),
    so this is a measured substitution, not a hopeful one."""
    try:
        return fetch_weekly_data(season, force_refresh)
    except RuntimeError:
        # Lazy import: pbp_fallback imports fetch_pbp_data from this
        # module, so a top-level import here would be circular.
        from edge_engine.ingestion.pbp_fallback import build_weekly_from_pbp

        logger.warning(
            "nflverse has no pre-aggregated weekly stats for %s yet -- reconstructing them "
            "from play-by-play instead (see ingestion/pbp_fallback.py).",
            season,
        )
        return build_weekly_from_pbp(season, force_refresh)


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
