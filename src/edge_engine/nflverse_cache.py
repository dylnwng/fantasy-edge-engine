"""Shared caching helper for nfl_data_py pulls. Used by both the usage-data
ingestion pipeline and the roster-state reference lookups (bye weeks, team
abbreviations, player ID resolution) so neither has to re-implement it.
"""

from __future__ import annotations

import pandas as pd

from edge_engine.paths import RAW_DIR


def cached_fetch(name: str, fetch_fn, force_refresh: bool = False) -> pd.DataFrame:
    """Cache a fetch under data/raw/<name>.parquet. `name` should already
    encode any key that makes the fetch unique (e.g. "weekly_data_2023").

    fetch_fn() failures get consistent context here rather than
    propagating whatever the underlying library happened to raise. A QA
    pass found real inconsistency: nfl_data_py itself raises a clear
    `ValueError: Data not available before 1999.` for a too-old season,
    but a season nflverse simply hasn't published yet (too new, or a
    typo) surfaces as a raw, unwrapped `HTTPError: 404 Not Found` several
    stack frames down in pandas' own parquet-over-HTTP reader -- no
    mention of which fetch failed or why. Wrapping (with `from e`, so the
    original message and traceback are still visible) makes every
    fetch failure equally diagnosable, whichever case it is."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}.parquet"
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)
    try:
        df = fetch_fn()
    except Exception as e:
        raise RuntimeError(
            f"Could not fetch {name!r} from nflverse: {e}. This usually means the "
            "season/args passed in don't exist or haven't been published yet."
        ) from e
    df.to_parquet(path, index=False)
    return df
