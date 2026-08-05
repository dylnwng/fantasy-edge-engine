"""Shared caching helper for nfl_data_py pulls. Used by both the usage-data
ingestion pipeline and the roster-state reference lookups (bye weeks, team
abbreviations, player ID resolution) so neither has to re-implement it.
"""

from __future__ import annotations

import pandas as pd

from edge_engine.paths import RAW_DIR


def cached_fetch(name: str, fetch_fn, force_refresh: bool = False) -> pd.DataFrame:
    """Cache a fetch under data/raw/<name>.parquet. `name` should already
    encode any key that makes the fetch unique (e.g. "weekly_data_2023")."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"{name}.parquet"
    if path.exists() and not force_refresh:
        return pd.read_parquet(path)
    df = fetch_fn()
    df.to_parquet(path, index=False)
    return df
