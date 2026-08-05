"""Local Parquet file store with upsert semantics.

A single-user weekly batch job doesn't justify a hosted database — a
Parquet file keyed by (season, week, player_id) is enough, as long as
re-ingesting an already-loaded week replaces those rows instead of
duplicating them.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

KEY_COLUMNS = ["season", "week", "player_id"]


def upsert(df: pd.DataFrame, path: Path, key_columns: list[str] = KEY_COLUMNS) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        existing = pd.read_parquet(path)
        new_keys = df[key_columns].drop_duplicates()
        merged = existing.merge(new_keys, on=key_columns, how="left", indicator=True)
        existing_kept = existing[merged["_merge"].to_numpy() == "left_only"]
        combined = pd.concat([existing_kept, df], ignore_index=True)
    else:
        combined = df.copy()

    combined = combined.sort_values(key_columns).reset_index(drop=True)
    combined.to_parquet(path, index=False)
    return combined
