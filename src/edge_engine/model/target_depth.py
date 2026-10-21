"""Average depth of target: what KIND of role a share represents.

`air_yards_share` says how much of a team's downfield passing game runs
through a player. It cannot distinguish a checkdown back from a field
stretcher -- two players can hold similar shares with very different
scoring distributions, because one accumulates his air yards eight yards
at a time and the other twenty.

aDOT (air yards per target) is that missing dimension. It is the most
incremental of the candidate features, being adjacent to a signal already
in the model rather than a new axis, so the prior on it should be
correspondingly weak. It is here to be measured, not assumed --
`scripts/compare_target_depth.py`.

Not adopted, not written into the ingested table.

Why this one can't ride through build_features
----------------------------------------------
Trailing aDOT is a RATIO OF SUMS, not a mean of ratios:

    correct:   sum(air_yards over window) / sum(targets over window)
    wrong:     mean(weekly air_yards / weekly targets)

The second weights a one-target week exactly as heavily as a ten-target
week, so a single deep incompletion in a quiet game can swing a player's
trailing depth more than a full afternoon of work. build_features trails
by mean, which is right for the share columns and wrong here, so the
window is computed in this module instead.

The same construction also handles zero-target weeks correctly and for
free: a week with no targets contributes nothing to either sum, rather
than being either a fabricated 0.0 depth or a NaN that nulls the row.

Undefined is left undefined
---------------------------
A player with no targets at all across the window has no depth of target
-- not a depth of zero. Those rows get NaN, and the caller decides
whether to drop them. Filling 0.0 would tell the model that every
non-receiving back runs the shallowest route tree in football, which is a
statement about missing data rather than about football.
"""

from __future__ import annotations

import pandas as pd

from edge_engine.ingestion.raw import fetch_pbp_data

DEPTH_COLUMNS = ["trailing_adot"]


def build_target_depth(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """(season, week, player_id, targets, air_yards) for one season.

    A "target" here is a pass attempt with both a named receiver AND a
    recorded air-yards value. Numerator and denominator therefore come
    from exactly the same plays -- counting targets whose air yards were
    never charted would silently deflate every aDOT.
    """
    pbp = fetch_pbp_data(season, force_refresh)
    if pbp.empty or "season_type" not in pbp.columns:
        raise RuntimeError(
            f"nflverse has no play-by-play for {season}, so target depth can't be built. "
            "Expected for a season that hasn't been played yet."
        )

    plays = pbp[pbp["season_type"] == "REG"]
    targets = plays[
        (plays["pass_attempt"] == 1)
        & plays["receiver_player_id"].notna()
        & plays["air_yards"].notna()
    ]

    grouped = targets.groupby(
        ["season", "week", "receiver_player_id"], as_index=False
    ).agg(targets=("air_yards", "size"), air_yards=("air_yards", "sum"))
    return grouped.rename(columns={"receiver_player_id": "player_id"})


def trailing_adot(depth: pd.DataFrame, window: int = 2) -> pd.DataFrame:
    """(season, week, player_id, trailing_adot) over the last `window`
    PLAYED games.

    Windows run over a player's own sequence of rows within a season, so
    a bye is skipped rather than counted as a zero-target week, and never
    cross a season boundary -- the same conventions features.py uses.
    Inclusive of the current week, matching how `trailing_*_avg` is built
    there (the label is the FOLLOWING week, so including the current one
    leaks nothing).
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    required = {"season", "week", "player_id", "targets", "air_yards"}
    missing = required - set(depth.columns)
    if missing:
        raise ValueError(f"depth is missing {sorted(missing)}")

    df = depth.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    dupes = df.duplicated(subset=["player_id", "season", "week"], keep=False)
    if dupes.any():
        sample = df.loc[dupes, ["player_id", "season", "week"]].drop_duplicates().head(5)
        raise ValueError(
            f"depth has {int(dupes.sum())} row(s) sharing a duplicate "
            "(player_id, season, week) key -- trailing_adot requires this to be unique. "
            f"Example duplicated keys:\n{sample.to_string(index=False)}"
        )

    grouped = df.groupby(["player_id", "season"], group_keys=False)
    rolled_air = grouped["air_yards"].transform(
        lambda s: s.rolling(window, min_periods=window).sum()
    )
    rolled_targets = grouped["targets"].transform(
        lambda s: s.rolling(window, min_periods=window).sum()
    )

    # Ratio of sums. Zero targets across the whole window leaves the depth
    # genuinely undefined rather than zero.
    df["trailing_adot"] = rolled_air.where(rolled_targets > 0) / rolled_targets.where(
        rolled_targets > 0
    )
    return df[["season", "week", "player_id", "trailing_adot"]]


def attach_target_depth(features: pd.DataFrame, depth: pd.DataFrame) -> pd.DataFrame:
    """Merge trailing aDOT onto a built feature table.

    Left join, and NO fill: a player with no targets in the window keeps
    a null, because "undefined depth" is not "zero depth". The caller
    decides whether to drop those rows -- and the comparison script does,
    for BOTH candidates, so neither is evaluated on a different
    population than the other.
    """
    merged = features.merge(depth, on=["season", "week", "player_id"], how="left")
    if len(merged) != len(features):
        raise ValueError(
            f"attaching target depth changed the row count "
            f"({len(features)} -> {len(merged)}) -- duplicate (season, week, player_id) keys."
        )
    return merged
