"""Trailing usage-trend features and next-week fantasy-point labels.

Directly implements the PRD's documented limitation: a player needs
`window` weeks of trailing usage data before a trend exists, so the
earliest weeks of a season produce no feature row at all (not a
degraded/zero-filled one) — by design, not by omission. With the default
window=2, that means no signal until week 3, matching "most useful from
Week 3 onward" in the PRD.

Trailing windows are computed over each player's own sequence of played
games (grouped by player_id, season), not raw calendar weeks — a bye
week simply isn't a row, so it's skipped rather than treated as a zero
week. The window never crosses a season boundary.
"""

from __future__ import annotations

import pandas as pd

USAGE_COLUMNS = [
    "snap_pct",
    "target_share",
    "air_yards_share",
    "red_zone_touches",
    "red_zone_target_share",
]

ID_COLUMNS = ["season", "week", "player_id", "position", "team"]


def build_features(
    player_week: pd.DataFrame,
    points: pd.Series,
    window: int = 2,
    usage_columns: list[str] | None = None,
) -> pd.DataFrame:
    """player_week: requirement 2's usage table for one or more seasons.
    points: fantasy points per row of player_week (same index/order),
    computed via scoring.compute_fantasy_points() under league settings.
    window: trailing window size in games (PRD calls for 2-3 weeks).

    Returns one row per (season, week, player_id) with trailing_* feature
    columns and `label_next_week_points` (NaN for a player's last game of
    a season, or any row without enough trailing history — drop those
    before training; prediction-time callers keep them since there's no
    future label to check yet for the most recent week)."""
    usage_columns = list(USAGE_COLUMNS if usage_columns is None else usage_columns)

    df = player_week.copy()
    df["points"] = points.to_numpy()
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    # A duplicate (player_id, season, week) row -- e.g. an upstream join
    # fan-out -- gets silently treated as a second real game by the
    # rolling-window logic below: it fabricates a trailing average from
    # the duplicate, and label_next_week_points ends up pointing at the
    # duplicate itself rather than the real next game. A QA pass
    # confirmed this concretely corrupts training data with no error.
    # Fail loudly here instead, at the one place every caller passes
    # through.
    dupes = df.duplicated(subset=["player_id", "season", "week"], keep=False)
    if dupes.any():
        sample = df.loc[dupes, ["player_id", "season", "week"]].drop_duplicates().head(5)
        raise ValueError(
            f"player_week has {int(dupes.sum())} row(s) sharing a duplicate "
            "(player_id, season, week) key -- build_features requires this to be "
            f"unique. Example duplicated keys:\n{sample.to_string(index=False)}"
        )

    grouped = df.groupby(["player_id", "season"], group_keys=False)

    for col in usage_columns:
        df[f"trailing_{col}_avg"] = grouped[col].transform(
            lambda s: s.rolling(window, min_periods=window).mean()
        )
        df[f"trailing_{col}_trend"] = grouped[col].transform(lambda s: s - s.shift(window))

    df["trailing_points_avg"] = grouped["points"].transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )
    df["label_next_week_points"] = grouped["points"].transform(lambda s: s.shift(-1))

    feature_cols = [c for c in df.columns if c.startswith("trailing_")]
    return df[[*ID_COLUMNS, *feature_cols, "label_next_week_points"]]


def feature_columns(window: int = 2, usage_columns: list[str] | None = None) -> list[str]:
    """The model's input feature column names, for use at both train and
    predict time so they never drift apart.

    `usage_columns` exists so an experiment can trail a DIFFERENT set of
    per-week columns through the identical windowing logic -- season
    boundaries, played-games-not-calendar-weeks, the duplicate-key guard --
    rather than reimplementing it and getting one of those subtleties
    wrong. Defaulted, so the shipped feature set is unchanged."""
    cols = list(USAGE_COLUMNS if usage_columns is None else usage_columns)
    out = [f"trailing_{c}_avg" for c in cols]
    out += [f"trailing_{c}_trend" for c in cols]
    out.append("trailing_points_avg")
    return out
