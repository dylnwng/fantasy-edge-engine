"""Per-player point variance — the storage the original PRD asked Phase 1
to keep even though Phase 1 itself never used it, so a later Monte Carlo
matchup simulator (Phase 2) wouldn't need to re-architect the data layer
to get it.

Deliberately a longer trailing window than the opportunity model's own
2-game trend window (features.py): a standard deviation from 2 points is
nearly meaningless, dominated by noise rather than reflecting a real
spread. 6 games balances "enough samples for a std that isn't just
noise" against "recent enough to reflect the player's current role" — a
player who just took over a starting job shouldn't have his prior
committee-back-era games diluting his spread estimate forever.
"""

from __future__ import annotations

import pandas as pd

VARIANCE_WINDOW = 6
MIN_GAMES_FOR_VARIANCE = 3  # below this, a stdev is a near-meaningless estimate


def compute_trailing_points_std(
    points_table: pd.DataFrame,  # (season, week, player_id, points) -- e.g. scoring.compute_points_for_seasons
    window: int = VARIANCE_WINDOW,
    min_games: int = MIN_GAMES_FOR_VARIANCE,
) -> pd.DataFrame:
    """One row per (season, week, player_id): trailing_points_std, the
    sample standard deviation (ddof=1) of the player's own actual points
    over their trailing `window` played games, up to and including this
    row's week -- same "as of this row" convention as features.py's
    trailing_points_avg (not shifted), so the two line up when both are
    looked up for the same "latest played week" row.

    NaN below `min_games` -- never silently backfilled with 0 or a
    league average; callers decide their own fallback."""
    df = points_table.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    grouped = df.groupby(["player_id", "season"], group_keys=False)["points"]
    df["trailing_points_std"] = grouped.transform(
        lambda s: s.rolling(window, min_periods=min_games).std(ddof=1)
    )
    return df[["season", "week", "player_id", "trailing_points_std"]]
