"""Accuracy experiment: does *how* a player's usage rose matter, not
just how much?

features.py's trend feature is an endpoint delta -- `s - s.shift(window)`
-- so it cannot distinguish a one-week spike from a sustained climb.
Both of these report exactly the same +40 trend at window=2:

    Player A: 20% -> 20% -> 60%   (one fluky game)
    Player B: 20% -> 40% -> 60%   (a real, progressive role change)

Those are very different bets. A is the classic waiver trap -- a garbage-
time or single-blowout usage spike that reverts the following week; B is
the durable role change the whole tool exists to catch early. This module
adds features that describe the *shape* of the climb, which is exactly
the precision-over-recall lever the PRD prioritizes (a shorter,
higher-trust list beats a longer one that catches more breakouts).

Deliberately standalone, same discipline as opponent_defense.py:
features.py never imports this, so it carries zero risk to the
already-validated model until train_with_persistence.py's honest A/B
says it earns a place. That experiment reported a real result either
way -- see EVALUATION.md.
"""

from __future__ import annotations

import pandas as pd

# Only the two clearest role-change signals, not all five usage columns.
# Each column here adds 2 features; doing all five would add 10 to a
# model with ~10k usable training rows, which is a real overfitting risk
# for a speculative experiment (the position-split experiment already
# showed this model is sensitive to how it's sliced).
PERSISTENCE_COLUMNS = ["snap_pct", "target_share"]

PERSISTENCE_WINDOW = 4  # games; needs >= 2 to have any week-over-week delta at all


def _rising_streak(s: pd.Series) -> pd.Series:
    """Consecutive week-over-week increases ending at each row.

    A player at 20/40/60 has streak 0,1,2 -- two straight weeks of
    growth. A player at 20/20/60 has streak 0,0,1 -- one. That gap is
    the entire point of this module."""
    rising = s.diff() > 0  # NaN diff on a player's first game -> False, correctly "not rising"
    # Standard consecutive-run trick: (~rising).cumsum() assigns a new
    # group id at every non-rising row, so cumsum within that group
    # counts the current unbroken run of rising rows.
    return rising.groupby((~rising).cumsum()).cumsum()


def _pct_weeks_rising(s: pd.Series, window: int) -> pd.Series:
    """Fraction of the last `window - 1` week-over-week deltas that were
    positive. Distinguishes a steady grind upward (1.0) from a sawtooth
    that happens to end high (0.5).

    A player's first game of a season has no prior game, so its delta is
    genuinely undefined -- kept as NaN rather than counted as "not
    rising", so min_periods below won't emit a value until there are
    `window - 1` *real* deltas to average. Counting that first row as a
    False would quietly understate the earliest reportable value."""
    deltas = s.diff()
    rising = deltas.gt(0).astype(float).where(deltas.notna())
    return rising.rolling(window - 1, min_periods=window - 1).mean()


def persistence_feature_columns(columns: list[str] | None = None) -> list[str]:
    """The feature column names this module produces, for use at both
    train and predict time so they never drift apart -- same contract as
    features.py's feature_columns()."""
    cols = columns or PERSISTENCE_COLUMNS
    return [f"{c}_rising_streak" for c in cols] + [f"{c}_pct_weeks_rising" for c in cols]


def compute_persistence_features(
    player_week: pd.DataFrame, window: int = PERSISTENCE_WINDOW
) -> pd.DataFrame:
    """(season, week, player_id, + persistence feature columns), to merge
    onto build_features()'s output via (season, week, player_id).

    Same "as of this row, inclusive" convention as features.py and
    variance.py -- every value uses only that player's own games up to
    and including this row's week, and never crosses a season boundary
    (the groupby includes season, so a player's week 1 has no streak
    inherited from last December)."""
    if window < 2:
        raise ValueError(f"persistence window must be >= 2 to have any week-over-week delta, got {window}")

    missing = [c for c in PERSISTENCE_COLUMNS if c not in player_week.columns]
    if missing:
        raise ValueError(f"player_week is missing required usage column(s): {missing}")

    df = player_week[["season", "week", "player_id", *PERSISTENCE_COLUMNS]].copy()
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    # Same duplicate-key guard as build_features/compute_trailing_points_std:
    # a duplicated row would be silently counted as another real game and
    # inflate a streak, which is precisely the signal this module sells.
    dupes = df.duplicated(subset=["player_id", "season", "week"], keep=False)
    if dupes.any():
        sample = df.loc[dupes, ["player_id", "season", "week"]].drop_duplicates().head(5)
        raise ValueError(
            f"player_week has {int(dupes.sum())} row(s) sharing a duplicate "
            "(player_id, season, week) key -- compute_persistence_features requires "
            f"this to be unique. Example duplicated keys:\n{sample.to_string(index=False)}"
        )

    grouped = df.groupby(["player_id", "season"], group_keys=False)
    for col in PERSISTENCE_COLUMNS:
        df[f"{col}_rising_streak"] = grouped[col].transform(_rising_streak)
        df[f"{col}_pct_weeks_rising"] = grouped[col].transform(lambda s: _pct_weeks_rising(s, window))

    return df[["season", "week", "player_id", *persistence_feature_columns()]]
