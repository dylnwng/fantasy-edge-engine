"""Alternative prediction targets for the opportunity model.

The shipped model predicts NEXT WEEK's fantasy points. But a waiver claim
is not a one-week rental -- you hold the player, and train.py already
reports its hit rate at two horizons (next week, and the following three
games) because three weeks is closer to the decision actually being made.

This module builds the multi-week target so that hypothesis can be
measured. It does not change what ships. Use
`scripts/compare_label_horizon.py` to put it through the walk-forward
protocol; adopting it is a separate decision gated on that evidence,
exactly like every other accuracy change in this project.

The forward window is constructed identically to
history._attach_following_performance -- next up-to-N games actually
PLAYED, never crossing a season boundary -- so the training target and
the thing history.py grades flags against are the same quantity rather
than two things that happen to share a name.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_HORIZON = 3


def forward_window_label(
    points_table: pd.DataFrame, horizon: int = DEFAULT_HORIZON, require_full_window: bool = True
) -> pd.DataFrame:
    """Mean points over each player's next `horizon` PLAYED games.

    `points_table` is (season, week, player_id, points) as
    compute_points_for_seasons returns. Returns the same keys plus
    `label_forward_points` and `forward_games`.

    Windows are taken over a player's own sequence of played games, so a
    bye or an inactive week is skipped rather than counted as a zero --
    the same convention features.py uses for trailing windows. The window
    never crosses a season boundary.

    `require_full_window` decides what happens at the end of a season,
    and the two answers serve different purposes:

      * True (default, for TRAINING): rows without a full window get a
        null label and are dropped. A label averaged over one game and a
        label averaged over three are not the same quantity, and mixing
        them teaches the model that the target's variance changes
        arbitrarily near week 17.
      * False (for GRADING past flags, as history.py does): judge a flag
        on whatever games actually followed, because throwing away a
        real outcome is worse than grading it on a shorter window.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be >= 1, got {horizon}")

    required = {"season", "week", "player_id", "points"}
    missing = required - set(points_table.columns)
    if missing:
        raise ValueError(f"points_table is missing {sorted(missing)}")

    pts = points_table.sort_values(["player_id", "season", "week"]).copy()

    dupes = pts.duplicated(subset=["player_id", "season", "week"], keep=False)
    if dupes.any():
        # Same guard as build_features: a duplicated key silently shifts
        # the forward window onto the duplicate instead of the real next
        # game, fabricating a label from a row that isn't a future game.
        sample = pts.loc[dupes, ["player_id", "season", "week"]].drop_duplicates().head(5)
        raise ValueError(
            f"points_table has {int(dupes.sum())} row(s) sharing a duplicate "
            "(player_id, season, week) key -- forward_window_label requires this to be "
            f"unique. Example duplicated keys:\n{sample.to_string(index=False)}"
        )

    grouped = pts.groupby(["player_id", "season"])["points"]
    shifted = []
    for offset in range(1, horizon + 1):
        col = f"_fwd_{offset}"
        pts[col] = grouped.shift(-offset)
        shifted.append(col)

    pts["forward_games"] = pts[shifted].notna().sum(axis=1)
    pts["label_forward_points"] = pts[shifted].mean(axis=1)

    if require_full_window:
        pts.loc[pts["forward_games"] < horizon, "label_forward_points"] = pd.NA

    return pts[["season", "week", "player_id", "label_forward_points", "forward_games"]].reset_index(
        drop=True
    )


def attach_forward_label(
    features: pd.DataFrame,
    points_table: pd.DataFrame,
    horizon: int = DEFAULT_HORIZON,
    require_full_window: bool = True,
) -> pd.DataFrame:
    """`features` (from build_features) with the multi-week label merged on.

    Left join: rows with no forward window keep their trailing features
    and get a null label, so the caller decides whether to drop them
    rather than having rows disappear silently mid-pipeline.
    """
    labels = forward_window_label(points_table, horizon, require_full_window)
    return features.merge(labels, on=["season", "week", "player_id"], how="left")
