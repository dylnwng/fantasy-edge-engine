"""Accuracy experiment: opponent-adjusted ("defense vs. position" / DvP)
feature -- how many fantasy points a player's upcoming opponent has
recently allowed to that position, on top of the opportunity model's
existing player-own-usage-trend features.

Deliberately standalone: features.py's build_features() never imports
this, so it carries zero risk to the already-validated opportunity model
until (and unless) train_with_dvp.py's experiment proves it earns a
place in the default feature set -- see that module and EVALUATION.md
for the honest go/no-go decision.

The leakage trap this module exists to avoid: player_week's own
`opponent` column is who a player *already played* -- a historical
fact, not a schedule. build_features()'s label_next_week_points is the
player's *next played* game (skips byes, via shift(-1) within each
player's own game sequence, grouped by player_id/season). A DvP feature
must resolve to the opponent of that *same* next game, or it'll quietly
describe a different week than the one being predicted -- and at true
prediction time there's no future row to read an opponent from at all.
That's why this needs a real schedule (nflverse's own, via
roster.nflverse_ref.fetch_schedules), not player_week's `opponent`.
"""

from __future__ import annotations

import pandas as pd

from edge_engine.roster.nflverse_ref import fetch_schedules

# Positions the opportunity model actually has usage data for -- K/D-ST
# have essentially zero player-level usage rows in nflverse (see
# EVALUATION.md), so there's no "defense vs K" signal worth building
# from this pipeline either.
DVP_POSITIONS = ["QB", "RB", "WR", "TE"]


def _unpivot_schedule(sched: pd.DataFrame) -> pd.DataFrame:
    """(season, week, team, opponent), REG season only, unpivoted from a
    single season's raw schedule (home_team/away_team columns) so both
    sides of every matchup get a row. Pulled out as a pure function,
    separate from the network-fetching wrapper below, so it's directly
    unit-testable against a synthetic schedule frame -- same pattern as
    espn_source.py's _resolve_game_final/_select_my_side."""
    sched = sched[sched["game_type"] == "REG"]
    home = sched[["season", "week", "home_team", "away_team"]].rename(
        columns={"home_team": "team", "away_team": "opponent"}
    )
    away = sched[["season", "week", "home_team", "away_team"]].rename(
        columns={"away_team": "team", "home_team": "opponent"}
    )
    return pd.concat([home, away], ignore_index=True)


def build_schedule_opponent_table(seasons: list[int]) -> pd.DataFrame:
    """(season, week, team, opponent) across all `seasons`. This is
    schedule *truth* -- unlike player_week's `opponent` column, it's
    available for a week that hasn't been played yet, which is exactly
    what prediction-time DvP lookups need."""
    frames = [_unpivot_schedule(fetch_schedules(season)) for season in seasons]
    return pd.concat(frames, ignore_index=True).reset_index(drop=True)


def build_points_allowed_table(player_week: pd.DataFrame, points: pd.Series, window: int = 8) -> pd.DataFrame:
    """(season, week, team, position, trailing_points_allowed_avg): a
    trailing average of how many fantasy points TEAM's defense has
    allowed to POSITION, over the team's own trailing `window` played
    games. Same "as of this row, inclusive" convention as variance.py's
    compute_trailing_points_std -- the caller is responsible for only
    looking up a row strictly before the game being predicted, exactly
    like variance.py's own callers are (see compute_dvp_feature)."""
    df = player_week[["season", "week", "player_id", "position", "opponent"]].copy()
    df["points"] = points.to_numpy()
    df = df[df["position"].isin(DVP_POSITIONS)]

    weekly_allowed = (
        df.groupby(["season", "week", "opponent", "position"], as_index=False)["points"]
        .sum()
        .rename(columns={"opponent": "team", "points": "points_allowed_this_week"})
    )
    weekly_allowed = weekly_allowed.sort_values(["team", "position", "season", "week"])
    grouped = weekly_allowed.groupby(["team", "position", "season"], group_keys=False)["points_allowed_this_week"]
    weekly_allowed["trailing_points_allowed_avg"] = grouped.transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )
    return weekly_allowed[["season", "week", "team", "position", "trailing_points_allowed_avg"]]


def compute_dvp_feature(
    merged_pre_features: pd.DataFrame,
    schedule_table: pd.DataFrame,
    points_allowed_table: pd.DataFrame,
    *,
    target_week: int | None = None,
) -> pd.DataFrame:
    """(season, week, player_id, dvp_points_allowed_avg) -- merge onto
    build_features()'s output via (season, week, player_id).

    `merged_pre_features` is the frame train.py/predict.py already have
    right before calling build_features() (needs season, week,
    player_id, position, team -- the player's OWN team, to look up their
    next opponent from the schedule).

    target_week=None (training mode): resolves each row's *next played
    game*'s opponent, mirroring build_features()'s own
    label_next_week_points exactly -- so the DvP feature and the label
    always refer to the same game, bye weeks included.

    target_week=<int> (prediction mode): every row's next game is simply
    target_week (already known to the caller -- there's no "next row" to
    look ahead to at prediction time)."""
    df = merged_pre_features[["season", "week", "player_id", "position", "team"]].copy()

    if target_week is None:
        df = df.sort_values(["player_id", "season", "week"])
        df["next_week"] = df.groupby(["player_id", "season"], group_keys=False)["week"].shift(-1)
    else:
        df["next_week"] = float(target_week)

    df = df.dropna(subset=["next_week"]).copy()
    # nflverse's various tables downcast week/season to different int
    # widths (int32 vs int64) depending on the source -- merge_asof
    # requires an exact dtype match on its "on"/"by" keys, so normalize
    # everything going into it rather than let that be a silent trap.
    df["next_week"] = df["next_week"].astype("int64")
    df["season"] = df["season"].astype("int64")

    sched = schedule_table.rename(columns={"week": "next_week"}).copy()
    sched["next_week"] = sched["next_week"].astype("int64")
    sched["season"] = sched["season"].astype("int64")
    df = df.merge(sched[["season", "next_week", "team", "opponent"]], on=["season", "next_week", "team"], how="left")
    df = df.dropna(subset=["opponent"])

    # The DvP number must reflect the opponent's defense as of BEFORE
    # that next game -- an as-of (backward) join to the most recent
    # points_allowed row strictly earlier than next_week, never an
    # exact match on next_week itself (that would leak the very
    # outcome the model is trying to predict).
    # Renamed to "allowed_as_of_week" (not "week") specifically to avoid
    # colliding with df's own already-present "week" column -- that
    # collision previously left neither column named plain "week" after
    # the merge, breaking the final column selection below.
    allowed = points_allowed_table.rename(columns={"team": "opponent", "week": "allowed_as_of_week"}).copy()
    allowed["allowed_as_of_week"] = allowed["allowed_as_of_week"].astype("int64")
    allowed["season"] = allowed["season"].astype("int64")
    allowed = allowed.sort_values("allowed_as_of_week")
    df = df.sort_values("next_week")
    df = pd.merge_asof(
        df,
        allowed,
        left_on="next_week",
        right_on="allowed_as_of_week",
        by=["season", "opponent", "position"],
        direction="backward",
        allow_exact_matches=False,
    )

    df = df.rename(columns={"trailing_points_allowed_avg": "dvp_points_allowed_avg"})
    return df[["season", "week", "player_id", "dvp_points_allowed_avg"]]
