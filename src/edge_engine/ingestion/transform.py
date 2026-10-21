"""Join raw nflverse tables into one per-player, per-week usage table.

Schema produced by build_player_week_table():
    season, week, player_id, player_name, position, team, opponent,
    snap_pct, target_share, air_yards_share,
    red_zone_touches, red_zone_target_share,
    fantasy_points_ppr

Scope: regular season only (season_type == "REG"). Postseason is excluded
because nflverse's postseason week numbers overlap regular-season week
numbers, and playoff usage isn't representative of a normal role anyway.

Known gap: true route-participation % is not in this table. It isn't
exposed anywhere in the public nflverse data (checked weekly data, snap
counts, NGS, PFR weekly, and FTN charting) — routes-run-by-non-targeted-
receivers is PFF-proprietary data, not public. snap_pct is the closest
available participation proxy; target_share and air_yards_share carry
most of the signal route participation would otherwise add.
"""

from __future__ import annotations

import pandas as pd

from edge_engine.ingestion import raw

BASE_COLUMNS = [
    "season",
    "week",
    "player_id",
    "player_name",
    "position",
    "team",
    "opponent",
    "target_share",
    "air_yards_share",
    "fantasy_points_ppr",
]

OUTPUT_COLUMNS = BASE_COLUMNS[:-1] + [
    "snap_pct",
    "red_zone_touches",
    "red_zone_target_share",
    "fantasy_points_ppr",
]

# The model only covers these positions (CLAUDE.md: "no usage data exists"
# for K/DST). Filtered here rather than in raw.py because two different
# sources feed into _base_usage() below -- the direct weekly-data fetch
# and pbp_fallback.py's play-by-play reconstruction, which sources
# identity from a separate, unfiltered roster table -- and filtering only
# one of them would let non-skill positions back in through the other.
# Real, not hypothetical: verified against 2021 (via reconstruction) and
# 2023 (via direct fetch), both of which otherwise carry a handful of
# defenders and linemen with a non-null target_share from trick plays and
# fumble-return scoring.
SKILL_POSITIONS = ["QB", "RB", "WR", "TE"]


def _base_usage(season: int, force_refresh: bool) -> pd.DataFrame:
    weekly = raw.fetch_weekly_data_or_reconstruct(season, force_refresh)
    weekly = weekly[weekly["season_type"] == "REG"].copy()
    weekly = weekly[weekly["position"].isin(SKILL_POSITIONS)]
    # nflverse's own weekly table carries both `player_name` (abbreviated)
    # and `player_display_name` (full), plus `recent_team`; the
    # play-by-play reconstruction (pbp_fallback.py) already emits the
    # normalized `player_name`/`team` names. Rename conditionally so
    # either source lands on the same schema instead of the fallback
    # path KeyError-ing on columns it was never going to have.
    if "player_display_name" in weekly.columns:
        weekly = weekly.drop(columns=["player_name"]).rename(columns={"player_display_name": "player_name"})
    if "recent_team" in weekly.columns:
        weekly = weekly.rename(columns={"recent_team": "team"})
    return weekly[
        [
            "season",
            "week",
            "player_id",
            "player_name",
            "position",
            "team",
            "opponent_team",
            "target_share",
            "air_yards_share",
            "fantasy_points_ppr",
        ]
    ].rename(columns={"opponent_team": "opponent"})


def _snap_pct(season: int, force_refresh: bool) -> pd.DataFrame:
    snaps = raw.fetch_snap_counts(season, force_refresh)
    snaps = snaps[snaps["game_type"] == "REG"].copy()
    snaps = (
        snaps.groupby(["pfr_player_id", "season", "week"], as_index=False)["offense_pct"]
        .max()
        .rename(columns={"offense_pct": "snap_pct"})
    )
    ids = raw.fetch_id_crosswalk(force_refresh)
    snaps = snaps.merge(ids, left_on="pfr_player_id", right_on="pfr_id", how="left")
    return snaps[["gsis_id", "season", "week", "snap_pct"]].rename(
        columns={"gsis_id": "player_id"}
    )


def _red_zone_stats(season: int, force_refresh: bool) -> pd.DataFrame:
    pbp = raw.fetch_pbp_data(season, force_refresh)
    pbp = pbp[pbp["season_type"] == "REG"].copy()
    rz = pbp[(pbp["yardline_100"].notna()) & (pbp["yardline_100"] <= 20)]

    rz_rush = rz[rz["rush_attempt"] == 1][["season", "week", "rusher_player_id"]].rename(
        columns={"rusher_player_id": "player_id"}
    )
    rz_pass = rz[rz["pass_attempt"] == 1][
        ["season", "week", "posteam", "receiver_player_id"]
    ].rename(columns={"receiver_player_id": "player_id"})

    rz_rush = rz_rush.dropna(subset=["player_id"])
    rz_pass_targets = rz_pass.dropna(subset=["player_id"])

    touches = pd.concat(
        [rz_rush[["season", "week", "player_id"]], rz_pass_targets[["season", "week", "player_id"]]]
    )
    red_zone_touches = (
        touches.groupby(["season", "week", "player_id"])
        .size()
        .reset_index(name="red_zone_touches")
    )

    player_rz_targets = (
        rz_pass_targets.groupby(["season", "week", "player_id"])
        .size()
        .reset_index(name="player_rz_targets")
    )

    team_rz_targets = (
        rz_pass_targets.groupby(["season", "week", "posteam"])
        .size()
        .reset_index(name="team_rz_targets")
        .rename(columns={"posteam": "team"})
    )

    red_zone_touches["red_zone_touches"] = red_zone_touches["red_zone_touches"].astype(int)
    return red_zone_touches, player_rz_targets, team_rz_targets


def attach_red_zone_stats(
    df: pd.DataFrame,
    red_zone_touches: pd.DataFrame,
    player_rz_targets: pd.DataFrame,
    team_rz_targets: pd.DataFrame,
) -> pd.DataFrame:
    """Merge red-zone stats onto a base (season, week, player_id, team) frame.

    A player who appeared in base usage but had zero red-zone touches (or
    targets) that week genuinely had zero — they played, we know the true
    count. That's different from red_zone_target_share: it stays null only
    when the player's team had no red-zone pass attempts at all that week,
    since the share is genuinely undefined then, not zero.
    """
    df = df.merge(red_zone_touches, on=["season", "week", "player_id"], how="left")
    df = df.merge(player_rz_targets, on=["season", "week", "player_id"], how="left")
    df = df.merge(team_rz_targets, on=["season", "week", "team"], how="left")

    df["red_zone_touches"] = df["red_zone_touches"].fillna(0).astype(int)
    df["player_rz_targets"] = df["player_rz_targets"].fillna(0)
    df["red_zone_target_share"] = df["player_rz_targets"] / df["team_rz_targets"]
    return df.drop(columns=["player_rz_targets", "team_rz_targets"])


def build_player_week_table(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """Build the normalized per-player, per-week usage table for one season."""
    base = _base_usage(season, force_refresh)
    snap_pct = _snap_pct(season, force_refresh)
    red_zone_touches, player_rz_targets, team_rz_targets = _red_zone_stats(season, force_refresh)

    df = base.merge(snap_pct, on=["season", "week", "player_id"], how="left")
    df = attach_red_zone_stats(df, red_zone_touches, player_rz_targets, team_rz_targets)

    return df[OUTPUT_COLUMNS].sort_values(["season", "week", "player_id"]).reset_index(drop=True)
