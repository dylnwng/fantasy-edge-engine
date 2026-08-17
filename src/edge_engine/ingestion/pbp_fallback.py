"""Rebuild the per-player weekly stat lines that `nflreadpy.
load_player_stats()` normally provides, from raw play-by-play instead.

Why this exists: nflverse publishes its pre-aggregated `player_stats_<year>`
table on a lag. For the 2025 season, `load_player_stats(2025)` returned
nothing while `load_pbp(2025)` returned a complete season (48,771 plays,
weeks 1-20) and `load_snap_counts(2025)` returned 26,612 rows. So the
underlying facts are all there; only the convenience aggregation is
missing. Without this module the opportunity model simply cannot see a
season that has already been played.

This is a reconstruction, so it is gated on reproducing a season where
nflverse's own numbers DO exist -- see validate_against_weekly_data()
and scripts/validate_pbp_fallback.py. Deriving stats and trusting them
because the code ran would be exactly the kind of unverified claim this
project's evaluation refuses to make elsewhere.

Scope: only the columns transform.py's _base_usage() actually consumes.
This is deliberately not a general-purpose reimplementation of
nflverse's aggregation -- every extra derived column is another thing
that can be subtly wrong.
"""

from __future__ import annotations

import pandas as pd

from edge_engine.ingestion.raw import fetch_pbp_data

# nflverse's own standard-scoring weights (nflfastR's fantasy_points
# definition). fantasy_points_ppr is this plus one point per reception.
_PASS_YARDS_PER_POINT = 25.0
_RUSH_REC_YARDS_PER_POINT = 10.0
_PASS_TD_POINTS = 4.0
_RUSH_REC_TD_POINTS = 6.0
_INTERCEPTION_POINTS = -2.0
_FUMBLE_LOST_POINTS = -2.0
_TWO_POINT_CONVERSION_POINTS = 2.0

OUTPUT_COLUMNS = [
    "season",
    "week",
    "player_id",
    "player_name",
    "position",
    "team",
    "opponent_team",
    "target_share",
    "air_yards_share",
    "fantasy_points",
    "fantasy_points_ppr",
    "season_type",
    # Raw per-occurrence counts scoring.py's bonus support reads. Not
    # needed for this league (its bonuses dict is empty) but carried so
    # a league WITH TD/INT bonuses gets real numbers instead of
    # scoring.py's "unsupported bonus" warning. Fumble bonuses are the
    # one gap: nflverse splits those three ways
    # (receiving/rushing/sack_fumbles_lost) and this reconstruction only
    # recovers the total, so a fumble bonus would warn rather than
    # silently mis-score.
    "passing_tds",
    "rushing_tds",
    "receiving_tds",
    "interceptions",
]


def _sum_by(pbp: pd.DataFrame, id_col: str, value_col: str, out_name: str) -> pd.DataFrame:
    """Sum `value_col` per (season, week, player), keyed off whichever
    role column identifies the player on that play (passer/rusher/
    receiver)."""
    df = pbp[pbp[id_col].notna()]
    out = (
        df.groupby(["season", "week", id_col], as_index=False)[value_col]
        .sum()
        .rename(columns={id_col: "player_id", value_col: out_name})
    )
    return out


def _player_team_map(pbp: pd.DataFrame) -> pd.DataFrame:
    """(season, week, player_id) -> team, opponent_team.

    A player's team on a play is the offense (posteam) whenever they're
    the passer/rusher/receiver, which is the only role this module
    aggregates. Takes the first observation per player-week rather than
    the modal one: a player has exactly one posteam within a given week
    (they play one game for one team), so the two agree, and a per-group
    mode() over ~30k groups is slow enough to dominate the whole
    reconstruction's runtime for no accuracy gain."""
    frames = []
    for id_col in ("passer_player_id", "rusher_player_id", "receiver_player_id"):
        part = pbp[pbp[id_col].notna()][["season", "week", id_col, "posteam", "defteam"]]
        frames.append(part.rename(columns={id_col: "player_id"}))
    stacked = pd.concat(frames, ignore_index=True)

    return (
        stacked.dropna(subset=["posteam"])
        .drop_duplicates(subset=["season", "week", "player_id"], keep="first")
        .rename(columns={"posteam": "team", "defteam": "opponent_team"})
        .reset_index(drop=True)
    )


def build_weekly_from_pbp(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """A per-player, per-week frame shaped like the subset of
    load_player_stats() that transform.py consumes, derived from
    play-by-play."""
    pbp = fetch_pbp_data(season, force_refresh)
    # nflreadpy raises ValueError for a season past its own
    # get_current_season() boundary (the Thursday after Labor Day) --
    # that surfaces through cached_fetch as a RuntimeError before this
    # function is even reached, with an equally clear message. This guard
    # covers the remaining case: an empty frame for a season inside that
    # range nflverse genuinely has no plays for yet, which would otherwise
    # die on a bare KeyError: 'season_type' several lines down.
    if pbp.empty or "season_type" not in pbp.columns:
        raise RuntimeError(
            f"nflverse has no play-by-play for {season} either, so weekly stats can't be "
            "reconstructed for it. This is expected for a season that hasn't been played yet."
        )
    pbp = pbp[pbp["season_type"] == "REG"].copy()

    # --- passing ---
    passing_yards = _sum_by(pbp, "passer_player_id", "passing_yards", "passing_yards")
    passing_tds = _sum_by(pbp, "passer_player_id", "pass_touchdown", "passing_tds")
    interceptions = _sum_by(pbp, "passer_player_id", "interception", "interceptions")

    # --- rushing ---
    rushing_yards = _sum_by(pbp, "rusher_player_id", "rushing_yards", "rushing_yards")
    rushing_tds = _sum_by(pbp, "rusher_player_id", "rush_touchdown", "rushing_tds")

    # --- receiving ---
    # A pass TD credits the passer with a passing TD and the receiver
    # with a receiving TD -- same play, two different stat lines.
    receiving_yards = _sum_by(pbp, "receiver_player_id", "receiving_yards", "receiving_yards")
    receiving_tds = _sum_by(pbp, "receiver_player_id", "pass_touchdown", "receiving_tds")
    receptions = _sum_by(pbp, "receiver_player_id", "complete_pass", "receptions")
    targets = _sum_by(pbp, "receiver_player_id", "pass_attempt", "targets")
    air_yards = _sum_by(pbp, "receiver_player_id", "air_yards", "air_yards")

    # --- turnovers ---
    fumbles = pbp[pbp["fumble_lost"] == 1]
    fumbles_lost = (
        fumbles.groupby(["season", "week", "fumbled_1_player_id"], as_index=False)["fumble_lost"]
        .sum()
        .rename(columns={"fumbled_1_player_id": "player_id", "fumble_lost": "fumbles_lost"})
    )

    # --- two-point conversions ---
    # Credited to whoever physically scored it (rusher or receiver);
    # small effect, included for fidelity rather than left as a silent
    # systematic undercount.
    two_pt = pbp[pbp["two_point_conv_result"] == "success"]
    two_pt_frames = []
    for id_col in ("rusher_player_id", "receiver_player_id"):
        part = two_pt[two_pt[id_col].notna()][["season", "week", id_col]].copy()
        part["two_point_conversions"] = 1
        two_pt_frames.append(part.rename(columns={id_col: "player_id"}))
    two_point = (
        pd.concat(two_pt_frames, ignore_index=True)
        .groupby(["season", "week", "player_id"], as_index=False)["two_point_conversions"]
        .sum()
        if two_pt_frames
        else pd.DataFrame(columns=["season", "week", "player_id", "two_point_conversions"])
    )

    stat_frames = [
        passing_yards, passing_tds, interceptions,
        rushing_yards, rushing_tds,
        receiving_yards, receiving_tds, receptions, targets, air_yards,
        fumbles_lost, two_point,
    ]
    weekly = _player_team_map(pbp)
    for frame in stat_frames:
        weekly = weekly.merge(frame, on=["season", "week", "player_id"], how="outer")

    # A player who simply didn't do a thing genuinely recorded zero of
    # it -- distinct from the null-means-unknown convention used for
    # shares below, where the denominator can be legitimately absent.
    stat_cols = [
        "passing_yards", "passing_tds", "interceptions", "rushing_yards", "rushing_tds",
        "receiving_yards", "receiving_tds", "receptions", "targets", "air_yards",
        "fumbles_lost", "two_point_conversions",
    ]
    for col in stat_cols:
        if col not in weekly.columns:
            weekly[col] = 0.0
    weekly[stat_cols] = weekly[stat_cols].fillna(0.0)

    weekly["fantasy_points"] = (
        weekly["passing_yards"] / _PASS_YARDS_PER_POINT
        + weekly["passing_tds"] * _PASS_TD_POINTS
        + weekly["interceptions"] * _INTERCEPTION_POINTS
        + weekly["rushing_yards"] / _RUSH_REC_YARDS_PER_POINT
        + weekly["rushing_tds"] * _RUSH_REC_TD_POINTS
        + weekly["receiving_yards"] / _RUSH_REC_YARDS_PER_POINT
        + weekly["receiving_tds"] * _RUSH_REC_TD_POINTS
        + weekly["fumbles_lost"] * _FUMBLE_LOST_POINTS
        + weekly["two_point_conversions"] * _TWO_POINT_CONVERSION_POINTS
    )
    weekly["fantasy_points_ppr"] = weekly["fantasy_points"] + weekly["receptions"]

    weekly = _attach_shares(weekly)
    weekly = _attach_identity(weekly, season)
    weekly["season_type"] = "REG"
    return weekly[OUTPUT_COLUMNS].sort_values(["season", "week", "player_id"]).reset_index(drop=True)


def _attach_shares(weekly: pd.DataFrame) -> pd.DataFrame:
    """target_share / air_yards_share against the player's own team's
    weekly totals. Left null (not zero) when the team denominator is
    zero, since the share is genuinely undefined then -- matching
    transform.py's treatment of red_zone_target_share."""
    team_totals = (
        weekly.groupby(["season", "week", "team"], as_index=False)[["targets", "air_yards"]]
        .sum()
        .rename(columns={"targets": "team_targets", "air_yards": "team_air_yards"})
    )
    weekly = weekly.merge(team_totals, on=["season", "week", "team"], how="left")

    # A player with no targets at all has an UNDEFINED receiving share,
    # not a zero one, and nflverse agrees: checked against real 2024
    # data, it never emits a 0.0 target_share -- only NaN or positive
    # (659 of 664 QB rows are NaN, as are 350 RB and 35 WR rows).
    #
    # This matters far beyond cosmetics. features.py drops rows with
    # null features, which is the mechanism that keeps quarterbacks out
    # of a model whose receiving-usage features carry no signal for them
    # (see EVALUATION.md's QB scope gap). Emitting 0.0 here silently
    # reversed that documented boundary: 659 reconstructed QB rows
    # survived the completeness check and got scored on meaningless
    # features.
    had_targets = weekly["targets"] > 0
    weekly["target_share"] = (weekly["targets"] / weekly["team_targets"]).where(
        had_targets & (weekly["team_targets"] > 0)
    )
    weekly["air_yards_share"] = (weekly["air_yards"] / weekly["team_air_yards"]).where(
        had_targets & (weekly["team_air_yards"] > 0)
    )
    return weekly


def _attach_identity(weekly: pd.DataFrame, season: int) -> pd.DataFrame:
    """Player name and position, which play-by-play doesn't carry
    reliably. load_rosters_weekly() is week-indexed and available for the
    current season, so a mid-season position change is reflected rather
    than flattened to a season-long guess."""
    import nflreadpy as nfr

    rosters = nfr.load_rosters_weekly(season).to_pandas()
    rosters = rosters.rename(columns={"gsis_id": "player_id", "full_name": "player_name"})
    rosters = rosters[["season", "week", "player_id", "player_name", "position"]].drop_duplicates(
        subset=["season", "week", "player_id"]
    )
    return weekly.merge(rosters, on=["season", "week", "player_id"], how="left")


def validate_against_weekly_data(season: int) -> dict:
    """Compare this module's reconstruction against nflverse's own
    weekly_data for a season where both exist. This is the gate that
    decides whether the 2025 reconstruction is trustworthy -- run it on
    a season nflverse HAS published, since 2025 by definition can't be
    checked directly."""
    from edge_engine.ingestion.raw import fetch_weekly_data

    derived = build_weekly_from_pbp(season)
    official = fetch_weekly_data(season)
    official = official[official["season_type"] == "REG"]

    merged = official[["season", "week", "player_id", "fantasy_points_ppr", "target_share", "air_yards_share"]].merge(
        derived[["season", "week", "player_id", "fantasy_points_ppr", "target_share", "air_yards_share"]],
        on=["season", "week", "player_id"],
        how="inner",
        suffixes=("_official", "_derived"),
    )

    results = {"season": season, "n_matched_rows": int(len(merged)), "n_official_rows": int(len(official))}
    for col in ("fantasy_points_ppr", "target_share", "air_yards_share"):
        both = merged[[f"{col}_official", f"{col}_derived"]].dropna()
        if both.empty:
            continue
        diff = (both[f"{col}_derived"] - both[f"{col}_official"]).abs()
        results[col] = {
            "correlation": float(both[f"{col}_derived"].corr(both[f"{col}_official"])),
            "mean_abs_diff": float(diff.mean()),
            "median_abs_diff": float(diff.median()),
            "pct_within_0.5": float((diff <= 0.5).mean()),
            "max_abs_diff": float(diff.max()),
        }
    return results
