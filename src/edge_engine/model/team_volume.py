"""Team-level volume: the multiplier the model currently can't see.

Every feature the opportunity model uses is a SHARE -- snap share, target
share, air-yards share. A share describes how much of his team's work a
player gets, but not how much work there is. A 25% target share on an
offence running 70 plays a game is a different opportunity from 25% on
one running 55, and the model presently has no way to tell them apart.

This is a structurally different axis from anything already in the
feature set, which matters: the accuracy ideas this project has rejected
(opponent adjustment, usage persistence) were both variants of signals
already present. And there is precedent for the multiplier mattering --
the QB model already carries `team_attempts` alongside `pass_share`, so
volume-times-share is exactly the shape of the one experiment that
replicated.

Not wired into the shipped model, and not written into the ingested usage
table -- changing that schema would force a re-ingest for something
unmeasured. `scripts/compare_team_volume.py` runs it through the
walk-forward protocol; adopting it is a separate decision gated on that
evidence.

Trailing discipline
-------------------
These columns are attached per week and then trailed by build_features
exactly like the usage columns, so a fold only ever sees team volume from
games played BEFORE the week being predicted. Handing the model a team's
contemporaneous pace would leak the very game it is trying to predict.
"""

from __future__ import annotations

import pandas as pd

from edge_engine.ingestion.raw import fetch_pbp_data

# Two columns, deliberately. Team pace and passing volume are the pieces
# that turn a share into an opportunity; adding every derivable team stat
# would spend the sample this project already knows is too small to
# resolve marginal effects.
TEAM_VOLUME_COLUMNS = ["team_plays", "team_pass_attempts"]


def build_team_volume(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """(season, week, team, team_plays, team_pass_attempts) for one season.

    `team_plays` counts pass and rush attempts, which is a plays-run
    proxy rather than a snap count -- it excludes kneels, spikes and
    special teams, none of which represent opportunity for the players
    this model ranks. Counted the same way qb_features.build_qb_volume
    counts, so the two modules agree on what a play is.
    """
    pbp = fetch_pbp_data(season, force_refresh)
    if pbp.empty or "season_type" not in pbp.columns:
        raise RuntimeError(
            f"nflverse has no play-by-play for {season}, so team volume can't be built. "
            "Expected for a season that hasn't been played yet."
        )

    plays = pbp[pbp["season_type"] == "REG"].copy()
    plays = plays[plays["posteam"].notna()]
    plays["pass_attempt"] = plays["pass_attempt"].fillna(0)
    plays["rush_attempt"] = plays["rush_attempt"].fillna(0)

    grouped = plays.groupby(["season", "week", "posteam"], as_index=False).agg(
        team_pass_attempts=("pass_attempt", "sum"),
        team_rush_attempts=("rush_attempt", "sum"),
    )
    grouped["team_plays"] = grouped["team_pass_attempts"] + grouped["team_rush_attempts"]
    return grouped.rename(columns={"posteam": "team"})[
        ["season", "week", "team", *TEAM_VOLUME_COLUMNS]
    ]


def attach_team_volume(player_week: pd.DataFrame, team_volume: pd.DataFrame) -> pd.DataFrame:
    """Merge team volume onto a per-player-week frame by (season, week, team).

    A player whose team has no volume row keeps a null rather than a
    zero. Zero would say the offence ran no plays that week, which is
    never true and would drag a trailing average toward a fabricated
    floor -- the caller drops those rows instead.
    """
    missing = {"season", "week", "team"} - set(player_week.columns)
    if missing:
        raise ValueError(f"player_week is missing {sorted(missing)}")

    merged = player_week.merge(team_volume, on=["season", "week", "team"], how="left")
    if len(merged) != len(player_week):
        # A duplicated (season, week, team) row in team_volume fans the
        # player frame out, silently duplicating player-weeks and
        # corrupting every trailing window downstream -- the failure
        # build_features guards against, caught here at its source.
        raise ValueError(
            f"attaching team volume changed the row count ({len(player_week)} -> {len(merged)}) "
            "-- team_volume has duplicate (season, week, team) keys."
        )
    return merged
