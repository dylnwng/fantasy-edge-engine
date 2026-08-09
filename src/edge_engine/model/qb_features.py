"""Quarterback usage features — the position the main model can't see.

The opportunity model's features are receiving-usage metrics (target share,
air-yards share) that are structurally null for a passer, so QBs get dropped
by features.py's null-drop. That was documented for most of this project as
"no usage signal exists at the position," reasoning from the fact that QB
snap share is effectively binary: 78% of QB weeks are >80% snaps, and the
median week-over-week move is 0.0 points against 9-10 for WR/RB/TE.

That inference was wrong, and `scripts/qb_signal_probe.py` measured it.
Snap share isn't the only usage metric. **Attempts, pass share, team pass
volume and rush attempts all move week to week**, and a model built on them
beats a trailing-points baseline by ~10-12% MAE with the confidence
interval clear of zero on two independent validation seasons (2024:
+0.73 [+0.42, +1.04]; 2025: +0.89 [+0.57, +1.23], 100% of resamples each).
It is the only experiment in EVALUATION.md to replicate.

Deliberately standalone from features.py. The WR/RB/TE pipeline is
validated and this must not perturb it -- QBs are unioned in at the
prediction layer, not merged into a shared feature space they don't share.

Fantasy points come from `compute_points_for_seasons`, so QB scoring
respects the league's own settings rather than hardcoding a weight table.
Only the volume features are derived here.
"""

from __future__ import annotations

import pandas as pd

from edge_engine.ingestion.raw import fetch_pbp_data

# Volume metrics that actually move for a quarterback. Snap share is
# excluded on purpose: it is ~binary at the position and would contribute
# noise dressed as a feature.
QB_VOLUME_COLUMNS = ["attempts", "rush_attempts", "air_yards", "pass_share", "team_attempts"]

# A quarterback who threw fewer than this in a week was mopping up or got
# hurt early; that isn't a fantasy-relevant starter week and including it
# teaches the model from garbage.
MIN_ATTEMPTS = 10

DEFAULT_WINDOW = 2


def build_qb_volume(season: int, force_refresh: bool = False) -> pd.DataFrame:
    """(season, week, player_id, team, + volume columns) for every real
    starting quarterback week in `season`."""
    pbp = fetch_pbp_data(season, force_refresh)
    if pbp.empty or "season_type" not in pbp.columns:
        raise RuntimeError(
            f"nflverse has no play-by-play for {season}, so QB volume features can't be "
            "built. Expected for a season that hasn't been played yet."
        )
    pbp = pbp[pbp["season_type"] == "REG"]

    passing = (
        pbp[pbp["passer_player_id"].notna()]
        .groupby(["season", "week", "passer_player_id"], as_index=False)
        .agg(
            attempts=("pass_attempt", "sum"),
            air_yards=("air_yards", "sum"),
            team=("posteam", "first"),
        )
        .rename(columns={"passer_player_id": "player_id"})
    )
    rushing = (
        pbp[pbp["rusher_player_id"].notna()]
        .groupby(["season", "week", "rusher_player_id"], as_index=False)
        .agg(rush_attempts=("rush_attempt", "sum"))
        .rename(columns={"rusher_player_id": "player_id"})
    )
    df = passing.merge(rushing, on=["season", "week", "player_id"], how="left")
    df["rush_attempts"] = df["rush_attempts"].fillna(0.0)

    team_pass = (
        pbp[pbp["pass_attempt"] == 1]
        .groupby(["season", "week", "posteam"], as_index=False)["pass_attempt"]
        .sum()
        .rename(columns={"posteam": "team", "pass_attempt": "team_attempts"})
    )
    df = df.merge(team_pass, on=["season", "week", "team"], how="left")

    # Share of his own offense's passing, which separates a true every-down
    # starter from a committee or an injury-shortened afternoon.
    df["pass_share"] = (df["attempts"] / df["team_attempts"]).where(df["team_attempts"] > 0)

    return df[df["attempts"] >= MIN_ATTEMPTS].reset_index(drop=True)


def qb_feature_columns(window: int = DEFAULT_WINDOW) -> list[str]:
    """Feature names, used at both train and predict time so the two can
    never drift apart -- same contract as features.py's feature_columns()."""
    return (
        [f"trailing_{c}_avg" for c in QB_VOLUME_COLUMNS]
        + [f"trailing_{c}_trend" for c in QB_VOLUME_COLUMNS]
        + ["trailing_points_avg"]
    )


def build_qb_features(
    volume: pd.DataFrame, points: pd.DataFrame, window: int = DEFAULT_WINDOW
) -> pd.DataFrame:
    """Trailing volume features plus `label_next_week_points`.

    `points` is (season, week, player_id, points) from
    compute_points_for_seasons -- league-scoring-aware, so a league with
    6-point passing touchdowns trains on its own numbers.

    Same conventions as features.py: windows are over a player's own
    played games (a bye is simply an absent row), never cross a season
    boundary, and the label is the next played game."""
    if window < 1:
        raise ValueError(f"QB feature window must be >= 1, got {window}")

    df = volume.merge(points, on=["season", "week", "player_id"], how="inner")
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)

    dupes = df.duplicated(subset=["player_id", "season", "week"], keep=False)
    if dupes.any():
        sample = df.loc[dupes, ["player_id", "season", "week"]].drop_duplicates().head(5)
        raise ValueError(
            f"QB volume has {int(dupes.sum())} row(s) sharing a duplicate "
            "(player_id, season, week) key -- build_qb_features requires this to be "
            f"unique. Example duplicated keys:\n{sample.to_string(index=False)}"
        )

    grouped = df.groupby(["player_id", "season"], group_keys=False)
    for col in QB_VOLUME_COLUMNS:
        df[f"trailing_{col}_avg"] = grouped[col].transform(
            lambda s: s.rolling(window, min_periods=window).mean()
        )
        df[f"trailing_{col}_trend"] = grouped[col].transform(lambda s: s - s.shift(window))

    df["trailing_points_avg"] = grouped["points"].transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )
    df["label_next_week_points"] = grouped["points"].transform(lambda s: s.shift(-1))
    df["position"] = "QB"

    keep = ["season", "week", "player_id", "position", "team", "points"]
    return df[[*keep, *qb_feature_columns(window), "label_next_week_points"]]


# (column, human label, formatter) for the explanation. Ordered by how
# much a manager should care: dropback volume first, then how much of the
# offense he owns, then legs.
_EXPLAIN_METRICS = [
    ("attempts", "pass attempts", lambda v: f"{v:.0f}"),
    ("pass_share", "share of team pass attempts", lambda v: f"{v:.0%}"),
    ("rush_attempts", "rush attempts", lambda v: f"{v:.0f}"),
    ("team_attempts", "team pass volume", lambda v: f"{v:.0f}"),
]

# Below these, a move is noise rather than a story.
_MIN_MOVE = {"attempts": 4.0, "pass_share": 0.10, "rush_attempts": 2.0, "team_attempts": 5.0}


def qb_usage_explanation(
    volume: pd.DataFrame, player_id: str, season: int, week: int, window: int = DEFAULT_WINDOW
) -> str:
    """Which QB volume metric actually moved, in a sentence.

    The main model's usage_trend_explanation reads snap share and target
    share, which are meaningless for a passer -- a quarterback scored well
    and explained as "usage steady, no single metric moved meaningfully"
    is a black box wearing an explanation's clothes. Requirement 3 asks
    for the specific metric that moved; for a QB that's volume.

    Reads the player's ACTUAL weekly values, exactly as usage_trend.py
    does. It deliberately does not use the trailing_*_avg / trailing_*_trend
    features: the avg is a rolling mean while the trend is a raw
    single-game delta, so subtracting one from the other produces a
    "before" number the player never posted (an early version of this
    reported "pass attempts 50 → 20", which was arithmetic, not history)."""
    rows = volume[
        (volume["season"] == season)
        & (volume["player_id"] == player_id)
        & (volume["week"] <= week)
        & (volume["week"] > week - window - 1)
    ].sort_values("week")

    if len(rows) < 2:
        return "Not enough games yet to show a volume trend."

    span = int(rows["week"].iloc[-1] - rows["week"].iloc[0])
    best = None
    for col, label, fmt in _EXPLAIN_METRICS:
        if col not in rows.columns:
            continue
        series = rows[col]
        if series.isna().any():
            continue
        first, last = float(series.iloc[0]), float(series.iloc[-1])
        move = last - first
        if abs(move) < _MIN_MOVE[col]:
            continue
        # Rank by how far past its own noise floor each metric moved, so
        # metrics on different scales compete fairly.
        score = abs(move) / _MIN_MOVE[col]
        if best is None or score > best[0]:
            best = (score, label, fmt, first, last, move)

    plural = "s" if span != 1 else ""
    if best is None:
        return f"Volume steady over the last {span} game{plural} — no single metric moved meaningfully."

    _score, label, fmt, first, last, move = best
    direction = "up" if move > 0 else "down"
    return f"{label} {fmt(first)} → {fmt(last)} over {span} game{plural} ({direction})"
