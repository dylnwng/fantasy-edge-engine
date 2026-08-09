"""Score the most recent trailing-window player-weeks with the trained
opportunity model. This is the "ranked score, not just a point estimate"
deliverable from requirement 3 — requirement 4 (free-agent ranking
output) filters this down to the free-agent pool and adds explanations.

Usage:
    python -m edge_engine.model.predict
"""

from __future__ import annotations

import pandas as pd
import xgboost as xgb

from edge_engine.model.config import ModelConfig, load_model_config
from edge_engine.model.features import build_features, feature_columns
from edge_engine.model.injury_context import get_injury_context, load_injury_reports
from edge_engine.model.scoring import compute_points_for_seasons
from edge_engine.model.train import MODEL_PATH
from edge_engine.paths import PLAYER_WEEK_PATH
from edge_engine.roster.interface import get_default_source
from edge_engine.roster.roster_status import get_flagged_roster_statuses, roster_status_note


def load_model() -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    return model


def score_as_of_week(season: int, target_week: int, config: ModelConfig | None = None) -> pd.DataFrame:
    """One row per player, using each player's trailing feature snapshot
    from their last game played strictly before `target_week` in
    `season` -- i.e. only what would have been knowable before that
    week's games happened. predicted_score, baseline_score, and whether
    the flag_margin threshold is cleared.

    This is the function anything asking "what would we have predicted
    for week N" must use -- including score_latest_week() below, and the
    matchup simulator's --week flag. Naively taking each player's single
    latest-available row out of the *entire* ingested file (an earlier
    version of this function did exactly that) silently leaks trailing
    data from *after* `target_week` into the prediction whenever
    `target_week` isn't the actual most recent week in the data --
    harmless for true live "predict next week" use where there's no
    future data yet, but wrong for any retrospective "as of week N"
    query once later weeks have since been ingested.

    Players with fewer than `trailing_window` games played before
    `target_week` have no valid features yet (see features.py) and are
    excluded here, not scored with degraded/null inputs."""
    config = config or load_model_config()
    model = load_model()

    player_week = pd.read_parquet(PLAYER_WEEK_PATH)
    player_week = player_week[(player_week["season"] == season) & (player_week["week"] < target_week)]

    league_config = get_default_source().get_league_config()
    points_table = compute_points_for_seasons([season], league_config.scoring)
    merged = player_week.merge(points_table, on=["season", "week", "player_id"], how="inner")

    features = build_features(merged, merged["points"], window=config.trailing_window)
    feat_cols = feature_columns(config.trailing_window)

    latest = (
        features.dropna(subset=feat_cols)
        .sort_values(["player_id", "week"])
        .groupby("player_id", as_index=False)
        .tail(1)
        .copy()
    )

    if latest.empty:
        # Genuinely no one has enough trailing games yet (e.g. asking for
        # week <= trailing_window+1 this season) -- return the same shape
        # any caller expects, not a crash several pandas frames deep from
        # feeding model.predict() a 0-row input.
        empty_cols = [
            *latest.columns,
            "predicted_score",
            "baseline_score",
            "flagged",
            "has_injury_context",
            "injury_explanation",
            "roster_status_note",
        ]
        return pd.DataFrame(columns=empty_cols)

    latest["predicted_score"] = model.predict(latest[feat_cols])
    latest["baseline_score"] = latest["trailing_points_avg"]
    latest["flagged"] = (latest["predicted_score"] - latest["baseline_score"]) > config.flag_margin

    # req 3b: attach injury context per candidate so a downstream ranking
    # explanation (requirement 4) can state whether the usage spike might
    # be injury-driven rather than an earned role change. Uses each
    # player's own trailing-snapshot week (not target_week), matching
    # what usage_trend_explanation elsewhere expects.
    injuries = load_injury_reports([season])
    contexts = [
        get_injury_context(
            player_week,
            injuries,
            row.player_id,
            int(row.season),
            int(row.week),
            usage_gap=config.injury_ahead_usage_gap,
            lookback=config.injury_lookback_weeks,
        )
        for row in latest.itertuples()
    ]
    latest["has_injury_context"] = [c.has_injury_context for c in contexts]
    latest["injury_explanation"] = [c.explanation for c in contexts]

    # Non-injury official roster status (suspended/PUP/exempt/reserve) --
    # a separate, structured-data source from the injury report above
    # (see roster_status.py's docstring); surfaced the same way, never
    # baked into predicted_score itself.
    roster_statuses = get_flagged_roster_statuses()
    latest["roster_status_note"] = [roster_status_note(row.player_id, roster_statuses) for row in latest.itertuples()]

    # Quarterbacks come from a SEPARATE model with its own feature space
    # (volume, not receiving usage) and are unioned in here rather than
    # merged upstream -- the WR/RB/TE pipeline above is validated and must
    # not be perturbed by a position it was never built for. QB rows carry
    # the identical column contract, so everything downstream is unaware
    # two models exist. Absent a trained QB artifact this is a no-op and
    # the tool behaves exactly as it did before QBs were covered.
    from edge_engine.model.predict_qb import score_qbs_as_of_week

    qbs = score_qbs_as_of_week(season, target_week, config, player_week=player_week)
    if not qbs.empty:
        latest = pd.concat([latest, qbs], ignore_index=True)

    return latest.sort_values("predicted_score", ascending=False).reset_index(drop=True)


def score_latest_week(config: ModelConfig | None = None) -> pd.DataFrame:
    """Live/current-week convenience wrapper: scores for the week
    immediately after the most recent week present in the ingested
    data -- equivalent to score_as_of_week(latest_season, last_week + 1).
    For anything other than "predict the next live week," call
    score_as_of_week() directly with an explicit week."""
    config = config or load_model_config()
    player_week = pd.read_parquet(PLAYER_WEEK_PATH)
    latest_season = int(player_week["season"].max())
    last_played_week = int(player_week.loc[player_week["season"] == latest_season, "week"].max())
    return score_as_of_week(latest_season, last_played_week + 1, config)


if __name__ == "__main__":
    result = score_latest_week()
    cols = ["season", "week", "player_id", "position", "team", "predicted_score", "baseline_score", "flagged"]
    print(result[cols].head(25).to_string(index=False))
    print()
    print("--- flagged players with injury context ---")
    with_context = result[result["flagged"] & result["has_injury_context"]]
    if with_context.empty:
        print("(none of the flagged players this week have injury context)")
    else:
        for row in with_context.itertuples():
            print(f"{row.player_id} ({row.position}, {row.team}): {row.injury_explanation}")
