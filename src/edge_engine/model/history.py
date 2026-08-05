"""P1: historical accuracy tracker.

For every player-week the model flagged as a breakout candidate, look at
how that player actually performed over the following up-to-3 games and
compare to their own baseline at the time of the flag. This is a longer,
more robust window than the single-week hit rate in train.py's
validation report, and is the concrete "players flagged early + how they
did" evidence the PRD's portfolio-artifact framing asks for.

Defaults to the held-out validation season only -- that's the only
result that's actually out-of-sample. Training seasons can be included
via --include-train-seasons, but those numbers are in-sample (the model
saw that data) and will look more flattering than real-world
performance; the CLI labels them accordingly rather than blending them
into one number.

Usage:
    python -m edge_engine.model.history
    python -m edge_engine.model.history --include-train-seasons
"""

from __future__ import annotations

import argparse

import pandas as pd
import xgboost as xgb

from edge_engine.model.config import ModelConfig, load_model_config
from edge_engine.model.features import build_features, feature_columns
from edge_engine.model.train import MODEL_PATH, _compute_points_for_seasons, _ensure_seasons_ingested
from edge_engine.roster.interface import get_default_source

FOLLOWING_WINDOW = 3


def _load_model() -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(str(MODEL_PATH))
    return model


def _attach_following_performance(flagged: pd.DataFrame, points_table: pd.DataFrame, window: int) -> pd.DataFrame:
    pts = points_table.sort_values(["player_id", "season", "week"]).copy()
    grouped = pts.groupby(["player_id", "season"])["points"]
    shifted_cols = []
    for offset in range(1, window + 1):
        col = f"_pts_next_{offset}"
        pts[col] = grouped.shift(-offset)
        shifted_cols.append(col)

    pts["following_games_played"] = pts[shifted_cols].notna().sum(axis=1)
    pts["following_avg_points"] = pts[shifted_cols].mean(axis=1)

    return flagged.merge(
        pts[["season", "week", "player_id", "following_games_played", "following_avg_points"]],
        on=["season", "week", "player_id"],
        how="left",
    )


def track_flagged_players(seasons: list[int], config: ModelConfig | None = None) -> pd.DataFrame:
    """One row per player-week the model flagged in `seasons`, with how
    that player actually performed over the following up-to-3 games vs.
    their baseline at the time of the flag. Flags with zero following
    games available (end of season) are excluded -- nothing to grade
    them against yet."""
    config = config or load_model_config()
    model = _load_model()

    player_week = _ensure_seasons_ingested(seasons)
    player_week = player_week[player_week["season"].isin(seasons)]

    league_config = get_default_source().get_league_config()
    points_table = _compute_points_for_seasons(seasons, league_config.scoring)
    merged = player_week.merge(points_table, on=["season", "week", "player_id"], how="inner")

    features = build_features(merged, merged["points"], window=config.trailing_window)
    feat_cols = feature_columns(config.trailing_window)
    scoreable = features.dropna(subset=feat_cols).copy()

    scoreable["predicted_score"] = model.predict(scoreable[feat_cols])
    scoreable["baseline_score"] = scoreable["trailing_points_avg"]
    scoreable["flagged"] = (scoreable["predicted_score"] - scoreable["baseline_score"]) > config.flag_margin

    flagged = scoreable[scoreable["flagged"]].copy()
    flagged = _attach_following_performance(flagged, points_table, FOLLOWING_WINDOW)
    flagged = flagged[flagged["following_games_played"] > 0]
    flagged["beat_baseline"] = flagged["following_avg_points"] > flagged["baseline_score"]

    cols = [
        "season", "week", "player_id", "position", "team",
        "predicted_score", "baseline_score",
        "following_games_played", "following_avg_points", "beat_baseline",
    ]
    return flagged[cols].sort_values(["season", "week"]).reset_index(drop=True)


def summarize(tracked: pd.DataFrame) -> dict:
    if tracked.empty:
        return {"n_flagged": 0, "hit_rate_3wk": None, "avg_points_above_baseline": None}
    return {
        "n_flagged": int(len(tracked)),
        "hit_rate_3wk": float(tracked["beat_baseline"].mean()),
        "avg_points_above_baseline": float((tracked["following_avg_points"] - tracked["baseline_score"]).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--include-train-seasons",
        action="store_true",
        help="also report in-sample training seasons, labeled separately (not blended with the out-of-sample number)",
    )
    args = parser.parse_args()

    config = load_model_config()

    print(f"=== Out-of-sample: validation season {config.validation_season} ===")
    val_tracked = track_flagged_players([config.validation_season], config)
    val_summary = summarize(val_tracked)
    _print_summary(val_summary)
    if val_summary["n_flagged"]:
        print(val_tracked.to_string(index=False))

    if args.include_train_seasons:
        print(f"\n=== In-sample (model saw this data) — training seasons {config.train_seasons} ===")
        print("These numbers are expected to look better than real-world performance; use for breadth, not as evidence.")
        train_tracked = track_flagged_players(config.train_seasons, config)
        _print_summary(summarize(train_tracked))


def _print_summary(summary: dict) -> None:
    print(f"Flagged {summary['n_flagged']} player-weeks with at least one following game to grade against.")
    if summary["n_flagged"]:
        print(f"3-week hit rate: {summary['hit_rate_3wk']:.1%}")
        print(f"Avg next-3wk points above baseline: {summary['avg_points_above_baseline']:.2f}")
    print()


if __name__ == "__main__":
    main()
