"""Does QB *volume* carry the same kind of signal target share does?

The project's stated scope gap is that the opportunity model doesn't cover
quarterbacks, on the grounds that its receiving-usage features are
undefined for them. That's true but incomplete as an argument: QBs are
tracked for attempts, passing yards and touchdowns, all of which this repo
already computes. So the fair question is not "is there data" but "does
QB volume predict next week better than just averaging his recent points."

This probe builds the QB-appropriate analogue of the usage features --
dropback volume, rush volume, air yards, team pass rate -- and runs the
exact test the rest of the project uses: beat a trailing-points baseline
on a held-out season, or don't ship.

Usage:
    python scripts/qb_signal_probe.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_engine.ingestion.raw import fetch_pbp_data  # noqa: E402

# Two independent validation seasons, each with a train set that excludes
# it. A single-season win is exactly what the usage-persistence experiment
# produced before reversing sign on fresh data -- so this replicates
# before concluding anything.
SPLITS = [
    ([2019, 2020, 2021, 2022, 2023], 2024),
    ([2019, 2020, 2021, 2022, 2023, 2024], 2025),
]
WINDOW = 2

# Standard scoring weights (nflfastR's fantasy_points definition).
PASS_YDS, RUSH_YDS = 25.0, 10.0
PASS_TD, RUSH_TD, INT_PTS = 4.0, 6.0, -2.0


def qb_weeks(season: int) -> pd.DataFrame:
    """Per-QB, per-week volume + production, straight from play-by-play."""
    pbp = fetch_pbp_data(season)
    pbp = pbp[pbp["season_type"] == "REG"]

    passing = (
        pbp[pbp["passer_player_id"].notna()]
        .groupby(["season", "week", "passer_player_id"], as_index=False)
        .agg(
            attempts=("pass_attempt", "sum"),
            passing_yards=("passing_yards", "sum"),
            passing_tds=("pass_touchdown", "sum"),
            interceptions=("interception", "sum"),
            air_yards=("air_yards", "sum"),
            team=("posteam", "first"),
        )
        .rename(columns={"passer_player_id": "player_id"})
    )
    rushing = (
        pbp[pbp["rusher_player_id"].notna()]
        .groupby(["season", "week", "rusher_player_id"], as_index=False)
        .agg(rush_attempts=("rush_attempt", "sum"),
             rushing_yards=("rushing_yards", "sum"),
             rushing_tds=("rush_touchdown", "sum"))
        .rename(columns={"rusher_player_id": "player_id"})
    )
    df = passing.merge(rushing, on=["season", "week", "player_id"], how="left").fillna(0.0)

    # Team pass volume that week -- context for how pass-heavy the offense is.
    team_pass = (
        pbp[pbp["pass_attempt"] == 1]
        .groupby(["season", "week", "posteam"], as_index=False)["pass_attempt"]
        .sum().rename(columns={"posteam": "team", "pass_attempt": "team_attempts"})
    )
    df = df.merge(team_pass, on=["season", "week", "team"], how="left")
    df["pass_share"] = (df["attempts"] / df["team_attempts"]).where(df["team_attempts"] > 0)

    df["points"] = (
        df["passing_yards"] / PASS_YDS + df["passing_tds"] * PASS_TD
        + df["interceptions"] * INT_PTS
        + df["rushing_yards"] / RUSH_YDS + df["rushing_tds"] * RUSH_TD
    )
    # Real starters only: a mop-up drive isn't a fantasy-relevant QB week.
    return df[df["attempts"] >= 10].copy()


FEATURES = ["attempts", "rush_attempts", "air_yards", "pass_share", "team_attempts"]


def build(seasons: list[int]) -> pd.DataFrame:
    frames = [qb_weeks(s) for s in seasons]
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["player_id", "season", "week"]).reset_index(drop=True)
    g = df.groupby(["player_id", "season"], group_keys=False)

    for col in FEATURES:
        df[f"trail_{col}"] = g[col].transform(lambda s: s.rolling(WINDOW, min_periods=WINDOW).mean())
        df[f"trend_{col}"] = g[col].transform(lambda s: s - s.shift(WINDOW))
    df["trail_points"] = g["points"].transform(lambda s: s.rolling(WINDOW, min_periods=WINDOW).mean())
    df["label"] = g["points"].transform(lambda s: s.shift(-1))
    return df


def evaluate(train_seasons: list[int], validation_season: int, feat_cols: list[str]) -> dict:
    train = build(train_seasons).dropna(subset=[*feat_cols, "label"])
    val = build([validation_season]).dropna(subset=[*feat_cols, "label"]).copy()
    if len(train) < 200 or len(val) < 50:
        raise SystemExit(f"Not enough QB rows for {validation_season}.")

    model = xgb.XGBRegressor(
        objective="reg:squarederror", n_estimators=200, max_depth=4,
        learning_rate=0.05, random_state=0,
    )
    model.fit(train[feat_cols], train["label"])
    val["pred"] = model.predict(val[feat_cols])

    actual = val["label"].to_numpy()
    model_err = np.abs(val["pred"].to_numpy() - actual)
    base_err = np.abs(val["trail_points"].to_numpy() - actual)

    # Paired bootstrap: same rows scored both ways each resample.
    rng = np.random.default_rng(0)
    diffs = np.array([
        base_err[idx].mean() - model_err[idx].mean()
        for idx in (rng.integers(0, len(actual), len(actual)) for _ in range(2000))
    ])
    return {
        "season": validation_season, "n": len(val),
        "model_mae": model_err.mean(), "baseline_mae": base_err.mean(),
        "mean_gain": diffs.mean(),
        "ci_low": np.percentile(diffs, 5), "ci_high": np.percentile(diffs, 95),
        "favor": (diffs > 0).mean(),
        "importances": sorted(zip(feat_cols, model.feature_importances_), key=lambda kv: -kv[1]),
    }


def main() -> None:
    feat_cols = [f"trail_{c}" for c in FEATURES] + [f"trend_{c}" for c in FEATURES] + ["trail_points"]
    results = [evaluate(tr, va, feat_cols) for tr, va in SPLITS]

    print("QB VOLUME SIGNAL — does it beat a trailing-points baseline?\n")
    header = f"{'season':<9}{'n':<7}{'model':<9}{'baseline':<11}{'MAE gain (90% CI)':<28}{'favor':<7}"
    print(header)
    print("-" * len(header))
    for r in results:
        ci = f"{r['mean_gain']:+.2f} [{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]"
        print(f"{r['season']:<9}{r['n']:<7}{r['model_mae']:<9.2f}{r['baseline_mae']:<11.2f}{ci:<28}{r['favor']:<7.0%}")

    print("\nTop features (most recent split):")
    for name, imp in results[-1]["importances"][:6]:
        print(f"  {name:22} {imp:.3f}")

    replicates = all(r["ci_low"] > 0 for r in results)
    print("\n" + "=" * 72)
    if replicates:
        print("REPLICATES: the CI excludes zero on BOTH validation seasons.")
        print("This clears the same bar that rejected DvP, persistence and the ROS model.")
    else:
        print("DOES NOT REPLICATE: at least one season's CI crosses zero.")
        print("Treat the winning season as noise, exactly like usage-persistence.")
    print("=" * 72)


if __name__ == "__main__":
    main()
