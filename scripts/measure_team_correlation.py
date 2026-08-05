"""One-off analysis: how correlated are same-team teammates' fantasy-point
residuals in the same week, really? Feeds simulation_config.yaml's
team_correlation setting -- measured from real 2018-2024 data, not
invented (same discipline as the K/D-ST std=0 decision documented in
EVALUATION.md).

"Residual" here means actual points minus the player's own running
average through their PRIOR games only (never including the week being
measured) -- isolating the shared game-script/weather/blowout component
team_correlation is meant to capture, not each player's own overall
opportunity level. Two players on a good offense trivially have
correlated raw point totals all season just because the offense is
good -- that's already captured by each player's own (mean, std) in
PlayerProjection. This measurement is specifically about whether they
swing together week to week *beyond* that.

Includes a same-size cross-team control group (residuals paired across
different teams within the same week) as a sanity check on the method
itself -- it should measure close to 0 correlation; if it doesn't,
something about the residual construction is wrong, not just the
same-team number being uninteresting.

Usage:
    python scripts/measure_team_correlation.py
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd

from edge_engine.model.scoring import compute_points_for_seasons
from edge_engine.paths import PLAYER_WEEK_PATH
from edge_engine.roster.interface import get_default_source

SEASONS = list(range(2018, 2025))


def main() -> None:
    player_week = pd.read_parquet(PLAYER_WEEK_PATH)
    player_week = player_week[player_week["season"].isin(SEASONS)]

    league_config = get_default_source().get_league_config()
    points_table = compute_points_for_seasons(SEASONS, league_config.scoring)

    df = player_week[["season", "week", "player_id", "team"]].merge(
        points_table, on=["season", "week", "player_id"], how="inner"
    )
    df = df.sort_values(["player_id", "season", "week"])
    df["expected_points"] = df.groupby(["player_id", "season"])["points"].transform(
        lambda s: s.shift(1).expanding().mean()
    )
    df = df.dropna(subset=["expected_points"])
    df["residual"] = df["points"] - df["expected_points"]

    same_pairs: list[tuple[float, float]] = []
    diff_pairs: list[tuple[float, float]] = []
    rng = np.random.default_rng(0)

    for (_season, _week), week_df in df.groupby(["season", "week"]):
        for _team, team_df in week_df.groupby("team"):
            residuals = team_df["residual"].tolist()
            if len(residuals) < 2:
                continue
            same_pairs.extend(itertools.combinations(residuals, 2))

        # Control: shuffle this week's residuals across team lines and
        # re-pair, giving a same-mechanism null sample of "unrelated"
        # pairs to sanity-check the method against.
        shuffled = week_df["residual"].sample(frac=1.0, random_state=rng.integers(0, 1_000_000)).to_numpy()
        half = len(shuffled) // 2
        diff_pairs.extend(zip(shuffled[:half], shuffled[half : 2 * half]))

    same_a, same_b = zip(*same_pairs)
    diff_a, diff_b = zip(*diff_pairs)

    same_corr = float(np.corrcoef(same_a, same_b)[0, 1])
    diff_corr = float(np.corrcoef(diff_a, diff_b)[0, 1])

    print(f"Same-team residual pairs:  {len(same_pairs)}")
    print(f"Same-team correlation:     {same_corr:.4f}")
    print()
    print(f"Cross-team control pairs:  {len(diff_pairs)}")
    print(f"Cross-team correlation:    {diff_corr:.4f}  (sanity check on the method -- should be close to 0)")
    print()
    print(f"Recommended team_correlation for simulation_config.yaml: {max(0.0, same_corr):.3f}")


if __name__ == "__main__":
    main()
