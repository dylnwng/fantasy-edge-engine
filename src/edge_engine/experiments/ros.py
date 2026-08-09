"""Rest-of-season (ROS) projection: mean fantasy points per remaining game.

    ############################################################
    ##  THIS MODEL FAILED ITS KILL GATE. DO NOT SHIP IT.      ##
    ############################################################

**Status: rejected.** `scripts/validate_ros.py` runs the PRD's §2.4
protocol and this model does not clear the §2.5 gate. It beats the
positional-prior baseline enormously (+1.0 to +1.4 MAE, 100% of
bootstrap resamples) -- but that baseline is replacement level, and
beating it is table stakes. Against the baseline that matters,
season-to-date PPG, it barely moves:

    2024 origins: +0.10 [-0.14,+0.34] · +0.05 [-0.11,+0.23] · -0.00 [-0.16,+0.17]
    2025 origins: +0.19 [-0.02,+0.41] · +0.16 [-0.03,+0.34] · +0.16 [+0.00,+0.32]

Every 2024 confidence interval crosses zero, and at week 12 it is a
dead tie. Rank correlation is no better than the baseline either (0.706
vs 0.726 at week 4) -- and ranking is precisely what trade and draft
need. In hindsight the mechanism is obvious: the shrinkage IS the model,
so it helps while a player's own sample is thin and adds noise once that
sample is the best available estimate.

Kept, not deleted, for the same reason train_with_dvp.py and
train_with_persistence.py are kept: a documented rejection stops the
next person rebuilding it from scratch. **It must not be imported into
the trade path** -- per §3.7, trade/compare.py ships as a pure surfacer
with no ROS number at all.

**Why this is a separate model and not the weekly one run repeatedly.**
The weekly model predicts `points[t+1]` from `usage[t-1:t]`. Chaining it
to reach week 17 would require predicting *usage*, which it does not do,
and its entire edge comes from recent *change* -- which is exactly the
thing that is unpredictable ten weeks out. Substituting predicted points
for observed usage compounds error at every step. The PRD's expectation
is that a naive chain loses to a flat positional baseline by about week
5, and it says not to build it just to prove that.

So this is a different estimator with a different target and different
features:

  target   : mean points per game over weeks t+1..17 (the remaining season)
  features : season-to-date usage RATE (not trend), games played,
             season-to-date PPG, prior-season PPG where available
  shrinkage: blend toward the positional replacement baseline with weight
             1/(1 + games_played/k), so a week-2 projection is mostly
             prior and a week-12 projection is mostly observed

Rate, not trend, is the whole point. Over a 10-week horizon "what is this
player's established role" beats "what changed last week", which is the
opposite of the weekly model's premise.

**Every projection carries an interval, and it is a required field, not
Optional.** A bare ROS point estimate is the single most likely way this
project starts producing overconfident output -- intervals for realistic
trade comparisons overlap most of the time, and that overlap IS the
answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Usage rates that describe an established role. Same columns the weekly
# model uses, but averaged season-to-date rather than differenced.
USAGE_RATE_COLUMNS = ["snap_pct", "target_share", "air_yards_share", "red_zone_touches"]

FINAL_WEEK = 17

# Shrinkage half-weight: at k games played, the projection is an even
# blend of observed rate and positional prior. 4 is deliberate rather
# than tuned -- a quarter-season is roughly when a role stops being noise
# -- and it is NOT swept, per EVALUATION.md's finding that threshold
# sweeps on this data manufacture gains that invert.
SHRINKAGE_K = 4.0

# Below this many games played, a season-to-date rate is too thin to
# lean on at all and the projection is essentially the prior.
MIN_GAMES_FOR_RATE = 1


@dataclass(frozen=True)
class RosProjection:
    player_id: str
    position: str
    estimate: float
    interval: tuple[float, float]  # required, never Optional -- see module docstring
    games_played: int
    shrinkage_weight: float  # 1.0 = fully observed, 0.0 = fully prior
    low_confidence: bool


def build_ros_table(
    player_week: pd.DataFrame, points: pd.Series, origin_week: int, final_week: int = FINAL_WEEK
) -> pd.DataFrame:
    """One row per player at a given origin week: season-to-date features
    plus the realised rest-of-season target.

    `origin_week` is the last week whose results are known. Features use
    weeks <= origin_week; the target uses weeks > origin_week, so there is
    no overlap and no leakage by construction."""
    df = player_week.copy()
    df["points"] = points.reindex(df.index).to_numpy()

    dupes = df.duplicated(subset=["player_id", "season", "week"], keep=False)
    if dupes.any():
        sample = df.loc[dupes, ["player_id", "season", "week"]].drop_duplicates().head(5)
        raise ValueError(
            f"player_week has {int(dupes.sum())} row(s) sharing a duplicate "
            "(player_id, season, week) key -- build_ros_table requires this to be unique. "
            f"Example duplicated keys:\n{sample.to_string(index=False)}"
        )

    past = df[df["week"] <= origin_week]
    future = df[(df["week"] > origin_week) & (df["week"] <= final_week)]
    if past.empty or future.empty:
        return pd.DataFrame()

    available_rates = [c for c in USAGE_RATE_COLUMNS if c in past.columns]
    agg = {f"rate_{c}": (c, "mean") for c in available_rates}
    agg["std_ppg"] = ("points", "std")
    agg["ppg_to_date"] = ("points", "mean")
    agg["games_played"] = ("week", "count")
    features = past.groupby(["player_id", "season", "position"], as_index=False).agg(**agg)

    target = (
        future.groupby(["player_id", "season"], as_index=False)["points"]
        .agg(ros_ppg="mean", ros_games="count")
    )

    table = features.merge(target, on=["player_id", "season"], how="inner")
    table["origin_week"] = origin_week
    return table


def positional_prior(table: pd.DataFrame) -> dict[str, float]:
    """Replacement-level PPG per position: the median season-to-date PPG
    among players with at least a couple of games. The thing a projection
    shrinks toward when a player's own sample is thin."""
    established = table[table["games_played"] >= 2]
    if established.empty:
        return {}
    return established.groupby("position")["ppg_to_date"].median().to_dict()


def project(
    table: pd.DataFrame, prior: dict[str, float], k: float = SHRINKAGE_K
) -> list[RosProjection]:
    """Shrunk projection with an interval.

    estimate = w * ppg_to_date + (1 - w) * positional_prior,  w = g/(g+k)

    The interval is the player's own scoring spread, widened when the
    projection is mostly prior rather than mostly observed -- a week-2
    projection genuinely knows less than a week-12 one and must say so."""
    out: list[RosProjection] = []
    for row in table.itertuples():
        games = int(row.games_played)
        base = prior.get(row.position)
        if base is None:
            base = float(row.ppg_to_date)

        weight = games / (games + k)
        estimate = weight * float(row.ppg_to_date) + (1 - weight) * float(base)

        # Spread: the player's own week-to-week std when we have one,
        # else the positional prior's magnitude as a crude stand-in.
        spread = float(row.std_ppg) if not pd.isna(row.std_ppg) else abs(base) * 0.5
        # Uncertainty about the MEAN of the remaining games falls with
        # games observed; the (1 - weight) term widens it further when
        # the estimate is mostly prior.
        half_width = 1.96 * spread / np.sqrt(max(games, 1)) + (1 - weight) * abs(base) * 0.5

        out.append(
            RosProjection(
                player_id=row.player_id,
                position=row.position,
                estimate=float(estimate),
                interval=(float(estimate - half_width), float(estimate + half_width)),
                games_played=games,
                shrinkage_weight=float(weight),
                low_confidence=games < MIN_GAMES_FOR_RATE + 1,
            )
        )
    return out


def intervals_overlap(a: RosProjection, b: RosProjection) -> bool:
    """True when two projections are not distinguishable given the data.
    Expected to be the common case for realistic comparisons, which is
    why it is a first-class state rather than an edge case."""
    lo_a, hi_a = a.interval
    lo_b, hi_b = b.interval
    return not (hi_a < lo_b or hi_b < lo_a)
