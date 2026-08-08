"""Phase 3's core metric: is a player's production keeping up with his usage?

This is the project's existing thesis (usage moves before points do) pointed at
players you already own instead of free agents. Two rostered players can both be
scoring 9 points a game while one is earning it on a growing role and the other
is riding touchdown luck on a role he doesn't have. The box score cannot tell
them apart. Usage can.

    usage_z    = z(usage composite over W, vs the positional pool, same weeks)
    points_z   = z(fantasy points over W, vs the positional pool, same weeks)
    divergence = usage_z - points_z

**On the usage composite.** The Phase 3-5 PRD instructs "reuse the existing
opportunity_score composite, do not define a second one." No such composite
exists in this codebase -- there is `predicted_score` (raw model output, already
in points units) and there are the individual usage columns, nothing in between.
`predicted_score` is also the wrong substitute: it is trained to predict points,
so `usage_z - points_z` computed from it would measure model error, not
usage-versus-production divergence.

So this module defines one, explicitly and narrowly: the mean of per-metric
z-scores against the positional pool. It is a *descriptive* composite for
divergence only. It is NOT the opportunity model's score and must not be fed
back into the model as a feature -- that would be a leak, since it is computed
from the same window it describes.

Positions are pooled separately throughout. A 60% snap share means something
different for a running back than for a receiver, and z-scoring against a mixed
pool would smear those distributions together.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

# The usage metrics that make up the composite. red_zone_target_share is
# deliberately excluded: it is null whenever a team had no red-zone pass
# attempts that week (see transform.py), which is common enough that
# including it would drop or destabilise a lot of rows for little signal
# beyond what red_zone_touches already carries.
USAGE_METRICS = ["snap_pct", "target_share", "air_yards_share", "red_zone_touches"]

DEFAULT_WINDOW = 3

# |divergence| below this is reported as "no signal" rather than narrated.
# A starting value, NOT a tuned one -- per EVALUATION.md, threshold sweeps
# on a few hundred flagged players manufacture 2-6pp gains that invert on
# the next season. Leave it at 1.0 unless a two-season replication with a
# bootstrap CI excluding zero says otherwise.
NEUTRAL_BAND = 1.0

# A pool this small makes a z-score meaningless (one outlier moves the
# mean and the std together). Below this the player gets no divergence
# and the reason is recorded in data_gaps.
MIN_POOL_SIZE = 8

Signal = Literal["undervalued", "overvalued", "neutral"]


@dataclass(frozen=True)
class PlayerDivergence:
    player_id: str
    position: str
    usage_z: float
    points_z: float
    divergence: float
    signal: Signal
    n_games: int
    metrics_used: tuple[str, ...]  # which usage metrics were non-null for this player


def _z(series: pd.Series) -> pd.Series:
    """Standard score against the series' own distribution. A zero-variance
    pool yields 0.0 for everyone rather than inf/NaN -- if every player at a
    position has identical usage, nobody diverges from the pool, which is
    the honest reading."""
    std = series.std(ddof=0)
    if std == 0 or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def compute_divergence(
    player_week: pd.DataFrame,
    points: pd.Series,
    season: int,
    week: int,
    window: int = DEFAULT_WINDOW,
    neutral_band: float = NEUTRAL_BAND,
) -> tuple[list[PlayerDivergence], list[str]]:
    """Returns (per-player divergence, data_gaps).

    The window is the `window` weeks ending at `week` inclusive. Every
    player league-wide with data in that window forms the positional pool;
    the caller filters down to their own roster afterwards. Pooling against
    the whole league (not just your roster) is the point -- "is this player
    outrunning his usage" is only answerable relative to his position.
    """
    gaps: list[str] = []

    if window < 1:
        raise ValueError(f"divergence window must be >= 1, got {window}")

    df = player_week[
        (player_week["season"] == season)
        & (player_week["week"] <= week)
        & (player_week["week"] > week - window)
    ].copy()

    if df.empty:
        gaps.append(f"No usage data for {season} weeks {week - window + 1}-{week}; divergence unavailable.")
        return [], gaps

    # Same uniqueness contract build_features and compute_trailing_points_std
    # enforce: a duplicated player-week would double-weight that game in
    # every mean below.
    dupes = df.duplicated(subset=["player_id", "season", "week"], keep=False)
    if dupes.any():
        sample = df.loc[dupes, ["player_id", "season", "week"]].drop_duplicates().head(5)
        raise ValueError(
            f"player_week has {int(dupes.sum())} row(s) sharing a duplicate "
            "(player_id, season, week) key -- compute_divergence requires this to be "
            f"unique. Example duplicated keys:\n{sample.to_string(index=False)}"
        )

    df["points"] = points.reindex(df.index).to_numpy()

    weeks_present = set(df["week"].unique())
    weeks_expected = {w for w in range(week - window + 1, week + 1) if w >= 1}
    missing_weeks = sorted(weeks_expected - weeks_present)
    if missing_weeks:
        gaps.append(
            f"No data for {season} week(s) {missing_weeks}; divergence is computed from "
            f"{len(weeks_present)} of {len(weeks_expected)} weeks in the window."
        )

    available_metrics = [m for m in USAGE_METRICS if m in df.columns]
    if len(available_metrics) < len(USAGE_METRICS):
        missing = sorted(set(USAGE_METRICS) - set(available_metrics))
        gaps.append(f"Usage metric(s) {missing} absent from the data; composite uses the rest.")
    if not available_metrics:
        gaps.append("No usage metrics available at all; divergence unavailable.")
        return [], gaps

    agg = {m: (m, "mean") for m in available_metrics}
    agg["points"] = ("points", "mean")
    agg["n_games"] = ("week", "count")
    per_player = df.groupby(["player_id", "position"], as_index=False).agg(**agg)

    results: list[PlayerDivergence] = []
    small_pools: list[str] = []

    for position, pool in per_player.groupby("position"):
        if len(pool) < MIN_POOL_SIZE:
            small_pools.append(f"{position} (n={len(pool)})")
            continue

        pool = pool.copy()
        # z-score each metric within the position, then average whichever
        # ones this player actually has. Averaging z-scores rather than raw
        # values is what lets metrics on different scales (a 0-1 share and a
        # touch count) contribute equally.
        z_cols = []
        for metric in available_metrics:
            z_col = f"_z_{metric}"
            pool[z_col] = _z(pool[metric])
            # A metric that is null for this player (target_share for a QB,
            # say) contributes nothing rather than counting as average.
            pool.loc[pool[metric].isna(), z_col] = pd.NA
            z_cols.append(z_col)

        pool["usage_z"] = pool[z_cols].mean(axis=1, skipna=True)
        pool["points_z"] = _z(pool["points"])

        for row in pool.itertuples():
            usage_z = getattr(row, "usage_z")
            points_z = getattr(row, "points_z")
            if pd.isna(usage_z) or pd.isna(points_z):
                continue
            divergence = float(usage_z) - float(points_z)
            results.append(
                PlayerDivergence(
                    player_id=row.player_id,
                    position=position,
                    usage_z=float(usage_z),
                    points_z=float(points_z),
                    divergence=divergence,
                    signal=classify(divergence, neutral_band),
                    n_games=int(row.n_games),
                    metrics_used=tuple(
                        m for m in available_metrics if not pd.isna(getattr(row, m))
                    ),
                )
            )

    if small_pools:
        gaps.append(
            f"Positional pool too small to z-score for: {', '.join(sorted(small_pools))} "
            f"(need >= {MIN_POOL_SIZE}); those players have no divergence."
        )

    return results, gaps


def classify(divergence: float, neutral_band: float = NEUTRAL_BAND) -> Signal:
    """Above the band, usage says he's better than his box score
    (undervalued). Below it, he's outrunning his role (overvalued).
    Inside it, say "no signal" out loud -- the PRD's rule 2: if neutral
    renders as blank space, users invent signal to fill it."""
    if divergence > neutral_band:
        return "undervalued"
    if divergence < -neutral_band:
        return "overvalued"
    return "neutral"


def explain(d: PlayerDivergence) -> str:
    """Plain-text reading of the number. Never returns a score, a grade, or
    an implied action -- context is surfaced, the manager decides."""
    if d.signal == "undervalued":
        return (
            f"Usage runs ahead of production ({d.divergence:+.1f} SD over {d.n_games} game(s)) — "
            "his role is bigger than his box score suggests."
        )
    if d.signal == "overvalued":
        return (
            f"Production runs ahead of usage ({d.divergence:+.1f} SD over {d.n_games} game(s)) — "
            "points are borrowed against a role he doesn't currently have."
        )
    return (
        f"No distinguishable difference between usage and production "
        f"({d.divergence:+.1f} SD, inside the ±{NEUTRAL_BAND:.1f} neutral band)."
    )
