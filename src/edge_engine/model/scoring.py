"""Fantasy point computation under explicit league scoring settings.

The model must never assume a scoring format — everything routes through
compute_fantasy_points() and the league's actual ScoringSettings
(requirement 1's league_config.yaml), confirmed full PPR for this league.

Operates on nflverse's raw weekly_data (edge_engine.ingestion.raw.
fetch_weekly_data), not the requirement-2 usage table, because
weekly_data carries nflverse's precomputed standard/PPR point totals plus
the raw TD/turnover counts needed for custom bonuses — fields the
usage table intentionally doesn't carry (that's not its job).

Known P0 gap: only per-occurrence bonuses (points per TD / turnover) are
supported. Yardage-threshold bonuses (e.g. "+3 for 100 rush yards") are
not computed — flagged via a warning if referenced, not silently
ignored. This doesn't block real usage today: this league's configured
bonuses dict is empty.
"""

from __future__ import annotations

import warnings

import pandas as pd

from edge_engine.ingestion.raw import fetch_weekly_data
from edge_engine.roster.models import ScoringSettings

# bonus config key -> raw weekly_data column(s) to sum, and the sign
# convention (turnovers are typically negative in league scoring).
_BONUS_STAT_COLUMNS: dict[str, list[str]] = {
    "pass_td": ["passing_tds"],
    "rush_td": ["rushing_tds"],
    "rec_td": ["receiving_tds"],
    "interception": ["interceptions"],
    "fumble_lost": ["receiving_fumbles_lost", "rushing_fumbles_lost", "sack_fumbles_lost"],
}


def compute_fantasy_points(weekly: pd.DataFrame, scoring: ScoringSettings) -> pd.Series:
    """weekly: nflverse raw weekly_data (or a season's worth of it) — must
    have `fantasy_points`, `fantasy_points_ppr`, and any raw stat columns
    referenced by scoring.bonuses. Returns a Series aligned to weekly's
    index."""
    if scoring.ppr_type == "standard":
        points = weekly["fantasy_points"].astype(float).copy()
    elif scoring.ppr_type == "full_ppr":
        points = weekly["fantasy_points_ppr"].astype(float).copy()
    elif scoring.ppr_type == "half_ppr":
        # Exactly halfway between standard and full PPR is mathematically
        # equivalent to +0.5/reception, since PPR's only difference from
        # standard is +1/reception.
        points = (weekly["fantasy_points"] + weekly["fantasy_points_ppr"]).astype(float) / 2
    else:
        raise ValueError(f"unknown ppr_type: {scoring.ppr_type!r}")

    unsupported = []
    for stat_key, per_occurrence_points in scoring.bonuses.items():
        cols = _BONUS_STAT_COLUMNS.get(stat_key)
        if cols is None or not all(c in weekly.columns for c in cols):
            unsupported.append(stat_key)
            continue
        occurrences = sum(weekly[c].fillna(0) for c in cols)
        points = points + occurrences * per_occurrence_points

    if unsupported:
        warnings.warn(
            f"Bonus stat(s) {unsupported} in league_config.yaml are not computed by "
            "compute_fantasy_points — only per-occurrence TD/turnover bonuses are "
            "supported in P0, not yardage-threshold bonuses.",
            stacklevel=2,
        )

    return points


def compute_points_for_seasons(seasons: list[int], scoring: ScoringSettings) -> pd.DataFrame:
    """Fantasy points under the league's actual scoring settings, computed
    from nflverse's raw weekly_data (which has the raw stat columns
    compute_fantasy_points needs) — not from the usage table, which
    intentionally doesn't carry them. Returns (season, week, player_id, points)."""
    frames = []
    for season in seasons:
        weekly = fetch_weekly_data(season)
        weekly = weekly[weekly["season_type"] == "REG"].copy()
        weekly["points"] = compute_fantasy_points(weekly, scoring).to_numpy()
        frames.append(weekly[["season", "week", "player_id", "points"]])
    return pd.concat(frames, ignore_index=True)
