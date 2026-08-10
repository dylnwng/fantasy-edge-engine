"""Vacated opportunity as a FEATURE, so the boundary can be measured.

The boundary in question
------------------------
CLAUDE.md holds a load-bearing invariant: "Context is surfaced, never
baked into the score." Injury context is computed today
(model/injury_context.py) and shown as explanation text beside a
candidate; it never moves `predicted_score`.

That is a good rule for CONTEXT the user should weigh. It is a different
claim to say the underlying fact carries no predictive signal. A backup's
snap share rises when the starter ahead of him is hurt, which is about as
direct a leading indicator as this sport offers, and it is currently
excluded from the model by design rather than by measurement.

The QB model is the precedent that makes this worth testing: that scope
boundary had been assumed for the entire project on an argument that
sounded mechanical and had never actually been measured, and when it was
finally measured it replicated on two independent seasons and shipped.

This module builds the features. It does NOT adopt them, and adopting
them would be a deliberate amendment to a documented invariant, not a
refactor -- see scripts/compare_vacated_opportunity.py.

The two features
----------------
`vacated_snap_share` -- the summed trailing snap share of same-position
teammates who are BOTH ahead of this player and carrying an injury
designation. Magnitude, not a flag: one hurt teammate at a 70% snap share
frees up far more than one at 20%.

`blocker_severity` -- the worst designation among those blockers
(Out=3 > Doubtful=2 > Questionable=1, 0 for none). Roughly, how likely
the vacancy is to actually happen.

Together they are magnitude times probability, which is what a binary
"is someone hurt" flag would throw away.

No leakage
----------
Both features use ONLY injury reports from weeks <= the feature week, and
trailing snap shares from weeks strictly before it -- the same windows
model.injury_context already uses. Week N's designations are historical
fact by the time the model predicts week N+1, so nothing here requires
information that wouldn't exist when the tool actually runs.

Deliberately NOT trailed
------------------------
Unlike the usage columns, these are attached as direct features rather
than averaged through build_features. The signal is "a blocker is hurt
now" -- a step change -- and running it through another trailing window
would smear exactly the transition worth detecting. The values are
already computed from a lookback window, so they are smoothed once
already.
"""

from __future__ import annotations

import pandas as pd

from edge_engine.model.injury_context import _STATUS_SEVERITY

VACATED_COLUMNS = ["vacated_snap_share", "blocker_severity"]

DEFAULT_USAGE_GAP = 0.15
DEFAULT_LOOKBACK = 2


def _trailing_usage_by_group(
    player_week: pd.DataFrame, lookback: int
) -> dict[tuple, dict[str, float]]:
    """(season, target_week, team, position) -> {player_id: trailing snap_pct}.

    A row at week w contributes to target weeks w+1 .. w+lookback, which
    reproduces injury_context._teammates_ahead's `week < target and week
    >= target - lookback` filter without re-scanning the whole frame once
    per candidate.
    """
    contributions = []
    for offset in range(1, lookback + 1):
        shifted = player_week[["season", "week", "player_id", "team", "position", "snap_pct"]].copy()
        shifted["target_week"] = shifted["week"] + offset
        contributions.append(shifted)

    prior = pd.concat(contributions, ignore_index=True)
    means = prior.groupby(
        ["season", "target_week", "team", "position", "player_id"], dropna=False
    )["snap_pct"].mean()

    grouped: dict[tuple, dict[str, float]] = {}
    for (season, week, team, position, player_id), usage in means.items():
        grouped.setdefault((season, week, team, position), {})[player_id] = usage
    return grouped


def _severity_by_player_week(injuries: pd.DataFrame, lookback: int) -> dict[tuple, int]:
    """(season, target_week, gsis_id) -> worst designation in the window.

    A report at week w applies to target weeks w .. w+lookback, matching
    injury_context's `week <= target and week >= target - lookback`. Note
    the window INCLUDES the feature week itself, unlike the trailing usage
    above -- a designation published for a game already played is known.
    """
    if injuries.empty:
        return {}

    inj = injuries[["season", "week", "gsis_id", "report_status"]].copy()
    inj["severity"] = inj["report_status"].map(_STATUS_SEVERITY)
    inj = inj[inj["severity"].notna()]
    if inj.empty:
        return {}

    frames = []
    for offset in range(0, lookback + 1):
        shifted = inj.copy()
        shifted["target_week"] = shifted["week"] + offset
        frames.append(shifted)

    expanded = pd.concat(frames, ignore_index=True)
    worst = expanded.groupby(["season", "target_week", "gsis_id"])["severity"].max()
    return {key: int(value) for key, value in worst.items()}


def build_vacated_opportunity(
    player_week: pd.DataFrame,
    injuries: pd.DataFrame,
    usage_gap: float = DEFAULT_USAGE_GAP,
    lookback: int = DEFAULT_LOOKBACK,
) -> pd.DataFrame:
    """(season, week, player_id, vacated_snap_share, blocker_severity).

    `injuries` is load_injury_reports()'s output. A player with no
    injured teammate ahead of him gets 0.0 / 0 -- a genuine "nothing is
    vacated", not a missing value, because the absence of a blocker on
    the injury report is itself the observation.
    """
    if usage_gap < 0:
        raise ValueError(f"usage_gap must be >= 0, got {usage_gap}")
    if lookback < 1:
        raise ValueError(f"lookback must be >= 1, got {lookback}")

    required = {"season", "week", "player_id", "team", "position", "snap_pct"}
    missing = required - set(player_week.columns)
    if missing:
        raise ValueError(f"player_week is missing {sorted(missing)}")

    trailing = _trailing_usage_by_group(player_week, lookback)
    severity = _severity_by_player_week(injuries, lookback)

    vacated_shares: list[float] = []
    severities: list[int] = []

    for row in player_week.itertuples():
        group = trailing.get((row.season, row.week, row.team, row.position), {})
        own = group.get(row.player_id, 0.0)
        if pd.isna(own):
            own = 0.0

        vacated = 0.0
        worst = 0
        for teammate_id, teammate_usage in group.items():
            if teammate_id == row.player_id or pd.isna(teammate_usage):
                continue
            if teammate_usage - own < usage_gap:
                continue
            hurt = severity.get((row.season, row.week, teammate_id))
            if hurt is None:
                continue
            vacated += float(teammate_usage)
            worst = max(worst, hurt)

        vacated_shares.append(vacated)
        severities.append(worst)

    out = player_week[["season", "week", "player_id"]].copy()
    out["vacated_snap_share"] = vacated_shares
    out["blocker_severity"] = severities
    return out.reset_index(drop=True)


def attach_vacated_opportunity(
    features: pd.DataFrame, vacated: pd.DataFrame
) -> pd.DataFrame:
    """Merge the features onto a built feature table by (season, week, player_id).

    Rows with no vacated-opportunity row get 0.0 / 0 rather than a null:
    "no injured blocker" is an observation, not missing data, and leaving
    nulls here would drop those rows from training for no reason.
    """
    merged = features.merge(vacated, on=["season", "week", "player_id"], how="left")
    if len(merged) != len(features):
        raise ValueError(
            f"attaching vacated opportunity changed the row count "
            f"({len(features)} -> {len(merged)}) -- duplicate (season, week, player_id) keys."
        )
    merged["vacated_snap_share"] = merged["vacated_snap_share"].fillna(0.0)
    merged["blocker_severity"] = merged["blocker_severity"].fillna(0).astype(int)
    return merged
