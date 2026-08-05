"""Turns a player's trailing-window usage rows into the "which specific
metric moved" story requirement 4 needs (e.g. "snap share 38% -> 64%
over 2 wks"), instead of a black-box score. get_usage_trend() exposes the
raw series (for a sparkline or other visualization); usage_trend_explanation()
formats the same selection into the sentence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# (column, display label, is_percentage) -- percentage metrics are
# preferred spotlighting candidates since they're the clearest read on a
# role change; red_zone_touches is a fallback when neither moved much.
_TRACKED_METRICS = [
    ("snap_pct", "snap share", True),
    ("target_share", "target share", True),
    ("red_zone_touches", "red zone touches", False),
]

_PERCENT_MOVE_FLOOR = 0.05  # 5 points -- below this a % metric isn't "the story"


@dataclass(frozen=True)
class UsageTrend:
    insufficient_history: bool
    moved_meaningfully: bool
    label: str | None
    values: list[float] = field(default_factory=list)
    weeks: list[int] = field(default_factory=list)
    is_percentage: bool = False
    n_weeks_span: int = 0


def get_usage_trend(player_week: pd.DataFrame, player_id: str, season: int, week: int, window: int) -> UsageTrend:
    rows = player_week[
        (player_week["season"] == season)
        & (player_week["player_id"] == player_id)
        & (player_week["week"] <= week)
        & (player_week["week"] > week - window)
    ].sort_values("week")

    if len(rows) < 2:
        return UsageTrend(insufficient_history=True, moved_meaningfully=False, label=None)

    n_weeks = int(rows["week"].iloc[-1] - rows["week"].iloc[0])
    weeks = rows["week"].tolist()

    percent_moves = []
    other_moves = []
    for col, label, is_pct in _TRACKED_METRICS:
        series = rows[col]
        if series.isna().any():
            continue
        delta = series.iloc[-1] - series.iloc[0]
        entry = (abs(delta), label, series.tolist(), is_pct)
        (percent_moves if is_pct else other_moves).append(entry)

    percent_moves.sort(key=lambda m: -m[0])
    if percent_moves and percent_moves[0][0] >= _PERCENT_MOVE_FLOOR:
        _, label, values, is_pct = percent_moves[0]
        return UsageTrend(False, True, label, values, weeks, is_pct, n_weeks)

    other_moves.sort(key=lambda m: -m[0])
    if other_moves and other_moves[0][0] >= 1:
        _, label, values, is_pct = other_moves[0]
        return UsageTrend(False, True, label, values, weeks, is_pct, n_weeks)

    # Nothing moved meaningfully -- still surface the closest-to-moving
    # series (for a flat sparkline) rather than nothing at all.
    fallback = percent_moves[0] if percent_moves else (other_moves[0] if other_moves else None)
    if fallback:
        _, label, values, is_pct = fallback
        return UsageTrend(False, False, label, values, weeks, is_pct, n_weeks)
    return UsageTrend(False, False, None, [], weeks, False, n_weeks)


def usage_trend_explanation(player_week: pd.DataFrame, player_id: str, season: int, week: int, window: int) -> str:
    trend = get_usage_trend(player_week, player_id, season, week, window)

    if trend.insufficient_history:
        return "Limited usage history this season — not enough games yet to show a trend."

    plural = "s" if trend.n_weeks_span != 1 else ""
    if not trend.moved_meaningfully or trend.label is None:
        return f"Usage steady over the last {trend.n_weeks_span} wk{plural} — no single metric moved meaningfully."

    f, l = trend.values[0], trend.values[-1]
    direction = "up" if l > f else "down"
    if trend.is_percentage:
        return f"{trend.label} {f:.0%} → {l:.0%} over {trend.n_weeks_span} wk{plural} ({direction})"
    return f"{trend.label} {f:.0f} → {l:.0f} over {trend.n_weeks_span} wk{plural} ({direction})"
