"""Turns a player's trailing-window usage rows into the "which specific
metric moved" explanation requirement 4 needs (e.g. "snap share 38% -> 64%
over 2 wks"), instead of a black-box score.
"""

from __future__ import annotations

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


def usage_trend_explanation(player_week: pd.DataFrame, player_id: str, season: int, week: int, window: int) -> str:
    rows = player_week[
        (player_week["season"] == season)
        & (player_week["player_id"] == player_id)
        & (player_week["week"] <= week)
        & (player_week["week"] > week - window)
    ].sort_values("week")

    if len(rows) < 2:
        return "Limited usage history this season — not enough games yet to show a trend."

    first, last = rows.iloc[0], rows.iloc[-1]
    n_weeks = int(last["week"] - first["week"])

    percent_moves = []
    other_moves = []
    for col, label, is_pct in _TRACKED_METRICS:
        f, l = first.get(col), last.get(col)
        if pd.isna(f) or pd.isna(l):
            continue
        delta = l - f
        (percent_moves if is_pct else other_moves).append((abs(delta), label, f, l, delta, is_pct))

    percent_moves.sort(key=lambda m: -m[0])
    if percent_moves and percent_moves[0][0] >= _PERCENT_MOVE_FLOOR:
        _, label, f, l, delta, _ = percent_moves[0]
        direction = "up" if delta > 0 else "down"
        return f"{label} {f:.0%} → {l:.0%} over {n_weeks} wk{'s' if n_weeks != 1 else ''} ({direction})"

    other_moves.sort(key=lambda m: -m[0])
    if other_moves and other_moves[0][0] >= 1:
        _, label, f, l, delta, _ = other_moves[0]
        direction = "up" if delta > 0 else "down"
        return f"{label} {f:.0f} → {l:.0f} over {n_weeks} wk{'s' if n_weeks != 1 else ''} ({direction})"

    return f"Usage steady over the last {n_weeks} wk{'s' if n_weeks != 1 else ''} — no single metric moved meaningfully."
