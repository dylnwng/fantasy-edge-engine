"""Trade comparison as a **pure surfacer**: what the usage data says about
each side. No projection, no verdict.

**Why there is no ROS number here.** The rest-of-season model in ros.py
failed its §2.5 kill gate -- it does not reliably beat season-to-date
PPG, and its rank correlation is no better either. §3.7 says exactly what
to do in that case: ship compare.py as a pure surfacer and keep the ROS
projection out of the trade path entirely. So this module shows
divergence and observed rates side by side and stops there. That is
still useful, and it is honest.

What is deliberately absent, and stays absent: `verdict`,
`recommendation`, `fairness_score`, a winner label, a letter grade. The
intervals on any honest projection of this would overlap most of the
time, and the tool refuses to launder that into confidence. Whether a
trade is good depends on your roster construction and your risk
appetite, neither of which this models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from edge_engine.insights.divergence import PlayerDivergence, compute_divergence, explain


@dataclass(frozen=True)
class PlayerSummary:
    player_id: str | None
    name: str
    position: str
    ppg_to_date: float | None
    games_played: int
    divergence: float | None
    signal: str
    context: list[str] = field(default_factory=list)  # text only, never numeric


@dataclass(frozen=True)
class TradeSide:
    players: list[PlayerSummary]

    @property
    def total_ppg_to_date(self) -> float:
        """Sum of OBSERVED points per game. Explicitly a description of
        what has already happened, not a projection of what will."""
        return sum(p.ppg_to_date or 0.0 for p in self.players)


@dataclass(frozen=True)
class TradeComparison:
    season: int
    week: int
    window: int
    outgoing: TradeSide
    incoming: TradeSide
    context: list[str]
    data_gaps: list[str]


def _summarise(
    names: list[str],
    resolved: dict[str, tuple[str, str]],  # name -> (player_id, position)
    rates: pd.DataFrame,
    divergence_by_id: dict[str, PlayerDivergence],
) -> tuple[list[PlayerSummary], list[str]]:
    gaps: list[str] = []
    out: list[PlayerSummary] = []
    rate_by_id = {r.player_id: r for r in rates.itertuples()}

    for name in names:
        entry = resolved.get(name.lower())
        if entry is None:
            gaps.append(f"'{name}' did not resolve to a player in the usage data.")
            out.append(PlayerSummary(None, name, "?", None, 0, None, "no_data",
                                     ["Unresolved — no usage history to show."]))
            continue

        player_id, position = entry
        rate = rate_by_id.get(player_id)
        d = divergence_by_id.get(player_id)

        context: list[str] = []
        if d is not None:
            context.append(explain(d))
        else:
            context.append("No usage data in the window; no divergence to report.")

        out.append(
            PlayerSummary(
                player_id=player_id, name=name, position=position,
                ppg_to_date=float(rate.ppg) if rate is not None else None,
                games_played=int(rate.games) if rate is not None else 0,
                divergence=d.divergence if d else None,
                signal=d.signal if d else "no_data",
                context=context,
            )
        )
    return out, gaps


def compare_trade(
    outgoing_names: list[str],
    incoming_names: list[str],
    player_week: pd.DataFrame,
    points: pd.Series,
    season: int,
    week: int,
    window: int = 3,
) -> TradeComparison:
    df = player_week.copy()
    df["points"] = points.reindex(df.index).to_numpy()
    to_date = df[df["week"] <= week]

    rates = (
        to_date.groupby(["player_id", "position"], as_index=False)
        .agg(ppg=("points", "mean"), games=("week", "count"))
    )

    resolved: dict[str, tuple[str, str]] = {}
    if "player_name" in df.columns:
        for row in df[["player_name", "player_id", "position"]].drop_duplicates().itertuples():
            if isinstance(row.player_name, str):
                resolved[row.player_name.lower()] = (row.player_id, row.position)

    divergences, gaps = compute_divergence(df, df["points"], season, week, window)
    by_id = {d.player_id: d for d in divergences}

    out_players, out_gaps = _summarise(outgoing_names, resolved, rates, by_id)
    in_players, in_gaps = _summarise(incoming_names, resolved, rates, by_id)

    context = [
        "This is a description of observed usage and production, not a projection. "
        "A rest-of-season model was built and rejected (see trade/ros.py and "
        "EVALUATION.md): it did not beat a season-to-date average, so no ROS number "
        "is shown rather than showing one that couldn't be validated.",
        "No verdict is offered. Whether this trade helps depends on your roster "
        "construction and risk appetite, neither of which this tool models.",
    ]

    return TradeComparison(
        season=season, week=week, window=window,
        outgoing=TradeSide(out_players), incoming=TradeSide(in_players),
        context=context, data_gaps=[*gaps, *out_gaps, *in_gaps],
    )
