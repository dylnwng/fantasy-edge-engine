"""Trade surfacer CLI.

    python -m edge_engine.trade --out "Josh Jacobs" --in "Puka Nacua"

Shows usage-vs-production divergence and observed rates for each side.
Renders no verdict, no fairness score, no winner -- see compare.py.
"""

from __future__ import annotations

import argparse

import pandas as pd

from edge_engine.model.scoring import compute_points_for_seasons
from edge_engine.paths import PLAYER_WEEK_PATH
from edge_engine.roster.interface import get_default_source
from edge_engine.trade.compare import TradeComparison, TradeSide, compare_trade


def _render_side(label: str, side: TradeSide) -> list[str]:
    out = [f"{label}"]
    if not side.players:
        out.append("  (none)")
        return out
    for p in side.players:
        ppg = f"{p.ppg_to_date:.1f}" if p.ppg_to_date is not None else "n/a"
        out.append(f"  {p.name:<24}{p.position:<5}{ppg:>6} ppg over {p.games_played} game(s)")
        for note in p.context:
            out.append(f"      {note}")
    out.append(f"  {'':<24}{'':<5}{side.total_ppg_to_date:>6.1f} ppg combined (observed, not projected)")
    return out


def render(comparison: TradeComparison) -> str:
    out: list[str] = []
    if comparison.data_gaps:
        out.append("┌─ DATA GAPS " + "─" * 60)
        out.extend(f"│ • {g}" for g in comparison.data_gaps)
        out.append("└" + "─" * 72)
        out.append("")

    out.append(f"TRADE SURFACER — {comparison.season} week {comparison.week} "
               f"(usage window: {comparison.window} games)")
    out.append("")
    out.extend(_render_side("OUT", comparison.outgoing))
    out.append("")
    out.extend(_render_side("IN", comparison.incoming))
    out.append("")
    for note in comparison.context:
        out.append(f"• {note}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", dest="outgoing", default="", help="comma-separated players you'd give up")
    parser.add_argument("--in", dest="incoming", default="", help="comma-separated players you'd receive")
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--window", type=int, default=3)
    args = parser.parse_args(argv)

    outgoing = [n.strip() for n in args.outgoing.split(",") if n.strip()]
    incoming = [n.strip() for n in args.incoming.split(",") if n.strip()]
    if not outgoing and not incoming:
        raise SystemExit('Nothing to compare. Try: --out "Josh Jacobs" --in "Puka Nacua"')

    player_week = pd.read_parquet(PLAYER_WEEK_PATH)
    season = int(player_week["season"].max())
    available = player_week[player_week["season"] == season]
    week = args.week if args.week is not None else int(available["week"].max())

    scoring = get_default_source().get_league_config().scoring
    points = compute_points_for_seasons([season], scoring)
    merged = available.merge(points, on=["season", "week", "player_id"], how="inner")

    comparison = compare_trade(
        outgoing, incoming, merged, merged["points"], season=season, week=week, window=args.window
    )
    print(render(comparison))


if __name__ == "__main__":
    main()
