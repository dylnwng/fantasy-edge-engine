"""Draft board and live tracker.

    python -m edge_engine.draft                      # the board
    python -m edge_engine.draft --position RB        # one position
    python -m edge_engine.draft --live               # interactive tracker

Rendering follows the PRD's clock-readable rules: tier-exhaustion first
(it's what tells you to reach or wait), best-available in the largest
type, positional runs as description with no recommendation attached,
and low-confidence entries visually marked rather than footnoted -- under
a pick clock a footnote is the same as no warning at all.

The interactive mode is deliberately keyboard-only and stateless between
picks: type a name, it leaves the pool. It does not talk to ESPN. Live
draft sync (PRD §4.4) is gated on verifying the `mDraftDetail` endpoint
against a mock draft first, and that spike hasn't been run.
"""

from __future__ import annotations

import argparse

import pandas as pd

from edge_engine.draft.board import BoardPlayer, build_board, late_season_divergence
from edge_engine.draft.market import ManualMarketPriceSource
from edge_engine.draft.tracker import DraftState, Pick, unfilled_slots
from edge_engine.model.scoring import compute_points_for_seasons
from edge_engine.paths import PLAYER_WEEK_PATH
from edge_engine.roster.interface import get_default_source

_TIER_WARN_AT = 3  # tiers this thin get flagged


def _render_gaps(gaps: list[str]) -> list[str]:
    if not gaps:
        return []
    out = ["┌─ DATA GAPS " + "─" * 60]
    out.extend(f"│ • {g}" for g in gaps)
    out.append("└" + "─" * 72)
    out.append("")
    return out


def render_board(board: list[BoardPlayer], gaps: list[str], limit: int, provenance: str) -> str:
    out = _render_gaps(gaps)
    out.append(f"DRAFT BOARD — {provenance}")
    out.append("Priced at ADP. This board makes no claim to beat the market;")
    out.append("the tags flag where last season's late usage disagrees with it.")
    out.append("")
    header = f"  {'ADP':<7}{'Player':<24}{'Rank':<7}{'Tm':<5}{'Tier':<6}"
    out.append(header)
    out.append("  " + "-" * (len(header) - 2))
    for b in board[:limit]:
        marker = " ?" if b.low_confidence else "  "
        # "RB3" and "T2" read instantly; "3.2" reads as a decimal.
        rank = f"{b.position}{b.position_rank}"
        out.append(f"  {b.adp:<7.1f}{b.name:<24}{rank:<7}{b.team:<5}{'T' + str(b.tier):<6}{marker}")
        for tag in b.tags:
            out.append(f"        · {tag}")
    if len(board) > limit:
        out.append(f"\n  ({len(board) - limit} more — --limit to show more)")
    out.append("")
    out.append("  ? = priced at ADP only (rookie, unmatched, or no late-season usage data)")
    return "\n".join(out)


def render_live(state: DraftState, my_picks: list[BoardPlayer], lineup_slots: dict[str, int]) -> str:
    out: list[str] = []

    thin = [t for t in state.tiers_remaining() if t.remaining <= _TIER_WARN_AT]
    out.append("TIER CLIFFS" + ("" if thin else " — none thin yet"))
    for t in thin[:8]:
        names = ", ".join(t.names[:3])
        out.append(f"  {t.position} tier {t.tier}: {t.remaining} left  ({names})")
    out.append("")

    out.append("BEST AVAILABLE")
    for b in state.best_available(5):
        marker = " ?" if b.low_confidence else ""
        out.append(f"  {b.adp:<7.1f}{b.name:<24}{b.position:<5}tier {b.tier}{marker}")
    out.append("")

    run = state.positional_run()
    if run:
        out.append(f"RUN: {run}")
        out.append("")

    unfilled = unfilled_slots(my_picks, lineup_slots)
    roster_line = ", ".join(f"{pos} x{n}" for pos, n in sorted(unfilled.items())) or "all starting slots filled"
    out.append(f"YOUR ROSTER ({len(my_picks)} picks) — still need: {roster_line}")
    for b in my_picks:
        out.append(f"  {b.position:<5}{b.name}")

    return "\n".join(out)


def _load_divergence(season: int) -> tuple[dict[str, float], list[str]]:
    """Late-season divergence from the most recent season with data.
    Returns empty (and says why) rather than failing the whole board --
    an ADP-only board is still the useful thing."""
    try:
        player_week = pd.read_parquet(PLAYER_WEEK_PATH)
    except FileNotFoundError:
        return {}, ["No ingested usage data; board is priced at ADP with no divergence tags."]

    available = player_week[player_week["season"] == season]
    if available.empty:
        return {}, [
            f"No usage data for {season}; board is priced at ADP with no divergence tags."
        ]

    scoring = get_default_source().get_league_config().scoring
    points = compute_points_for_seasons([season], scoring)
    merged = available.merge(points, on=["season", "week", "player_id"], how="inner")
    return late_season_divergence(merged, merged["points"], season=season)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=40, help="board rows to show")
    parser.add_argument("--position", default=None, help="filter to one position")
    parser.add_argument("--season", type=int, default=None, help="season to read late usage from")
    parser.add_argument("--live", action="store_true", help="interactive pick tracker")
    args = parser.parse_args(argv)

    source = ManualMarketPriceSource()
    prices = source.get_market_prices()

    if args.season is not None:
        season = args.season
    else:
        player_week = pd.read_parquet(PLAYER_WEEK_PATH)
        season = int(player_week["season"].max())

    divergence, gaps = _load_divergence(season)
    board, board_gaps = build_board(prices, divergence_by_id=divergence)
    gaps = [*gaps, *board_gaps]

    if args.position:
        board = [b for b in board if b.position == args.position.upper()]

    provenance = f"{source.describe()}, late usage from {season}"

    if not args.live:
        print(render_board(board, gaps, args.limit, provenance))
        return

    league_config = get_default_source().get_league_config()
    state = DraftState(board=board)
    my_picks: list[BoardPlayer] = []
    by_name = {b.name.lower(): b for b in board}

    print(render_board(board, gaps, 10, provenance))
    print("\nLive tracker. Enter a drafted player's name; prefix with '*' if it's your pick.")
    print("Ctrl-D or 'quit' to exit.\n")

    round_number, pick_number = 1, 1
    while True:
        try:
            raw = input(f"R{round_number}.{pick_number:02d} > ").strip()
        except EOFError:
            break
        if not raw or raw.lower() in {"quit", "exit"}:
            break

        mine = raw.startswith("*")
        name = raw.lstrip("*").strip()
        match = by_name.get(name.lower())
        if match is None:
            print(f"  '{name}' isn't on the board — check spelling, or it's already gone.")
            continue

        state = state.apply(Pick(round_number, pick_number, match.name))
        if mine:
            my_picks.append(match)

        pick_number += 1
        if pick_number > 12:  # conventional league size; only drives the label
            round_number, pick_number = round_number + 1, 1

        print()
        print(render_live(state, my_picks, league_config.lineup_slots))
        print()


if __name__ == "__main__":
    main()
