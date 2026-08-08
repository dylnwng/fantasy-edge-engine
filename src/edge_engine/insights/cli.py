"""Phase 3 CLI: what is actually wrong with my roster.

    python -m edge_engine.insights [--week N] [--window N] [--all]

Rendering follows the PRD's UI rules, which are invariant enforcement
rather than decoration:

  - data_gaps renders FIRST, above the results. A user reading a partial
    report must know it's partial before reading it, not after.
  - Bye-adjusted worst-week scarcity is the headline, because it's the
    number nobody computes by hand.
  - Divergence renders as a diverging bar centred at zero with the
    threshold marked, so outliers are visible without reading digits.
  - The neutral band collapses to one line. Listing twelve no-signal
    players buries the two that matter.
  - No team grade. No roster score. Context never renders as a number.
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import pandas as pd

from edge_engine.insights.divergence import DEFAULT_WINDOW, NEUTRAL_BAND
from edge_engine.insights.report import TeamInsightReport, build_report
from edge_engine.model.config import load_model_config
from edge_engine.model.injury_context import load_injury_reports
from edge_engine.model.predict import score_as_of_week
from edge_engine.model.scoring import compute_points_for_seasons
from edge_engine.paths import PLAYER_WEEK_PATH
from edge_engine.roster.interface import get_default_source

_BAR_HALF_WIDTH = 12  # characters either side of centre
_BAR_SCALE = 3.0  # divergence value that fills the bar


def _divergence_bar(divergence: float | None) -> str:
    """Diverging bar centred at zero: left for overvalued, right for
    undervalued, with the neutral band shown as a dotted centre region."""
    if divergence is None:
        return " " * (_BAR_HALF_WIDTH * 2 + 1)

    clamped = max(-_BAR_SCALE, min(_BAR_SCALE, divergence))
    cells = int(round(abs(clamped) / _BAR_SCALE * _BAR_HALF_WIDTH))
    neutral_cells = max(1, int(round(NEUTRAL_BAND / _BAR_SCALE * _BAR_HALF_WIDTH)))

    left = [" "] * _BAR_HALF_WIDTH
    right = [" "] * _BAR_HALF_WIDTH
    for i in range(neutral_cells):
        left[_BAR_HALF_WIDTH - 1 - i] = "·"
        right[i] = "·"

    if divergence < 0:
        for i in range(cells):
            left[_BAR_HALF_WIDTH - 1 - i] = "█"
    else:
        for i in range(cells):
            right[i] = "█"

    return "".join(left) + "│" + "".join(right)


def render(report: TeamInsightReport, show_all: bool = False) -> str:
    out: list[str] = []

    if report.data_gaps:
        out.append("┌─ DATA GAPS " + "─" * 60)
        for gap in report.data_gaps:
            out.append(f"│ • {gap}")
        out.append("└" + "─" * 72)
        out.append("")

    out.append(f"ROSTER INSIGHTS — {report.season} week {report.week} "
               f"(usage window: {report.window} games)")
    out.append("")

    out.append("BYE-ADJUSTED SCARCITY  (the week your depth actually breaks)")
    header = f"  {'Pos':<6}{'Startable':<11}{'Need':<7}{'Now':<7}{'Worst wk':<11}{'Then':<7}"
    out.append(header)
    out.append("  " + "-" * (len(header) - 2))
    for s in sorted(report.scarcity, key=lambda x: x.position):
        worst_wk = str(s.worst_week) if s.worst_week is not None else "n/a"
        worst = s.worst_week_surplus
        worst_str = f"{worst:+d}" if worst is not None else "n/a"
        flag = "  ⚠" if worst is not None and worst < 0 else ""
        out.append(
            f"  {s.position:<6}{s.startable_now:<11}{s.required:<7}{s.surplus_now:<+7d}"
            f"{worst_wk:<11}{worst_str:<7}{flag}"
        )
    out.append("")

    signalled = [p for p in report.players if p.signal in ("undervalued", "overvalued")]
    neutral = [p for p in report.players if p.signal == "neutral"]
    no_data = [p for p in report.players if p.signal == "no_data"]

    out.append(f"USAGE vs PRODUCTION   (overvalued ◄── │ ──► undervalued, "
               f"·· = ±{NEUTRAL_BAND:.1f} neutral band)")
    if not signalled:
        out.append("  No player is distinguishable from his own usage this week.")
    for p in signalled:
        out.append(f"  {p.name:<22}{p.position:<5}{_divergence_bar(p.divergence)} {p.divergence:+.2f}")
        for note in p.context:
            out.append(f"      {note}")
    out.append("")

    if neutral and not show_all:
        out.append(f"  ({len(neutral)} player(s) show no distinguishable signal — --all to list them)")
    elif neutral:
        out.append("  No distinguishable signal:")
        for p in neutral:
            out.append(f"    {p.name:<22}{p.position:<5}{p.divergence:+.2f}")
    if no_data:
        out.append(f"  ({len(no_data)} player(s) have no usage data in the window)")
    out.append("")

    if report.bye_collisions:
        out.append("BYE COLLISIONS")
        for c in report.bye_collisions:
            out.append(f"  Week {c.week}: {c.position} — {', '.join(c.players)}")
        out.append("")

    if report.exposure_notes:
        out.append("EXPOSURE")
        for note in report.exposure_notes:
            out.append(f"  • {note}")

    return "\n".join(out)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--week", type=int, default=None, help="week to analyze (default: roster's as-of week)")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW, help="usage window in games")
    parser.add_argument("--all", action="store_true", help="list neutral-signal players individually")
    args = parser.parse_args(argv)

    config = load_model_config()
    source = get_default_source()
    meta = source.get_roster_meta()
    league_config = source.get_league_config()
    rostered = source.get_rostered_players()

    season = league_config.season
    week = args.week if args.week is not None else meta.as_of_week

    player_week = pd.read_parquet(PLAYER_WEEK_PATH)
    season_fallback_note: list[str] = []

    available = player_week[player_week["season"] == season]
    if available.empty:
        # Normal in the offseason: the ESPN league year rolls over to the
        # upcoming season months before a single snap is played. Describe
        # the most recent season that actually happened rather than
        # refusing to run -- but say so, loudly, in data_gaps.
        if player_week.empty:
            raise SystemExit(
                "No ingested usage data at all. Run:\n"
                "  python -m edge_engine.ingestion.pipeline --seasons 2018 2019 2020 2021 2022 2023 2024 2025"
            )
        fallback = int(player_week["season"].max())
        season_fallback_note.append(
            f"League year is {season}, which has no usage data yet (season not started). "
            f"This report describes {fallback} instead — the most recent season played."
        )
        season = fallback
        available = player_week[player_week["season"] == season]
        # The roster's as-of week refers to the LEAGUE year we just fell
        # away from, so it means nothing in the season now being
        # described. Ignore it unless the user named a week explicitly.
        if args.week is None:
            week = None

    # A roster "as-of week 0" (preseason) has no usage window behind it in
    # its own season; fall back to the last week actually played so the
    # report describes something real rather than an empty window.
    latest_played = int(available["week"].max())
    if week is None or week < 1 or week > latest_played:
        week = latest_played

    points = compute_points_for_seasons([season], league_config.scoring)
    merged = available.merge(points, on=["season", "week", "player_id"], how="inner")

    scored = score_as_of_week(season, week + 1, config)
    injuries = load_injury_reports([season])

    report = build_report(
        rostered=rostered,
        player_week=merged,
        points=merged["points"],
        scored=scored,
        lineup_slots=league_config.lineup_slots,
        injuries=injuries,
        season=season,
        week=week,
        window=args.window,
    )
    if season_fallback_note:
        report = replace(report, data_gaps=[*season_fallback_note, *report.data_gaps])
    print(render(report, show_all=args.all))


if __name__ == "__main__":
    main()
