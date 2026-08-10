"""Requirement 5: roster-fit re-ranking.

A separate layer applied *after* the opportunity model (requirement 3/4).
One layer answers "is this player's role growing" (the model); this one
answers "does that matter to me" (position scarcity + bye-week overlap
against your own roster). Keeping them separate means a bad
recommendation is easy to attribute to the right layer instead of being
buried in one opaque score.

Two factors only, per the PRD -- this is not a lineup optimizer:
  1. Position scarcity, measured against the league's starting lineup
     requirements (not raw roster counts).
  2. Bye-week collision with a same-position rostered player -- shown as
     explanation text, never baked silently into the score.

Usage:
    python -m edge_engine.ranking.roster_fit
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date

from edge_engine.model.config import load_model_config
from edge_engine.ranking.output import RankedFreeAgent, build_free_agent_rankings
from edge_engine.roster.interface import get_default_source
from edge_engine.roster.models import LeagueConfig, Player

# Standard fantasy convention: a FLEX slot can be filled by any of these.
FLEX_ELIGIBLE_POSITIONS = {"RB", "WR", "TE"}


@dataclass(frozen=True)
class RosterFitCandidate:
    candidate: RankedFreeAgent  # pre-re-rank opportunity score lives here, untouched
    roster_fit_score: float
    scarcity_multiplier: float
    position_need: float
    rostered_count_at_position: int
    bye_week_collision: str | None
    final_explanation: str


def position_need(lineup_slots: dict[str, int], position: str) -> float:
    """Starters needed at `position`, including a proportional share of
    FLEX for FLEX-eligible positions -- e.g. 1 FLEX split 3 ways among
    RB/WR/TE contributes ~0.33 of a starter each, not a whole one.

    Clamped at 0: a malformed lineup_slots value (e.g. a stray "-" typo
    in a hand-edited league_config.yaml) would otherwise produce a
    negative `need`, which scarcity_multiplier() below turns into a
    negative multiplier -- silently flipping the sign of every
    candidate's roster_fit_score at that position and inverting the
    ranking with no error. A slot count can never be genuinely negative,
    so 0 (this position needs nothing) is the correct floor, not a
    guess."""
    need = lineup_slots.get(position, 0)
    flex = lineup_slots.get("FLEX", 0)
    if position in FLEX_ELIGIBLE_POSITIONS and flex:
        need += flex / len(FLEX_ELIGIBLE_POSITIONS)
    return max(0.0, need)


def scarcity_multiplier(lineup_slots: dict[str, int], rostered: list[Player], position: str) -> tuple[float, float, int]:
    """>1 when you're thinner at this position than your league starts,
    <1 when you're deeper than you need. Laplace-smoothed (+1 on both
    sides) so a position with zero rostered players gets a large but
    finite boost instead of dividing by zero."""
    need = position_need(lineup_slots, position)
    rostered_count = sum(1 for p in rostered if p.position == position)
    multiplier = (need + 1) / (rostered_count + 1)
    return multiplier, need, rostered_count


def bye_week_collision_note(rostered: list[Player], position: str, bye_week: int | None) -> str | None:
    if bye_week is None:
        return None
    colliding = [p for p in rostered if p.position == position and p.bye_week == bye_week]
    if not colliding:
        return None
    names = ", ".join(p.name for p in colliding)
    return f"Bye week collision: shares week {bye_week} bye with your rostered {position} {names}."


def apply_roster_fit(
    candidates: list[RankedFreeAgent],
    league_config: LeagueConfig,
    rostered: list[Player],
    bye_weeks_by_player_id: dict[str, int | None],
) -> list[RosterFitCandidate]:
    results = []
    for c in candidates:
        multiplier, need, rostered_count = scarcity_multiplier(league_config.lineup_slots, rostered, c.position)
        bye_week = bye_weeks_by_player_id.get(c.player_id)
        bye_note = bye_week_collision_note(rostered, c.position, bye_week)

        final_explanation = c.explanation
        if bye_note:
            final_explanation = f"{final_explanation} {bye_note}"

        results.append(
            RosterFitCandidate(
                candidate=c,
                roster_fit_score=c.predicted_score * multiplier,
                scarcity_multiplier=multiplier,
                position_need=need,
                rostered_count_at_position=rostered_count,
                bye_week_collision=bye_note,
                final_explanation=final_explanation,
            )
        )

    results.sort(key=lambda r: r.roster_fit_score, reverse=True)
    return results


ACTIONABLE_TIERS = ("High", "Medium")


def select_visible(
    results: list[RosterFitCandidate], top: int | None = None, show_all: bool = False
) -> tuple[list[RosterFitCandidate], int]:
    """Returns (rows to print, how many were omitted).

    Default is the actionable tiers only. A real run against a live ESPN
    league produced 293 ranked candidates -- 5 High, 38 Medium, 249 Low
    -- so printing everything means 85% of the weekly output is the tool's
    own "Low confidence, $1 speculative claim" tier scrolling past the
    handful of names actually worth a bid. That's the precision-over-recall
    philosophy the PRD picked (a short, high-trust list), applied to the
    output and not just the model."""
    if show_all:
        return results, 0
    if top is not None:
        return results[:top], max(0, len(results) - top)
    visible = [r for r in results if r.candidate.confidence_tier in ACTIONABLE_TIERS]
    return visible, len(results) - len(visible)


def _print_table(results: list[RosterFitCandidate], omitted: int = 0) -> None:
    if not results:
        if omitted:
            print(f"(no High/Medium-confidence candidates this week — {omitted} Low-tier candidates hidden; --all to see them)")
        else:
            print("(no free-agent candidates to rank)")
        return
    # The probability column only appears once a calibration artifact has
    # been fitted, so an un-calibrated checkout sees exactly the table it
    # always did rather than a column of blanks.
    show_probability = any(r.candidate.hit_probability is not None for r in results)
    prob_header = f"{'P(hit)':<8}" if show_probability else ""
    header = (
        f"{'Rank':<5}{'Player':<22}{'Pos':<5}{'Team':<6}"
        f"{'Opp.Score':<11}{'Fit x':<7}{'Final':<8}{prob_header}{'Tier':<8}Explanation"
    )
    print(header)
    print("-" * len(header))
    for i, r in enumerate(results, start=1):
        c = r.candidate
        prob = ""
        if show_probability:
            prob = f"{c.hit_probability:<8.0%}" if c.hit_probability is not None else f"{'-':<8}"
        print(
            f"{i:<5}{c.name:<22}{c.position:<5}{c.team:<6}"
            f"{c.predicted_score:<11.1f}{r.scarcity_multiplier:<7.2f}{r.roster_fit_score:<8.1f}"
            f"{prob}{c.confidence_tier:<8}{r.final_explanation}"
        )
    if omitted:
        print(f"\n({omitted} lower-confidence candidate(s) hidden — use --all to see the full list)")


def _print_drops(drop_candidates: list, swaps: list, show_all: bool) -> None:
    """The drop side of the decision. Printed after the adds because it's
    only actionable once you know what you're claiming."""
    from edge_engine.ranking.drop import droppable

    if not drop_candidates:
        return

    print("\n--- Who to drop ---")
    available = droppable(drop_candidates)
    if not available:
        print("(no safe drop: every rostered player is protected — see below)")
    else:
        shown = available if show_all else available[:5]
        header = f"{'Player':<22}{'Pos':<5}{'Value':<8}Why"
        print(header)
        print("-" * len(header))
        for c in shown:
            print(f"{c.player.name:<22}{c.player.position:<5}{c.roster_value:<8.1f}{c.explanation}")
        if len(shown) < len(available):
            print(f"  ...and {len(available) - len(shown)} more droppable (--all to see them)")

    protected = [c for c in drop_candidates if c.is_protected]
    if protected and show_all:
        print("\nProtected (never recommended):")
        for c in protected:
            print(f"  {c.player.name} ({c.player.position}) — {c.protected_reason}")
    elif protected:
        print(f"\n({len(protected)} player(s) protected from dropping — --all to see why)")

    if swaps:
        print("\nSuggested swaps:")
        for s in swaps:
            verdict = f"+{s.gain:.1f}" if s.is_upgrade else f"{s.gain:.1f} — NOT an upgrade"
            print(f"  Claim {s.add_name} ({s.add_value:.1f}) for {s.drop_name} ({s.drop_value:.1f}): {verdict}")


def _print_unresolved(unresolved: list[Player], show_all: bool) -> None:
    """Unresolved free agents are surfaced, never silently dropped (a
    standing requirement) -- but a live ESPN league produced 25 of them,
    mostly practice-squad names nflverse has no usage data for at all.
    Collapsed to a count by default so it doesn't bury the actual
    rankings; --all prints every one."""
    if not unresolved:
        return
    print(f"\n{len(unresolved)} free agent(s) did not resolve and are excluded above — check spelling/team:")
    shown = unresolved if show_all else unresolved[:5]
    for p in shown:
        print(f"  {p.name} ({p.position}, {p.team}) [{p.match_status}: {p.match_note}]")
    if len(shown) < len(unresolved):
        print(f"  ...and {len(unresolved) - len(shown)} more (--all to see them)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--top", type=int, default=None,
        help="show the top N candidates by roster-fit score, regardless of confidence tier",
    )
    group.add_argument(
        "--all", action="store_true",
        help="show every ranked candidate (default: High/Medium-confidence tiers only)",
    )
    args = parser.parse_args(argv)

    config = load_model_config()
    source = get_default_source()
    meta = source.get_roster_meta()
    league_config = source.get_league_config()
    rostered = source.get_rostered_players()

    staleness = (date.today() - meta.as_of_date).days
    print(f"Roster data as of week {meta.as_of_week} ({meta.as_of_date})", end="")
    print(f"  ⚠ {staleness} days old — update roster_state/ before trusting this" if staleness > 7 else "")
    print()

    # Scored once and shared: the free-agent ranking and the drop
    # recommendation need the same frame, and scoring twice would double
    # the slowest part of this command for no benefit.
    from edge_engine.model.predict import score_latest_week
    from edge_engine.ranking.drop import rank_drop_candidates, suggest_swaps

    scored = score_latest_week(config)

    candidates, unresolved = build_free_agent_rankings(config, scored=scored)
    # Bye weeks come from the same resolved Player records
    # build_free_agent_rankings already matched against.
    bye_weeks_by_player_id = {
        p.player_id: p.bye_week for p in source.get_free_agents() if p.player_id
    }

    results = apply_roster_fit(candidates, league_config, rostered, bye_weeks_by_player_id)
    visible, omitted = select_visible(results, top=args.top, show_all=args.all)
    _print_table(visible, omitted)

    drop_candidates = rank_drop_candidates(rostered, scored, league_config)
    swaps = suggest_swaps(visible, drop_candidates)
    _print_drops(drop_candidates, swaps, show_all=args.all)

    _print_unresolved(unresolved, show_all=args.all)


if __name__ == "__main__":
    main()
