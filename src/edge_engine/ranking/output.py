"""Requirement 4: free-agent ranking output.

Cross-references the opportunity model's ranked scores (requirement 3,
including requirement 3b's injury context) against the free-agent pool
via the roster-state interface (requirement 1) -- never a direct ESPN
call, and never reading roster files directly; only get_default_source().

Usage:
    python -m edge_engine.ranking.output
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from edge_engine.model.config import ModelConfig, load_model_config
from edge_engine.model.predict import score_latest_week
from edge_engine.paths import PLAYER_WEEK_PATH
from edge_engine.ranking.usage_trend import usage_trend_explanation
from edge_engine.roster.interface import get_default_source
from edge_engine.roster.models import LeagueConfig, Player

ConfidenceTier = str  # "High" | "Medium" | "Low"


@dataclass(frozen=True)
class RankedFreeAgent:
    player_id: str
    name: str
    position: str
    team: str
    season: int
    week: int  # the exact scored week the score/explanation are computed from
    predicted_score: float
    baseline_score: float
    confidence_tier: ConfidenceTier
    explanation: str


def confidence_tier(predicted_score: float, baseline_score: float, flag_margin: float) -> ConfidenceTier:
    """A rough guide only, per the PRD -- not a dollar amount. High = worth
    a real FAAB bid; Medium = worth a look; Low = $1 speculative claim."""
    margin = predicted_score - baseline_score
    if margin >= 2 * flag_margin:
        return "High"
    if margin >= flag_margin:
        return "Medium"
    return "Low"


def _explanation_for(player_week: pd.DataFrame, row: pd.Series, window: int) -> str:
    # Quarterbacks arrive from a separate model and carry their own
    # volume-based explanation. usage_trend_explanation reads snap and
    # target share, which are structurally null for a passer -- it would
    # report "usage steady, no metric moved meaningfully" for every QB
    # regardless of what his workload actually did.
    qb_explanation = getattr(row, "usage_explanation", None)
    if qb_explanation is not None and pd.notna(qb_explanation) and str(qb_explanation).strip():
        trend = str(qb_explanation)
    else:
        trend = usage_trend_explanation(player_week, row.player_id, int(row.season), int(row.week), window)

    parts = [trend]
    if row.has_injury_context:
        parts.append(row.injury_explanation)
    # roster_status_note is None for the common case, but pandas coerces
    # a mostly-None object column to NaN on assignment (see predict.py) --
    # NaN is truthy in Python, so a plain `if row.roster_status_note`
    # check would append the literal string "nan" to every explanation.
    if pd.notna(row.roster_status_note):
        parts.append(row.roster_status_note)
    return _join_sentences(parts)


def _join_sentences(parts: list[str]) -> str:
    """Join explanation fragments into readable prose.

    Some fragments already end in a full stop and some don't, so naively
    joining with ". " produced doubled periods in the UI ("...show a
    trend.. Usage spike coincides..."). Normalise each fragment's ending
    before joining rather than leaving the caller to guess."""
    cleaned = [p.strip().rstrip(".") for p in parts if p and str(p).strip()]
    return ". ".join(cleaned) + "." if cleaned else ""


_FLEX_ELIGIBLE = {"RB", "WR", "TE"}
_NON_STARTING_SLOTS = {"BENCH", "IR", "FLEX"}


def _startable_positions(league_config: LeagueConfig) -> set[str]:
    """Positions this league actually starts.

    A player's nflverse position can legitimately differ from the one his
    fantasy platform lists: Bo Melton is carried as a WR by ESPN (so he's
    in the free-agent pool) but nflverse correctly has him as a DB after
    Green Bay converted him to cornerback. Without this filter he shows up
    as a defensive back in a standard league's waiver recommendations,
    which reads as a bug even though the data is right.

    Derived from the league's own lineup slots rather than hardcoded, so
    an IDP league that genuinely starts defensive backs keeps them."""
    positions = {
        slot for slot, count in league_config.lineup_slots.items()
        if count > 0 and slot.upper() not in _NON_STARTING_SLOTS
    }
    if league_config.lineup_slots.get("FLEX", 0) > 0:
        positions |= _FLEX_ELIGIBLE
    return positions


def build_free_agent_rankings(config: ModelConfig | None = None) -> tuple[list[RankedFreeAgent], list[Player]]:
    """Returns (ranked candidates sorted by predicted_score desc, unresolved
    free agents that couldn't be matched to nflverse data -- surfaced, not
    silently dropped from the ranking)."""
    config = config or load_model_config()
    source = get_default_source()
    free_agents = source.get_free_agents()

    resolved = {p.player_id: p for p in free_agents if p.player_id}
    unresolved = [p for p in free_agents if not p.player_id]

    scored = score_latest_week(config)
    player_week = pd.read_parquet(PLAYER_WEEK_PATH)

    pool = scored[scored["player_id"].isin(resolved.keys())].copy()
    pool = pool[pool["position"].isin(_startable_positions(source.get_league_config()))]

    candidates = []
    for row in pool.itertuples():
        player = resolved[row.player_id]
        candidates.append(
            RankedFreeAgent(
                player_id=row.player_id,
                name=player.name,
                position=row.position,
                team=row.team,
                season=int(row.season),
                week=int(row.week),
                predicted_score=float(row.predicted_score),
                baseline_score=float(row.baseline_score),
                confidence_tier=confidence_tier(row.predicted_score, row.baseline_score, config.flag_margin),
                explanation=_explanation_for(player_week, row, config.trailing_window),
            )
        )

    candidates.sort(key=lambda c: c.predicted_score, reverse=True)
    return candidates, unresolved


def _print_table(candidates: list[RankedFreeAgent]) -> None:
    if not candidates:
        print("(no free-agent candidates have a valid score this week)")
        return
    header = f"{'Rank':<5}{'Player':<22}{'Pos':<5}{'Team':<6}{'Score':<8}{'Tier':<8}Explanation"
    print(header)
    print("-" * len(header))
    for i, c in enumerate(candidates, start=1):
        print(f"{i:<5}{c.name:<22}{c.position:<5}{c.team:<6}{c.predicted_score:<8.1f}{c.confidence_tier:<8}{c.explanation}")


def main() -> None:
    config = load_model_config()
    source = get_default_source()
    meta = source.get_roster_meta()

    staleness = (date.today() - meta.as_of_date).days
    print(f"Roster data as of week {meta.as_of_week} ({meta.as_of_date})", end="")
    print(f"  ⚠ {staleness} days old — update roster_state/ before trusting this" if staleness > 7 else "")
    print()

    candidates, unresolved = build_free_agent_rankings(config)
    _print_table(candidates)

    if unresolved:
        print(f"\n{len(unresolved)} free agent(s) did not resolve and are excluded above — check spelling/team:")
        for p in unresolved:
            print(f"  {p.name} ({p.position}, {p.team}) [{p.match_status}: {p.match_note}]")


if __name__ == "__main__":
    main()
