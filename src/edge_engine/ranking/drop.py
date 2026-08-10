"""The other half of a waiver claim: who comes off the roster.

Every add requires a drop, and the tool has always ranked adds while
saying nothing about the drop -- leaving the actual weekly decision
half-answered. This closes that.

Valuation is deliberately NOT a second opinion. A rostered player is
scored with exactly the quantity roster_fit gives a free agent
(`predicted_score * scarcity_multiplier`), so adds and drops sit on one
comparable scale and "is this swap an upgrade" becomes a real question
instead of a comparison between two unrelated numbers. Reusing
roster_fit's own scarcity function rather than copying its formula also
means the two can't drift apart.

This is roster management, not a lineup optimizer -- it says who is
least valuable to own, never who to start. Start/sit is Phase 2's job
and stays there.

Two things it will not do:

  * **Value a player it has no data on.** An unresolved name, or a player
    without a full trailing window, gets no score and is never
    recommended. Recommending a drop on the basis of missing data is
    exactly the failure mode bye_weeks.py refuses -- and here it would
    cost you a real player.
  * **Break your starting lineup.** Dropping your only quarterback is
    wrong at any score, so a position that would fall below the league's
    dedicated starting slots is protected structurally rather than by
    hoping the score sorts it out.

Injury and roster-status context (IR, suspended, PUP) is attached as
text and never moves the ranking, matching the invariant the rest of the
project holds: a suspended player may well be your best drop, but that's
your call to make with the fact in front of you, not something the score
decides silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from edge_engine.ranking.roster_fit import position_need, scarcity_multiplier
from edge_engine.roster.models import LeagueConfig, Player

NO_DATA_REASON = "no usage data — not enough games played, or the name didn't resolve"


@dataclass(frozen=True)
class DropCandidate:
    player: Player
    # None when the model has no row for this player. Kept as None rather
    # than 0.0 so "we don't know" can never be mistaken for "worthless".
    predicted_score: float | None
    scarcity_multiplier: float
    roster_value: float | None
    rostered_count_at_position: int
    position_need: float
    protected_reason: str | None
    explanation: str

    @property
    def is_protected(self) -> bool:
        return self.protected_reason is not None


def _required_starters(lineup_slots: dict[str, int], position: str) -> int:
    """Dedicated starting slots at `position`, ignoring FLEX.

    FLEX is excluded on purpose. position_need() spreads a FLEX slot
    fractionally across RB/WR/TE for *valuation*, which is right for
    ranking but wrong for a hard structural rule -- you must be able to
    field a starter at every dedicated slot regardless of how the FLEX
    gets filled, and a fractional requirement can't express that.
    """
    return max(0, int(lineup_slots.get(position, 0)))


def _protection_reason(
    lineup_slots: dict[str, int], position: str, rostered_count: int
) -> str | None:
    """Why this player must not be dropped, or None if he may be."""
    required = _required_starters(lineup_slots, position)
    if rostered_count - 1 < required:
        return (
            f"dropping would leave {rostered_count - 1} {position}(s) "
            f"for {required} starting slot(s)"
        )
    return None


def _context_notes(row: pd.Series | None) -> list[str]:
    """Injury / roster-status text carried through from the scored frame.

    Surfaced, never scored -- see the module docstring.
    """
    if row is None:
        return []
    notes = []
    status = row.get("roster_status_note")
    if status is not None and pd.notna(status) and str(status).strip():
        notes.append(str(status))
    if bool(row.get("has_injury_context")):
        explanation = row.get("injury_explanation")
        if explanation is not None and pd.notna(explanation) and str(explanation).strip():
            notes.append(str(explanation))
    return notes


def rank_drop_candidates(
    rostered: list[Player],
    scored: pd.DataFrame,
    league_config: LeagueConfig,
) -> list[DropCandidate]:
    """Your roster ordered by how droppable each player is, least
    valuable first.

    Protected players (structurally undroppable, or unvaluable for want
    of data) always sort last, whatever their score, so the top of the
    list is only ever players it is actually safe to act on.
    """
    by_id: dict[str, pd.Series] = {}
    if not scored.empty and "player_id" in scored.columns:
        # tail(1) rather than assuming uniqueness: score_as_of_week unions
        # two models, and a duplicate id would otherwise raise here rather
        # than in the place that could do something about it.
        for player_id, group in scored.groupby("player_id"):
            by_id[player_id] = group.iloc[-1]

    candidates: list[DropCandidate] = []
    for player in rostered:
        multiplier, need, rostered_count = scarcity_multiplier(
            league_config.lineup_slots, rostered, player.position
        )
        row = by_id.get(player.player_id) if player.player_id else None

        predicted = None
        if row is not None:
            raw = row.get("predicted_score")
            if raw is not None and pd.notna(raw):
                predicted = float(raw)

        protected = _protection_reason(league_config.lineup_slots, player.position, rostered_count)
        if predicted is None and protected is None:
            protected = NO_DATA_REASON

        value = predicted * multiplier if predicted is not None else None
        notes = _context_notes(row)

        parts: list[str] = []
        if protected:
            parts.append(f"Protected: {protected}.")
        elif value is not None:
            parts.append(
                f"Worth {value:.1f} to you "
                f"({predicted:.1f} projected x {multiplier:.2f} for {rostered_count} "
                f"rostered {player.position} vs {need:.1f} needed)."
            )
        parts.extend(notes)

        candidates.append(
            DropCandidate(
                player=player,
                predicted_score=predicted,
                scarcity_multiplier=multiplier,
                roster_value=value,
                rostered_count_at_position=rostered_count,
                position_need=need,
                protected_reason=protected,
                explanation=" ".join(parts),
            )
        )

    # Protected last; then least valuable first. Unvaluable players are
    # protected by construction above, so `roster_value` is never None in
    # the unprotected group and the sort key can't compare None to float.
    candidates.sort(key=lambda c: (c.is_protected, c.roster_value if c.roster_value is not None else 0.0))
    return candidates


def droppable(candidates: list[DropCandidate]) -> list[DropCandidate]:
    """Only the players it is actually safe to drop."""
    return [c for c in candidates if not c.is_protected]


@dataclass(frozen=True)
class Swap:
    add_name: str
    add_value: float
    drop_name: str
    drop_value: float

    @property
    def gain(self) -> float:
        return self.add_value - self.drop_value

    @property
    def is_upgrade(self) -> bool:
        return self.gain > 0


def suggest_swaps(
    ranked_adds: list, drop_candidates: list[DropCandidate], limit: int = 3
) -> list[Swap]:
    """Pair the best available adds against the most droppable players.

    `ranked_adds` are roster_fit's RosterFitCandidates, already sorted.
    Both sides are measured with the same `predicted_score * scarcity`
    quantity, so the comparison is meaningful rather than notional.

    One simplification, stated rather than hidden: scarcity is computed
    against the roster as it stands now, so a swap that changes the count
    at a position shifts the multiplier slightly for anything after it.
    Each pairing is evaluated independently and they are not intended to
    be applied all at once -- they're candidate moves, in order.
    """
    available = droppable(drop_candidates)
    swaps: list[Swap] = []
    for add, drop in zip(ranked_adds[:limit], available[:limit]):
        swaps.append(
            Swap(
                add_name=add.candidate.name,
                add_value=float(add.roster_fit_score),
                drop_name=drop.player.name,
                drop_value=float(drop.roster_value),
            )
        )
    return swaps
