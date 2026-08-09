"""Live draft state: who's gone, what's left in each tier, and whether a
run is underway.

This is the part of the draft tool that actually earns its place. Under a
90-second clock a human cannot reliably track that RB tier 3 has two
players left, or that four of the last six picks were receivers. Software
does it for free, and none of it requires a projection model -- which is
why this ships regardless of whether any model validates.

Pure state, no I/O. A pick set is a **set keyed by (round, pick)**, not an
append-only list, because a poll against a real draft API will hand you
duplicates and out-of-order picks and you reconcile rather than
accumulate. That decision belongs here rather than in whatever transport
eventually feeds it.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field, replace

from edge_engine.draft.board import BoardPlayer


@dataclass(frozen=True)
class Pick:
    round_number: int
    pick_number: int
    player_name: str

    @property
    def key(self) -> tuple[int, int]:
        return (self.round_number, self.pick_number)


@dataclass(frozen=True)
class TierRemaining:
    position: str
    tier: int
    remaining: int
    names: tuple[str, ...]


@dataclass(frozen=True)
class DraftState:
    board: list[BoardPlayer]
    picks: dict[tuple[int, int], Pick] = field(default_factory=dict)

    def apply(self, pick: Pick) -> "DraftState":
        """Idempotent by (round, pick): re-applying the same slot replaces
        rather than double-counts. A poll that returns the same pick twice
        must not remove two players from the pool."""
        return replace(self, picks={**self.picks, pick.key: pick})

    def apply_all(self, picks: list[Pick]) -> "DraftState":
        state = self
        for p in picks:
            state = state.apply(p)
        return state

    @property
    def drafted_names(self) -> set[str]:
        return {p.player_name for p in self.picks.values()}

    @property
    def picks_in_order(self) -> list[Pick]:
        return [self.picks[k] for k in sorted(self.picks)]

    def available(self) -> list[BoardPlayer]:
        gone = self.drafted_names
        return [b for b in self.board if b.name not in gone]

    def best_available(self, n: int = 3, position: str | None = None) -> list[BoardPlayer]:
        pool = self.available()
        if position:
            pool = [b for b in pool if b.position == position]
        return pool[:n]

    def tiers_remaining(self, position: str | None = None) -> list[TierRemaining]:
        """Per position and tier, who's left. Sorted so the tier about to
        run out sits at the top -- the single highest-value thing on a
        draft screen, because it is what tells you to reach now or wait."""
        buckets: dict[tuple[str, int], list[BoardPlayer]] = {}
        for b in self.available():
            if position and b.position != position:
                continue
            buckets.setdefault((b.position, b.tier), []).append(b)

        out = [
            TierRemaining(
                position=pos, tier=tier, remaining=len(players),
                names=tuple(p.name for p in sorted(players, key=lambda x: x.adp)),
            )
            for (pos, tier), players in buckets.items()
        ]
        # Thinnest tier first, then by position and tier for stability.
        out.sort(key=lambda t: (t.remaining, t.position, t.tier))
        return out

    def positional_run(self, lookback: int = 6, threshold: int = 4) -> str | None:
        """Descriptive only: "4 of the last 6 picks were WR". No
        recommendation is attached, deliberately -- whether a run means
        chase it or fade it depends on your roster, and the tool doesn't
        model that."""
        recent = self.picks_in_order[-lookback:]
        if len(recent) < lookback:
            return None

        by_name = {b.name: b for b in self.board}
        positions = [by_name[p.player_name].position for p in recent if p.player_name in by_name]
        if not positions:
            return None

        position, count = Counter(positions).most_common(1)[0]
        if count >= threshold:
            return f"{count} of the last {len(recent)} picks were {position}"
        return None


def unfilled_slots(
    my_picks: list[BoardPlayer], lineup_slots: dict[str, int]
) -> dict[str, int]:
    """Starting slots you still need to fill. FLEX is reported separately
    rather than being auto-assigned, since which player fills it is a
    decision the tool doesn't make."""
    have = Counter(p.position for p in my_picks)
    unfilled: dict[str, int] = {}
    for slot, required in lineup_slots.items():
        upper = slot.upper()
        if upper in {"BENCH", "IR"} or required <= 0:
            continue
        if upper == "FLEX":
            unfilled["FLEX"] = required
            continue
        short = required - have.get(slot, 0)
        if short > 0:
            unfilled[slot] = short
    return unfilled
