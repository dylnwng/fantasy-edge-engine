"""The draft board: players priced at ADP, grouped into tiers, and tagged
where last season's late usage disagrees with the market.

**This board makes no claim to beat ADP.** It prices everything at the
market number, on purpose. Its value is bookkeeping under a 90-second
clock -- how many players are left in a tier, whether a positional run is
underway -- which software wins for free and humans reliably get wrong.
That value does not depend on a projection model, which is why this
ships without a kill gate while custom preseason projections (5-full) do
not ship at all until they beat ADP on a held-out season.

**The one genuine edge claim, and it is narrow.** A player whose usage
spiked over weeks 13-17 last season is priced at an ADP reflecting his
*full-season* average. That is a usage-versus-market divergence, it is
computable from data already in this repo, and it needs no new model.
It is a tag on the board, never a re-ranking: the board stays in ADP
order and the tag tells you where the market may be looking at the
wrong window.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from edge_engine.draft.market import MarketPrice
from edge_engine.insights.divergence import compute_divergence

# Weeks that make up "late last season". The stretch where a role change
# happens late enough that a full-season average -- which is what ADP
# reflects -- still averages it away.
LATE_SEASON_WEEKS = (13, 14, 15, 16, 17)

# A tier break is declared where the ADP gap to the next player at the
# same position exceeds this many picks. A heuristic, deliberately not
# tuned: per EVALUATION.md, threshold sweeps on a few hundred rows
# manufacture gains that invert next season, and there is no held-out
# draft to tune against anyway.
TIER_GAP_PICKS = 12.0

# Positions with no player-level usage data in nflverse at all, so they can
# never carry a buy-low/sell-high tag. They still belong ON the board -- you
# draft them -- but their untagged state is by design, not a data problem.
_NO_USAGE_DATA_POSITIONS = frozenset({"K", "DST"})


@dataclass(frozen=True)
class BoardPlayer:
    name: str
    position: str
    team: str
    adp: float
    position_rank: int          # 1 = first player at this position off the board
    tier: int                   # 1 = best tier at this position
    player_id: str | None = None
    divergence: float | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)
    low_confidence: bool = False


def assign_tiers(prices: list[MarketPrice], gap: float = TIER_GAP_PICKS) -> dict[str, list[tuple[MarketPrice, int, int]]]:
    """Group each position's players into tiers by ADP gap.

    Returns position -> [(price, position_rank, tier)], ADP-ascending. A
    tier is just "players the market prices similarly"; the break points
    are where the market itself leaves a hole."""
    by_position: dict[str, list[MarketPrice]] = {}
    for p in prices:
        by_position.setdefault(p.position, []).append(p)

    out: dict[str, list[tuple[MarketPrice, int, int]]] = {}
    for position, players in by_position.items():
        players = sorted(players, key=lambda p: p.adp)
        tier = 1
        rows: list[tuple[MarketPrice, int, int]] = []
        for rank, player in enumerate(players, start=1):
            if rank > 1 and (player.adp - players[rank - 2].adp) > gap:
                tier += 1
            rows.append((player, rank, tier))
        out[position] = rows
    return out


def late_season_divergence(
    player_week: pd.DataFrame,
    points: pd.Series,
    season: int,
    weeks: tuple[int, ...] = LATE_SEASON_WEEKS,
) -> tuple[dict[str, float], list[str]]:
    """player_id -> divergence over the late-season stretch of `season`.

    Reuses the Phase 3 metric rather than defining a second one: the
    question ("is usage ahead of production") is identical, only the
    window differs."""
    last_week = max(weeks)
    window = len(weeks)
    results, gaps = compute_divergence(player_week, points, season=season, week=last_week, window=window)
    return {d.player_id: d.divergence for d in results}, gaps


def build_board(
    prices: list[MarketPrice],
    divergence_by_id: dict[str, float] | None = None,
    rookie_ids: set[str] | None = None,
    divergence_threshold: float = 1.0,
) -> tuple[list[BoardPlayer], list[str]]:
    """Returns (board in ADP order, data_gaps).

    `divergence_by_id` is optional -- with it absent the board is a pure
    ADP board, which is still the useful thing. That is the whole point
    of 5-lite: it degrades to well-organised market pricing rather than
    to nothing."""
    gaps: list[str] = []
    divergence_by_id = divergence_by_id or {}
    rookie_ids = rookie_ids or set()

    tiers = assign_tiers(prices)
    board: list[BoardPlayer] = []

    unresolved = 0
    for position, rows in tiers.items():
        for price, rank, tier in rows:
            tags: list[str] = []
            divergence = None
            low_confidence = False

            if price.player_id is None:
                unresolved += 1
                low_confidence = True
                if position in _NO_USAGE_DATA_POSITIONS:
                    tags.append("ranked by ADP — this position isn't modeled")
                else:
                    tags.append("no usage history matched — priced at ADP only")
            elif price.player_id in rookie_ids:
                low_confidence = True
                tags.append("rookie — no NFL usage history, priced at ADP only")
            else:
                divergence = divergence_by_id.get(price.player_id)
                if divergence is None:
                    low_confidence = True
                    tags.append("no late-season usage data — priced at ADP only")
                elif divergence > divergence_threshold:
                    tags.append(
                        f"late-season usage ran ahead of production ({divergence:+.1f} SD) — "
                        "ADP reflects his full-season average"
                    )
                elif divergence < -divergence_threshold:
                    tags.append(
                        f"late-season production ran ahead of usage ({divergence:+.1f} SD) — "
                        "ADP may be pricing a role he didn't hold"
                    )

            board.append(
                BoardPlayer(
                    name=price.name, position=position, team=price.team, adp=price.adp,
                    position_rank=rank, tier=tier, player_id=price.player_id,
                    divergence=divergence, tags=tuple(tags), low_confidence=low_confidence,
                )
            )

    if unresolved:
        # Kickers and team defenses have no player-level usage data at all
        # -- they can never carry a tag, and saying "check the spelling"
        # about them is actively misleading advice. Separate the expected
        # case from the one actually worth looking into.
        no_stats_by_design = sum(
            1 for p in prices if p.player_id is None and p.position in _NO_USAGE_DATA_POSITIONS
        )
        worth_checking = unresolved - no_stats_by_design

        if no_stats_by_design:
            gaps.append(
                f"{no_stats_by_design} kicker/defense entries carry no buy-low or sell-high tag. "
                "That's expected — there's no snap or target data for those positions."
            )
        if worth_checking:
            gaps.append(
                f"{worth_checking} player(s) couldn't be matched to stats, so they're priced at "
                "ADP only. Usually rookies with no NFL history, or two players sharing a name."
            )

    board.sort(key=lambda b: b.adp)
    return board, gaps
