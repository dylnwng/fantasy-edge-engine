"""Market price (ADP) behind a Protocol, mirroring RosterStateSource.

Why a Protocol for one CSV: ADP is the first non-nflverse dependency in
this project, and the PRD is explicit that it must survive its source
changing or dying in August -- which is exactly when it would happen and
exactly when there is no time to fix it. So the board depends on
`MarketPriceSource`, never on a file format or a vendor.

**On live ADP APIs.** Checked while building this: Sleeper's public
player endpoint (`api.sleeper.app/v1/players/nfl`, no auth) works and
usefully carries `gsis_id` -- the same join key this project already
uses -- plus depth-chart position, but it does not expose aggregate ADP.
FantasyPros and Underdog gate theirs behind keys. Since ADP is a
once-a-year input that a human exports in August rather than a live
feed, a CSV is not a degraded fallback here, it is the appropriate
primary implementation. A future `SleeperMarketPriceSource` can be added
behind this same Protocol without touching the board.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from edge_engine.paths import DATA_DIR
from edge_engine.roster.player_lookup import PlayerLookup

DRAFT_DIR = DATA_DIR / "draft"
DEFAULT_ADP_PATH = DRAFT_DIR / "adp.csv"


@dataclass(frozen=True)
class MarketPrice:
    name: str
    position: str
    team: str
    adp: float
    player_id: str | None = None  # resolved nflverse gsis_id, if matched
    match_status: str = "unmatched"
    match_note: str = ""


class MarketPriceSource(Protocol):
    def get_market_prices(self) -> list[MarketPrice]: ...

    def describe(self) -> str:
        """Human-readable provenance, rendered on the board. A draft board
        priced at an unknown vintage of ADP is a trap."""
        ...


class ManualMarketPriceSource:
    """Reads `data/draft/adp.csv`: columns `name,position,team,adp`.

    Export from wherever you like (FantasyPros, Sleeper, Underdog) -- the
    board only cares about the four columns. Names are resolved to
    nflverse ids through the same PlayerLookup the roster sources use, so
    a player's ADP and his usage history join correctly."""

    def __init__(self, path: Path | None = None, lookup: PlayerLookup | None = None):
        self.path = path or DEFAULT_ADP_PATH
        self._lookup = lookup

    def _get_lookup(self) -> PlayerLookup:
        if self._lookup is None:
            self._lookup = PlayerLookup.build()
        return self._lookup

    def describe(self) -> str:
        return f"ADP from {self.path}"

    def get_market_prices(self) -> list[MarketPrice]:
        try:
            with open(self.path, newline="") as f:
                rows = list(csv.DictReader(f))
        except FileNotFoundError as e:
            raise RuntimeError(
                f"{self.path} doesn't exist. The draft board needs an ADP export with "
                "columns: name,position,team,adp — see data/draft/README.md."
            ) from e

        lookup = self._get_lookup()
        prices: list[MarketPrice] = []
        for i, row in enumerate(rows):
            try:
                name = row["name"].strip()
                position = row["position"].strip().upper()
                team = row["team"].strip().upper()
                raw_adp = row["adp"].strip()
            except KeyError as e:
                raise RuntimeError(
                    f"{self.path}: missing required column {e.args[0]!r} (row {i + 2}). "
                    "Expected columns: name,position,team,adp"
                ) from e

            try:
                adp = float(raw_adp)
            except ValueError as e:
                raise RuntimeError(
                    f"{self.path}: adp={raw_adp!r} on row {i + 2} isn't a number."
                ) from e
            if adp <= 0:
                raise RuntimeError(f"{self.path}: adp={adp} on row {i + 2} must be positive.")

            gsis_id, status, note = lookup.resolve(name, position, team)
            prices.append(
                MarketPrice(
                    name=name, position=position, team=team, adp=adp,
                    player_id=gsis_id, match_status=status, match_note=note,
                )
            )

        if not prices:
            raise RuntimeError(f"{self.path} has no rows; the draft board would be empty.")
        return prices
