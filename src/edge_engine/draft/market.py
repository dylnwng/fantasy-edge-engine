"""Market price (ADP) behind a Protocol, mirroring RosterStateSource.

Why a Protocol for one CSV: ADP is the first non-nflverse dependency in
this project, and the PRD is explicit that it must survive its source
changing or dying in August -- which is exactly when it would happen and
exactly when there is no time to fix it. So the board depends on
`MarketPriceSource`, never on a file format or a vendor.

**On live ADP APIs.** Fantasy Football Calculator publishes a documented
public JSON endpoint (`/api/v1/adp/<format>?teams=N&year=YYYY`, no key,
no auth) with real aggregate ADP from actual mock drafts -- that's
`FantasyFootballCalculatorSource` below, and it's the default. Sleeper's
public player endpoint carries `gsis_id` and depth charts but no
aggregate ADP; FantasyPros and Underdog gate theirs behind keys.

The CSV implementation remains, because the PRD's requirement stands:
this is the project's only non-nflverse dependency and it must survive
its source changing or dying in August, which is exactly when it would
happen and exactly when there's no time to fix it. If the API is
unreachable the board falls back to `adp.csv` and says so.
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


# League scoring -> the format slug FFC's API expects.
_FFC_FORMAT = {"standard": "standard", "half_ppr": "half-ppr", "full_ppr": "ppr"}
_FFC_TEAM_SIZES = (8, 10, 12, 14)
FFC_URL = "https://fantasyfootballcalculator.com/api/v1/adp/{fmt}?teams={teams}&year={year}"

# ADP moves over days, not minutes. Cache so a dashboard that reruns on
# every widget click doesn't hammer someone else's free API.
CACHE_TTL_HOURS = 12


class FantasyFootballCalculatorSource:
    """Live ADP from Fantasy Football Calculator's public JSON API.

    Real aggregate ADP from actual mock drafts, no key required. Matched
    to your league's scoring so a standard league isn't priced off PPR
    boards.

    Cached to disk for `CACHE_TTL_HOURS`; on any network failure it
    raises with a message pointing at the CSV fallback rather than
    silently serving a stale or empty board."""

    def __init__(
        self,
        year: int,
        ppr_type: str = "full_ppr",
        teams: int = 12,
        lookup: PlayerLookup | None = None,
        cache_dir: Path | None = None,
    ):
        if ppr_type not in _FFC_FORMAT:
            raise RuntimeError(
                f"ADP isn't available for scoring type {ppr_type!r} "
                f"(expected one of {sorted(_FFC_FORMAT)})."
            )
        # FFC only publishes these league sizes; snap to the nearest
        # rather than 404 on an unusual one.
        self.teams = min(_FFC_TEAM_SIZES, key=lambda t: abs(t - teams))
        self.year = year
        self.ppr_type = ppr_type
        self._lookup = lookup
        self.cache_dir = cache_dir or DRAFT_DIR

    def describe(self) -> str:
        return (
            f"{self.year} ADP from Fantasy Football Calculator "
            f"({self.ppr_type.replace('_', ' ')}, {self.teams}-team)"
        )

    def _get_lookup(self) -> PlayerLookup:
        if self._lookup is None:
            self._lookup = PlayerLookup.build()
        return self._lookup

    @property
    def _cache_path(self) -> Path:
        fmt = _FFC_FORMAT[self.ppr_type]
        return self.cache_dir / f"adp_ffc_{self.year}_{fmt}_{self.teams}.json"

    def _fetch(self) -> list[dict]:
        import json
        import time
        import urllib.request

        cache = self._cache_path
        if cache.exists() and (time.time() - cache.stat().st_mtime) < CACHE_TTL_HOURS * 3600:
            return json.loads(cache.read_text())["players"]

        url = FFC_URL.format(fmt=_FFC_FORMAT[self.ppr_type], teams=self.teams, year=self.year)
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "edge-engine/1.0"})
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.load(response)
        except Exception as e:
            if cache.exists():
                # Stale beats nothing on draft day, but say which it is.
                raise RuntimeError(
                    f"Couldn't reach the ADP service ({type(e).__name__}). A cached copy from "
                    f"{time.ctime(cache.stat().st_mtime)} exists at {cache} — delete it to force "
                    "a retry, or use data/draft/adp.csv instead."
                ) from e
            raise RuntimeError(
                f"Couldn't reach the ADP service ({type(e).__name__}: {e}). Export ADP by hand "
                "to data/draft/adp.csv instead — see data/draft/README.md."
            ) from e

        if payload.get("status") != "Success" or not payload.get("players"):
            raise RuntimeError(
                f"The ADP service returned no players for {self.year}. It may not have published "
                f"{self.year} boards yet; use data/draft/adp.csv instead."
            )

        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(payload))
        return payload["players"]

    def get_market_prices(self) -> list[MarketPrice]:
        lookup = self._get_lookup()
        prices: list[MarketPrice] = []
        for row in self._fetch():
            name = str(row.get("name", "")).strip()
            position = str(row.get("position", "")).strip().upper()
            team = str(row.get("team", "")).strip().upper()
            adp = row.get("adp")
            if not name or adp is None:
                continue
            # FFC's position labels differ from this project's: DEF for
            # team defenses (here: DST) and PK for kickers (here: K).
            # Without this, every kicker and defense lands under a
            # position the league config doesn't recognise and silently
            # falls out of the board's position filter.
            position = {"DEF": "DST", "PK": "K"}.get(position, position)

            gsis_id, status, note = lookup.resolve(name, position, team)
            prices.append(
                MarketPrice(
                    name=name, position=position, team=team, adp=float(adp),
                    player_id=gsis_id, match_status=status, match_note=note,
                )
            )
        if not prices:
            raise RuntimeError(f"No usable ADP rows came back for {self.year}.")
        return prices


def default_market_source(
    year: int, ppr_type: str = "full_ppr", teams: int = 12
) -> MarketPriceSource:
    """Live ADP if reachable, hand-exported CSV if not.

    Order matters: the live board is current and needs no maintenance,
    but a draft is an unrepeatable event, so a local CSV must always be
    able to take over."""
    try:
        source = FantasyFootballCalculatorSource(year=year, ppr_type=ppr_type, teams=teams)
        source._fetch()  # fail fast here rather than mid-render
        return source
    except RuntimeError:
        return ManualMarketPriceSource()
