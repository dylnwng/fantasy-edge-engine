"""Live draft pick acquisition from ESPN.

**Spike result (PRD §4.4 required verifying this before designing on top
of it).** Confirmed against the real league: espn_api's `League` already
fetches ESPN's `mDraftDetail` view using the same cookie auth the roster
source uses, and `league.draft` returns `BasePick` objects carrying
round, pick-within-round, player name and player id. A 2025 query
returned 192 real picks; a 2026 query (league not yet drafted) returned
0 picks without erroring. The endpoint exists, it is reachable, and it
is pollable, so live sync is viable.

Deliberately **not** browser automation or DOM scraping of the draft UI:
that breaks on draft night, which is the one moment with zero recovery
time.

Polling contract: `fetch_picks()` returns the FULL pick set every call,
not a delta. The tracker keys picks by (round, pick) and reconciles, so
duplicates and out-of-order arrivals are harmless by construction --
which is why this layer doesn't need to track what it has already seen.
"""

from __future__ import annotations

import os
from typing import Protocol

from edge_engine.draft.tracker import Pick

MIN_POLL_SECONDS = 2.0  # never poll faster; ESPN rate-limits


class DraftPickSource(Protocol):
    def fetch_picks(self) -> list[Pick]: ...

    def describe(self) -> str: ...


class EspnDraftPickSource:
    """Polls ESPN's draft detail for the configured league.

    Reads the same ESPN_* environment variables as the roster source
    rather than inventing a second credential path."""

    def __init__(self, league_id: int | None = None, year: int | None = None,
                 espn_s2: str | None = None, swid: str | None = None):
        try:
            self.league_id = league_id or int(os.environ["ESPN_LEAGUE_ID"])
            self.year = year or int(os.environ["ESPN_YEAR"])
            self._espn_s2 = espn_s2 or os.environ["ESPN_S2"]
            self._swid = swid or os.environ["ESPN_SWID"]
        except KeyError as e:
            raise RuntimeError(
                f"Live draft sync needs {e.args[0]}, which isn't set. Copy .env.example "
                "to .env and fill in your league's values."
            ) from e
        self._league = None
        self._fetched_once = False

    def describe(self) -> str:
        return f"ESPN league {self.league_id} ({self.year}) draft detail"

    def _get_league(self):
        if self._league is None:
            from espn_api.football import League
            from espn_api.requests.espn_requests import ESPNAccessDenied, ESPNInvalidLeague

            try:
                self._league = League(
                    league_id=self.league_id, year=self.year,
                    espn_s2=self._espn_s2, swid=self._swid,
                )
            except ESPNAccessDenied as e:
                raise RuntimeError(
                    "ESPN rejected the ESPN_S2/ESPN_SWID cookies in .env. On draft night "
                    "this is the failure to expect -- refresh them from your browser's dev "
                    "tools and restart, or fall back to entering picks by hand."
                ) from e
            except ESPNInvalidLeague as e:
                raise RuntimeError(
                    f"ESPN says league {self.league_id} ({self.year}) doesn't exist. Check "
                    "ESPN_LEAGUE_ID and ESPN_YEAR."
                ) from e
        return self._league

    def fetch_picks(self) -> list[Pick]:
        league = self._get_league()
        # refresh_draft() re-hits mDraftDetail; on the first call the
        # constructor has already populated it. Gated on "have we fetched
        # before", NOT on whether league.draft is non-empty: a server
        # started before the draft begins sees an empty pick list, and an
        # emptiness guard would then never refresh again -- the exact
        # draft-night startup sequence would show zero live picks forever.
        if self._fetched_once:
            league.refresh_draft()
        self._fetched_once = True
        raw = getattr(league, "draft", []) or []
        return [
            Pick(
                round_number=int(p.round_num),
                pick_number=int(p.round_pick),
                player_name=str(p.playerName),
            )
            for p in raw
            if getattr(p, "playerName", None)
        ]
