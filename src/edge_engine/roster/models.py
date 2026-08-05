"""Data model for the roster-state interface (PRD requirement 1).

These types are the contract between "wherever roster data comes from"
(manual CSV/YAML today, a live ESPN connector later) and everything
downstream (ranking, roster-fit re-ranking). Downstream code should only
ever depend on these types plus RosterStateSource in interface.py — never
on the manual file formats directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

MatchStatus = Literal["matched", "unmatched", "ambiguous"]


@dataclass(frozen=True)
class Player:
    name: str  # as entered by the user
    position: str
    team: str
    player_id: str | None = None  # resolved nflverse gsis_id, if matched
    bye_week: int | None = None
    match_status: MatchStatus = "unmatched"
    match_note: str = ""
    slot_position: str | None = None  # this-week starter/bench slot ("RB","FLEX","BE","IR"...); only ESPN populates it


@dataclass(frozen=True)
class ScoringSettings:
    ppr_type: Literal["standard", "half_ppr", "full_ppr"]
    bonuses: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class WaiverConfig:
    system: Literal["FAAB", "priority"]
    season_budget: float
    min_bid: float
    clear_day: str
    post_clear_rule: str = "first_come_first_available"


@dataclass(frozen=True)
class LeagueConfig:
    season: int
    scoring: ScoringSettings
    lineup_slots: dict[str, int]
    waivers: WaiverConfig


@dataclass(frozen=True)
class RosterMeta:
    as_of_week: int
    as_of_date: date
    remaining_faab: float
