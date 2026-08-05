"""Resolves a (mean, std) point projection for any player appearing in a
matchup, with an explicit, labeled fallback chain -- every player gets a
projection back, never silently dropped, and every non-"model" tier
carries a note explaining why it isn't the primary path.

Tier order (first match wins):
  1. bye                        -- on bye this week: mean=0, std=0
  2. final                      -- game already played: mean=actual points, std=0
  3. model                      -- opportunity model's predicted_score + this
                                    player's own trailing_points_std (>=3 games)
  4. model_position_std_fallback -- model has a score (>=2 games) but <3 games
                                    for a reliable std -- borrow a position-average std
  5. espn_projection_only       -- no model score at all: unresolved player,
                                    <2 games played, or a position (K/D-ST) the
                                    opportunity model has no usage data for at
                                    all. Mean = ESPN's own projection. Std = 0.0
                                    for K/D-ST specifically (no fabricated spread
                                    for positions with no real history), a
                                    position-average std otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import pandas as pd

from edge_engine.model.config import load_model_config
from edge_engine.model.predict import score_as_of_week
from edge_engine.model.scoring import compute_points_for_seasons
from edge_engine.model.variance import compute_trailing_points_std
from edge_engine.roster.interface import get_default_source
from edge_engine.roster.matchup import MatchupPlayer
from edge_engine.simulation.config import SimulationConfig, load_simulation_config

# No real usage history exists for these positions at all (K=7 rows,
# D/ST=0 rows across the entire ingested nflverse dataset) -- treated as
# deterministic at ESPN's own projection rather than inventing a spread.
_ZERO_VARIANCE_POSITIONS = {"K", "D/ST", "DST"}


@dataclass(frozen=True)
class PlayerProjection:
    player_id: str | None
    name: str
    position: str
    mean: float
    std: float
    source: str  # "bye" | "final" | "model" | "model_position_std_fallback" | "espn_projection_only"
    note: str  # non-empty whenever source != "model"


def _scored_lookup(
    season: int, week: int, sim_config: SimulationConfig
) -> tuple[dict[str, tuple[float, float | None]], dict[str, float]]:
    """Runs score_as_of_week(season, week) + compute_trailing_points_std()
    once, scoped to only data knowable before `week` -- not "whatever's
    latest in the whole file" (see predict.py's score_as_of_week
    docstring for why that distinction matters here specifically: this
    is exactly what powers the --week flag's "what would we have said
    before week N" use case). Returns (player_id -> (predicted_score,
    trailing_points_std_or_None), position -> average available
    trailing_points_std)."""
    model_config = load_model_config()
    scored = score_as_of_week(season, week, model_config)
    if scored.empty:
        return {}, {}

    league_config = get_default_source().get_league_config()
    points_table = compute_points_for_seasons([season], league_config.scoring)
    std_table = compute_trailing_points_std(
        points_table, window=sim_config.variance_window, min_games=sim_config.min_games_for_variance
    )

    merged = scored.merge(std_table, on=["season", "week", "player_id"], how="left")

    scored_by_player_id = {
        row.player_id: (float(row.predicted_score), None if pd.isna(row.trailing_points_std) else float(row.trailing_points_std))
        for row in merged.itertuples()
    }
    position_avg_std = (
        merged.dropna(subset=["trailing_points_std"]).groupby("position")["trailing_points_std"].mean().to_dict()
    )
    return scored_by_player_id, position_avg_std


def resolve_projection(
    matchup_player: MatchupPlayer,
    scored_by_player_id: dict[str, tuple[float, float | None]],
    position_avg_std: dict[str, float],
    sim_config: SimulationConfig,
) -> PlayerProjection:
    p = matchup_player.player

    if matchup_player.on_bye:
        return PlayerProjection(p.player_id, p.name, p.position, 0.0, 0.0, "bye", "on bye this week")

    if matchup_player.game_final:
        return PlayerProjection(
            p.player_id, p.name, p.position, matchup_player.actual_points, 0.0, "final",
            "game already final this week -- using actual points, not a projection",
        )

    entry = scored_by_player_id.get(p.player_id) if p.player_id else None

    if entry is None:
        if p.position in _ZERO_VARIANCE_POSITIONS:
            std = 0.0
            note = (
                f"{p.position} has no real usage history in nflverse's data (checked: near-zero rows "
                "league-wide) -- using ESPN's own projection with zero variance rather than a fabricated spread"
            )
        else:
            std = position_avg_std.get(p.position, sim_config.untracked_position_std)
            note = (
                "unresolved player (no nflverse match)" if not p.player_id
                else "fewer than 2 games played this season -- falling back to ESPN's own "
                     "projection with a position-average std"
            )
        return PlayerProjection(p.player_id, p.name, p.position, matchup_player.espn_projected_points, std, "espn_projection_only", note)

    predicted_score, trailing_std = entry
    if trailing_std is None:
        std = position_avg_std.get(p.position, sim_config.untracked_position_std)
        return PlayerProjection(
            p.player_id, p.name, p.position, predicted_score, std, "model_position_std_fallback",
            f"fewer than {sim_config.min_games_for_variance} trailing games -- using position-average "
            "std instead of this player's own",
        )
    return PlayerProjection(p.player_id, p.name, p.position, predicted_score, trailing_std, "model", "")


def build_projections(
    matchup_players: list[MatchupPlayer],
    season: int,
    week: int,
    sim_config: SimulationConfig | None = None,
) -> list[PlayerProjection]:
    """Resolves every player in `matchup_players` to a PlayerProjection for
    the matchup being played in `week` of `season`. Runs the expensive
    lookups (score_as_of_week, variance) once, not once per player.

    `season`/`week` matter: they scope the model's trailing-data lookup
    to only what was knowable before this specific week (see
    score_as_of_week), not just "whatever's most recent in the ingested
    file" -- required for the --week flag to mean what it says."""
    sim_config = sim_config or load_simulation_config()
    scored_by_player_id, position_avg_std = _scored_lookup(season, week, sim_config)
    return [resolve_projection(mp, scored_by_player_id, position_avg_std, sim_config) for mp in matchup_players]


def make_projection_fn(
    season: int, week: int, sim_config: SimulationConfig | None = None
) -> Callable[[MatchupPlayer], PlayerProjection]:
    """Runs the expensive score_as_of_week()/variance lookups once (scoped
    to `season`/`week` -- see build_projections), then returns a callable
    resolving individual players against that fixed snapshot -- for the
    FLEX optimizer, which needs a projection per candidate lineup without
    re-running those lookups each time."""
    sim_config = sim_config or load_simulation_config()
    scored_by_player_id, position_avg_std = _scored_lookup(season, week, sim_config)
    return lambda mp: resolve_projection(mp, scored_by_player_id, position_avg_std, sim_config)
