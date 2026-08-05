"""Live ESPN connector (P2 future consideration from the PRD) -- a
drop-in replacement for ManualRosterStateSource that implements the same
RosterStateSource protocol from interface.py. Nothing downstream (the
model, the ranking/roster-fit layers) needs to change; only
get_default_source()'s wiring does.

ESPN's fantasy API is unofficial and, for a private league, cookie
authenticated (espn_s2 + SWID). Those two values are session credentials
for your own ESPN account, not something this code can obtain on its
own -- see .env.example for how to pull them from your browser's own
dev tools. They're read from environment variables (typically via a
gitignored .env file), never hardcoded, never logged.

Known limitations, stated up front rather than silently assumed away:
  - Requires ESPN_TEAM_ID (found in your team's URL on the ESPN site) to
    identify which roster is "yours" -- espn_api doesn't expose a
    reliable owner-SWID-to-team mapping to auto-detect this.
  - Scoring conversion only recognizes reception points cleanly mapping
    to standard/half/full PPR (0 / 0.5 / 1.0 per reception). An
    unusual value (e.g. 1.5 PPR) raises rather than silently rounding,
    since a wrong scoring assumption would silently corrupt every
    downstream point calculation.
  - Bonus stats are derived as *deltas* from nflverse's standard-scoring
    baseline (4/6/6/-2/-2 for pass/rush/rec TD, INT, fumble lost) --
    because nflverse's own fantasy_points columns already bake those
    in, and compute_fantasy_points() only expects genuine deltas from
    that baseline, not absolute point values.
  - min_bid and waiver clear_day aren't exposed by ESPN's API at all;
    reasonable ESPN-standard defaults ($1, Wednesday) are used, called
    out explicitly rather than presented as pulled from the league.
  - This module has not been exercised against a real live league in
    this session -- there were no credentials available to test with.
    The mapping logic is unit-tested against synthetic espn_api-shaped
    objects (tests/test_espn_source.py), but a first real run should be
    treated as the actual integration test.
"""

from __future__ import annotations

import os
from datetime import date

from edge_engine.roster.bye_weeks import get_bye_weeks
from edge_engine.roster.matchup import MatchupPlayer, MatchupSnapshot
from edge_engine.roster.models import LeagueConfig, Player, RosterMeta, ScoringSettings, WaiverConfig
from edge_engine.roster.player_lookup import PlayerLookup

# nflverse's implicit standard-scoring baseline, already baked into the
# fantasy_points/fantasy_points_ppr columns compute_fantasy_points() uses.
# A league's bonuses dict should only carry the *difference* from this.
_STANDARD_BASELINE = {
    "pass_td": 4.0,
    "rush_td": 6.0,
    "rec_td": 6.0,
    "interception": -2.0,
    "fumble_lost": -2.0,
}

# ESPN scoringItems statId -> our bonus key (see espn_api's
# SETTINGS_SCORING_FORMAT_MAP for the full list of stat ids).
_ESPN_STAT_ID_TO_BONUS_KEY = {
    4: "pass_td",
    25: "rush_td",
    43: "rec_td",
    20: "interception",
    72: "fumble_lost",
}
_RECEPTION_STAT_ID = 53

# ESPN uses a couple of team abbreviations nflverse's team_desc doesn't
# carry as an alias (checked at build time: LAR is covered, WSH is not).
_ESPN_TEAM_ALIASES = {"WSH": "WAS"}

# Flex-eligible slot labels (anything spanning more than one position).
_FLEX_LABEL_MARKERS = ("/", "OP", "TQB")


def _normalize_team(team: str) -> str:
    return _ESPN_TEAM_ALIASES.get(team, team)


def _ppr_type_from_scoring_format(scoring_format: list[dict]) -> str:
    reception_points = next(
        (item["points"] for item in scoring_format if item.get("id") == _RECEPTION_STAT_ID), 0.0
    )
    if reception_points == 0:
        return "standard"
    if reception_points == 0.5:
        return "half_ppr"
    if reception_points == 1.0:
        return "full_ppr"
    raise ValueError(
        f"League's reception scoring is {reception_points} pts/catch, which isn't "
        "standard (0), half (0.5), or full (1.0) PPR. Refusing to guess -- add "
        "support for this value explicitly rather than silently mis-scoring every "
        "prediction."
    )


def _bonuses_from_scoring_format(scoring_format: list[dict]) -> dict[str, float]:
    points_by_stat_id = {item["id"]: item["points"] for item in scoring_format}
    bonuses = {}
    for stat_id, bonus_key in _ESPN_STAT_ID_TO_BONUS_KEY.items():
        if stat_id not in points_by_stat_id:
            continue
        delta = points_by_stat_id[stat_id] - _STANDARD_BASELINE[bonus_key]
        if delta != 0:
            bonuses[bonus_key] = delta
    return bonuses


def _map_lineup_slots(position_slot_counts: dict[str, int]) -> dict[str, int]:
    slots: dict[str, int] = {}
    flex_total = 0
    for label, count in position_slot_counts.items():
        if not count:
            continue
        if label == "D/ST":
            slots["DST"] = slots.get("DST", 0) + count
        elif label == "BE":
            slots["BENCH"] = slots.get("BENCH", 0) + count
        elif label == "IR":
            slots["IR"] = slots.get("IR", 0) + count
        elif any(marker in label for marker in _FLEX_LABEL_MARKERS):
            flex_total += count
        else:
            slots[label] = slots.get(label, 0) + count
    if flex_total:
        slots["FLEX"] = flex_total
    return slots


class EspnRosterStateSource:
    def __init__(
        self,
        league_id: int | None = None,
        year: int | None = None,
        team_id: int | None = None,
        espn_s2: str | None = None,
        swid: str | None = None,
    ):
        try:
            self.league_id = league_id or int(os.environ["ESPN_LEAGUE_ID"])
            self.year = year or int(os.environ["ESPN_YEAR"])
            self.team_id = team_id or int(os.environ["ESPN_TEAM_ID"])
            self._espn_s2 = espn_s2 or os.environ["ESPN_S2"]
            self._swid = swid or os.environ["ESPN_SWID"]
        except KeyError as e:
            raise RuntimeError(
                f"EDGE_ENGINE_ROSTER_SOURCE=espn but {e.args[0]} isn't set. Copy "
                ".env.example to .env and fill in your league's values."
            ) from e

        self._league = None
        self._lookup: PlayerLookup | None = None
        self._league_config: LeagueConfig | None = None

    def _get_league(self):
        if self._league is None:
            from espn_api.football import League

            self._league = League(
                league_id=self.league_id, year=self.year, espn_s2=self._espn_s2, swid=self._swid
            )
        return self._league

    def _get_lookup(self) -> PlayerLookup:
        if self._lookup is None:
            self._lookup = PlayerLookup.build()
        return self._lookup

    def _get_my_team(self):
        league = self._get_league()
        for team in league.teams:
            if team.team_id == self.team_id:
                return team
        raise ValueError(
            f"No team with team_id={self.team_id} in league {self.league_id}. "
            "Check ESPN_TEAM_ID against the team URL on the ESPN site."
        )

    def _map_player(self, espn_player, byes: dict[str, int]) -> Player:
        name = espn_player.name
        position = espn_player.position
        team = _normalize_team(espn_player.proTeam)

        gsis_id, status, note = self._get_lookup().resolve(name, position, team)
        # Regular roster Player objects carry .lineupSlot; BoxPlayer (from
        # box_scores(), used by get_current_matchup) carries .slot_position
        # instead -- both duck-type through here.
        slot_position = getattr(espn_player, "slot_position", None) or getattr(espn_player, "lineupSlot", None) or None
        return Player(
            name=name,
            position=position,
            team=team,
            player_id=gsis_id,
            bye_week=byes.get(team),
            match_status=status,
            match_note=note,
            slot_position=slot_position,
        )

    @staticmethod
    def _resolve_game_final(box_player) -> tuple[bool, bool]:
        """Returns (on_bye, game_final). espn_api's BoxPlayer.game_played
        defaults to 100 and is only ever recomputed inside the "this
        team's game is in this week's pro schedule" branch -- a bye-week
        player never enters that branch, so game_played silently stays
        at its default 100. on_bye_week must be checked first, or a bye
        player reads as "game final." Pulled out as its own pure
        function so this precedence is directly unit-testable."""
        on_bye = bool(getattr(box_player, "on_bye_week", False))
        game_final = (not on_bye) and getattr(box_player, "game_played", 0) == 100
        return on_bye, game_final

    def _map_matchup_player(self, box_player, byes: dict[str, int]) -> MatchupPlayer:
        on_bye, game_final = self._resolve_game_final(box_player)
        return MatchupPlayer(
            player=self._map_player(box_player, byes),
            espn_projected_points=float(getattr(box_player, "projected_points", 0.0)),
            actual_points=float(getattr(box_player, "points", 0.0)) if game_final else 0.0,
            game_final=game_final,
            on_bye=on_bye,
        )

    @staticmethod
    def _select_my_side(box_scores, my_team_id: int):
        """Pure-ish selection logic factored out for unit testing against
        lightweight fake BoxScore-shaped objects. Returns
        (my_team, my_lineup, opponent_team, opponent_lineup) or None if no
        two-team matchup exists for my_team_id in this week's box scores
        (e.g. a bye week in an odd-team league)."""
        for box_score in box_scores:
            home_team, away_team = getattr(box_score, "home_team", None), getattr(box_score, "away_team", None)
            home_id = getattr(home_team, "team_id", None)
            away_id = getattr(away_team, "team_id", None)
            if home_id != my_team_id and away_id != my_team_id:
                continue
            if home_team is None or away_team is None:
                return None  # this team's bye week -- no opponent to simulate against
            if home_id == my_team_id:
                return home_team, box_score.home_lineup, away_team, box_score.away_lineup
            return away_team, box_score.away_lineup, home_team, box_score.home_lineup
        return None

    def get_current_matchup(self, week: int | None = None) -> MatchupSnapshot:
        league = self._get_league()
        target_week = week or league.current_week
        byes = get_bye_weeks(self.get_league_config().season)

        box_scores = league.box_scores(week=target_week)
        selected = self._select_my_side(box_scores, self.team_id)
        if selected is None:
            raise ValueError(
                f"No two-team matchup found for team_id={self.team_id} in week {target_week} "
                "-- a bye week in an odd-team league, or the week hasn't been scheduled yet."
            )
        my_team, my_lineup, opp_team, opp_lineup = selected

        return MatchupSnapshot(
            week=target_week,
            my_team_name=my_team.team_name,
            opponent_team_name=opp_team.team_name,
            my_lineup=[self._map_matchup_player(p, byes) for p in my_lineup],
            opponent_lineup=[self._map_matchup_player(p, byes) for p in opp_lineup],
        )

    def get_rostered_players(self) -> list[Player]:
        season = self.get_league_config().season
        byes = get_bye_weeks(season)
        team = self._get_my_team()
        return [self._map_player(p, byes) for p in team.roster]

    def get_free_agents(self) -> list[Player]:
        season = self.get_league_config().season
        byes = get_bye_weeks(season)
        league = self._get_league()

        players = []
        for position in ("QB", "RB", "WR", "TE"):
            players.extend(league.free_agents(size=200, position=position))
        return [self._map_player(p, byes) for p in players]

    def get_league_config(self) -> LeagueConfig:
        if self._league_config is None:
            settings = self._get_league().settings

            scoring = ScoringSettings(
                ppr_type=_ppr_type_from_scoring_format(settings.scoring_format),
                bonuses=_bonuses_from_scoring_format(settings.scoring_format),
            )
            waivers = WaiverConfig(
                system="FAAB" if settings.faab else "priority",
                season_budget=float(settings.acquisition_budget),
                min_bid=1.0,  # not exposed by the API; ESPN's standard default
                clear_day="Wednesday",  # not exposed by the API; ESPN's standard default
                post_clear_rule="first_come_first_available",
            )
            self._league_config = LeagueConfig(
                season=self.year,
                scoring=scoring,
                lineup_slots=_map_lineup_slots(settings.position_slot_counts),
                waivers=waivers,
            )
        return self._league_config

    def get_roster_meta(self) -> RosterMeta:
        league = self._get_league()
        config = self.get_league_config()
        team = self._get_my_team()

        remaining_faab = config.waivers.season_budget - float(team.acquisition_budget_spent)
        return RosterMeta(
            as_of_week=league.current_week,
            as_of_date=date.today(),
            remaining_faab=remaining_faab,
        )
