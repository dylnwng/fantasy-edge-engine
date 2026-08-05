"""Quick status view over the roster-state interface.

Not the ranking output (that's requirement 4) — just a way to check that
roster_state/ files parse, resolve, and are current before running
anything downstream.

Usage:
    python -m edge_engine.roster.cli
"""

from __future__ import annotations

from datetime import date

from edge_engine.roster.interface import get_default_source


def main() -> None:
    source = get_default_source()

    config = source.get_league_config()
    meta = source.get_roster_meta()
    roster = source.get_rostered_players()
    free_agents = source.get_free_agents()

    staleness = (date.today() - meta.as_of_date).days
    print(f"League: {config.season} season, {config.scoring.ppr_type}, "
          f"{config.waivers.system} (${meta.remaining_faab} remaining)")
    print(f"Roster data as of week {meta.as_of_week} ({meta.as_of_date})", end="")
    if staleness > 7:
        print(f"  ⚠ {staleness} days old — update roster_state/ before trusting this")
    else:
        print()

    print(f"\nRostered players ({len(roster)}):")
    for p in roster:
        flag = "" if p.match_status == "matched" else f"  [{p.match_status}: {p.match_note}]"
        bye = f"bye {p.bye_week}" if p.bye_week else "bye ?"
        print(f"  {p.name} ({p.position}, {p.team}, {bye}){flag}")

    print(f"\nFree agents tracked ({len(free_agents)}):")
    for p in free_agents:
        flag = "" if p.match_status == "matched" else f"  [{p.match_status}: {p.match_note}]"
        bye = f"bye {p.bye_week}" if p.bye_week else "bye ?"
        print(f"  {p.name} ({p.position}, {p.team}, {bye}){flag}")

    unmatched = [p for p in roster + free_agents if p.match_status != "matched"]
    if unmatched:
        print(f"\n{len(unmatched)} player(s) did not resolve cleanly — check spelling/team above.")


if __name__ == "__main__":
    main()
