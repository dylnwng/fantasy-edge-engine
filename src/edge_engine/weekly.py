"""Single weekly entry point: refresh the current season's usage data,
then run free-agent ranking (roster-fit) and, if a live matchup exists,
the matchup simulator.

Built to fix a real usability gap, not a speculative one: the README's
quickstart is 4 separate manual commands, and re-running the ingestion
pipeline without --force-refresh silently serves whatever's already
cached (see ingestion/pipeline.py + nflverse_cache.py) -- it never hits
the network again, with no warning that it didn't. Copy-pasting the same
quickstart command every Tuesday produces zero new data forever. This
force-refreshes only the CURRENT season (historical training seasons in
model_config.yaml are immutable once a season ends -- re-pulling them
weekly would be wasted network time for no benefit), and treats a
missing/expired ESPN session or a not-yet-published nflverse season as
recoverable, surfaced conditions instead of a raw traceback -- same
"surfaced, never silent" discipline as injury_context.py and
roster_status.py elsewhere in this project.

Retraining the model is deliberately NOT part of this command --
predict.py loads a static trained model artifact and only needs fresh
features, not a fresh fit, so there's nothing to gain from retraining
every week. Run `python -m edge_engine.model.train` separately, and only
when you actually want to (e.g. once more of a season's data exists).

Usage:
    python -m edge_engine.weekly
"""

from __future__ import annotations

from edge_engine.ingestion.pipeline import ingest_seasons
from edge_engine.ranking.roster_fit import main as run_roster_fit
from edge_engine.roster.interface import RosterStateSource, get_default_source


def refresh_current_season(season: int) -> None:
    """Force-refreshes only `season`. Any failure here (most likely:
    nflverse hasn't published this season's weekly stats table yet -- a
    real, previously-confirmed external blocker, not a bug in this
    pipeline) is reported and swallowed rather than aborting the whole
    weekly run -- ranking/roster-fit below can still produce useful
    output from whatever's already ingested."""
    print(f"Refreshing {season} usage data...")
    try:
        ingest_seasons([season], force_refresh=True)
    except Exception as e:
        print(
            f"  Could not refresh {season} data ({e}) -- nflverse may not have "
            f"published this season's weekly stats yet. Continuing with "
            f"whatever data is already ingested."
        )


def run_matchup_simulator_if_live(source: RosterStateSource) -> None:
    if not hasattr(source, "get_current_matchup"):
        print(
            "Skipping matchup simulator: requires EDGE_ENGINE_ROSTER_SOURCE=espn "
            "(opponent rosters have no manual-CSV equivalent)."
        )
        return
    try:
        from edge_engine.simulation.matchup_cli import main as run_matchup_cli

        run_matchup_cli()
    except Exception as e:
        # No live matchup to simulate yet -- e.g. the season hasn't
        # started, or a bye week in an odd-team league. Deliberately
        # broad: get_current_matchup() raises a clean ValueError for the
        # case it recognizes, but confirmed directly (an actual offseason
        # run, week 0) that espn_api's own box_scores() can also raise a
        # raw KeyError deep inside its own parsing when asked about a
        # week with no real matchup data yet -- not something worth
        # trying to enumerate every possible library-internal exception
        # for, when the outcome is the same either way: skip this step,
        # ranking output above is still valid. Any genuine ESPN auth
        # failure would already have surfaced earlier, in the roster-fit
        # step above, which hits the same _get_league() first.
        print(f"Skipping matchup simulator: {e}")


def main() -> None:
    source = get_default_source()
    try:
        season = source.get_league_config().season

        refresh_current_season(season)
        print()

        print("=== Free-agent rankings (roster-fit) ===")
        run_roster_fit()

        print()
        print("=== Matchup simulator ===")
        run_matchup_simulator_if_live(source)
    except RuntimeError as e:
        # An ESPN auth/config failure (see espn_source.py's _get_league) --
        # already a clear, actionable message by construction. Stop here
        # rather than letting a raw traceback be the last thing printed.
        raise SystemExit(f"\nStopped: {e}") from e


if __name__ == "__main__":
    main()
