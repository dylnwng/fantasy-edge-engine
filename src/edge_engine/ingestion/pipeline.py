"""Entry point: ingest one or more NFL seasons into the local player-week store.

Usage:
    python -m edge_engine.ingestion.pipeline --seasons 2023 2024
    python -m edge_engine.ingestion.pipeline --seasons 2026 --force-refresh
"""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from edge_engine.ingestion.transform import build_player_week_table
from edge_engine.paths import PLAYER_WEEK_PATH
from edge_engine.storage import upsert

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def ingest_seasons(seasons: list[int], force_refresh: bool = False) -> pd.DataFrame:
    result = None
    for season in seasons:
        logger.info("Building player-week table for %s season", season)
        season_df = build_player_week_table(season, force_refresh=force_refresh)
        logger.info("Upserting %d rows for %s season", len(season_df), season)
        result = upsert(season_df, PLAYER_WEEK_PATH)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="bypass the raw-data cache and re-download from nflverse",
    )
    args = parser.parse_args()
    ingest_seasons(args.seasons, force_refresh=args.force_refresh)


if __name__ == "__main__":
    main()
