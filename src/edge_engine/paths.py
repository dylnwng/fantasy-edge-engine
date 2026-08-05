from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ROSTER_STATE_DIR = DATA_DIR / "roster_state"

PLAYER_WEEK_PATH = PROCESSED_DIR / "player_week.parquet"
