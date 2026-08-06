import pandas as pd

from edge_engine.storage import upsert


def test_upsert_replaces_matching_keys_not_duplicates(tmp_path):
    path = tmp_path / "player_week.parquet"

    week1 = pd.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 1],
            "player_id": ["A", "B"],
            "fantasy_points_ppr": [10.0, 5.0],
        }
    )
    upsert(week1, path)

    # Re-ingesting the same week with corrected values should replace, not add.
    week1_corrected = pd.DataFrame(
        {
            "season": [2023],
            "week": [1],
            "player_id": ["A"],
            "fantasy_points_ppr": [12.5],
        }
    )
    result = upsert(week1_corrected, path)

    assert len(result) == 2
    a_row = result[result["player_id"] == "A"]
    assert a_row["fantasy_points_ppr"].iloc[0] == 12.5
    assert result.duplicated(subset=["season", "week", "player_id"]).sum() == 0


def test_upsert_appends_new_weeks(tmp_path):
    path = tmp_path / "player_week.parquet"

    week1 = pd.DataFrame({"season": [2023], "week": [1], "player_id": ["A"], "fantasy_points_ppr": [10.0]})
    week2 = pd.DataFrame({"season": [2023], "week": [2], "player_id": ["A"], "fantasy_points_ppr": [8.0]})

    upsert(week1, path)
    result = upsert(week2, path)

    assert len(result) == 2
    assert set(result["week"]) == {1, 2}


def test_upsert_dedupes_duplicate_keys_within_a_single_batch(tmp_path):
    # Two rows sharing the same (season, week, player_id) key in ONE call
    # -- e.g. an upstream join fan-out -- used to both get written,
    # violating this module's own "replaces, not duplicates" contract.
    path = tmp_path / "player_week.parquet"

    batch = pd.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 1],
            "player_id": ["A", "A"],
            "fantasy_points_ppr": [10.0, 20.0],
        }
    )
    result = upsert(batch, path)

    assert len(result) == 1
    # keep="last": the later row in the batch wins.
    assert result.iloc[0]["fantasy_points_ppr"] == 20.0
