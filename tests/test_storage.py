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
