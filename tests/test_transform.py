import pandas as pd

from edge_engine.ingestion.transform import attach_red_zone_stats


def test_red_zone_target_share_zero_vs_null():
    # P1 and P2 both got red-zone targets for LA (2 total). P3 is on LA too,
    # had a red-zone rush (a touch) but no red-zone target -> share should be
    # a real 0, not null, since LA *did* throw in the red zone that week.
    # P4 is on NYJ, which had zero red-zone pass attempts that week -> share
    # is genuinely undefined, must stay null.
    base = pd.DataFrame(
        {
            "season": [2023, 2023, 2023, 2023],
            "week": [1, 1, 1, 1],
            "player_id": ["P1", "P2", "P3", "P4"],
            "team": ["LA", "LA", "LA", "NYJ"],
        }
    )
    red_zone_touches = pd.DataFrame(
        {
            "season": [2023, 2023, 2023, 2023],
            "week": [1, 1, 1, 1],
            "player_id": ["P1", "P2", "P3", "P4"],
            "red_zone_touches": [1, 1, 1, 1],
        }
    )
    player_rz_targets = pd.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 1],
            "player_id": ["P1", "P2"],
            "player_rz_targets": [1, 1],
        }
    )
    team_rz_targets = pd.DataFrame(
        {
            "season": [2023],
            "week": [1],
            "team": ["LA"],
            "team_rz_targets": [2],
        }
    )

    result = attach_red_zone_stats(base, red_zone_touches, player_rz_targets, team_rz_targets)
    result = result.set_index("player_id")

    assert result.loc["P1", "red_zone_target_share"] == 0.5
    assert result.loc["P2", "red_zone_target_share"] == 0.5
    # P3: had zero targets, but team had a share to divide -> 0.0, not null.
    assert result.loc["P3", "red_zone_target_share"] == 0.0
    # P4: team had no red-zone pass attempts at all -> genuinely undefined.
    assert pd.isna(result.loc["P4", "red_zone_target_share"])
    # red_zone_touches is always a known count, never null.
    assert result["red_zone_touches"].isna().sum() == 0
