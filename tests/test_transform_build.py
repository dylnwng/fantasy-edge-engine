import pandas as pd
import pytest

from edge_engine.ingestion import transform
from edge_engine.ingestion.transform import OUTPUT_COLUMNS, _base_usage, _red_zone_stats, _snap_pct

PBP_COLUMNS = [
    "season", "week", "season_type", "posteam", "yardline_100",
    "rush_attempt", "pass_attempt", "rusher_player_id", "receiver_player_id",
]


def _play(**overrides):
    row = {
        "season": 2024, "week": 1, "season_type": "REG", "posteam": "GB",
        "yardline_100": 10.0, "rush_attempt": 0, "pass_attempt": 0,
        "rusher_player_id": None, "receiver_player_id": None,
    }
    row.update(overrides)
    return row


def _stub_pbp(monkeypatch, plays):
    df = pd.DataFrame(plays, columns=PBP_COLUMNS)
    monkeypatch.setattr(transform.raw, "fetch_pbp_data", lambda season, force_refresh=False: df)


# ---- red-zone aggregation ----

def test_only_plays_inside_the_twenty_count(monkeypatch):
    _stub_pbp(monkeypatch, [
        _play(yardline_100=10.0, rush_attempt=1, rusher_player_id="P1"),
        _play(yardline_100=45.0, rush_attempt=1, rusher_player_id="P1"),  # midfield, not red zone
        _play(yardline_100=None, rush_attempt=1, rusher_player_id="P1"),  # unknown spot
    ])

    touches, _, _ = _red_zone_stats(2024, False)

    assert touches.set_index("player_id").loc["P1", "red_zone_touches"] == 1


def test_red_zone_touches_count_both_carries_and_targets(monkeypatch):
    _stub_pbp(monkeypatch, [
        _play(rush_attempt=1, rusher_player_id="P1"),
        _play(rush_attempt=1, rusher_player_id="P1"),
        _play(pass_attempt=1, receiver_player_id="P1"),
    ])

    touches, _, _ = _red_zone_stats(2024, False)

    assert touches.set_index("player_id").loc["P1", "red_zone_touches"] == 3


def test_postseason_red_zone_plays_are_excluded(monkeypatch):
    _stub_pbp(monkeypatch, [
        _play(week=1, rush_attempt=1, rusher_player_id="P1"),
        _play(week=19, season_type="POST", rush_attempt=1, rusher_player_id="P1"),
    ])

    touches, _, _ = _red_zone_stats(2024, False)

    assert list(touches["week"]) == [1]


def test_targets_without_a_receiver_are_dropped_not_counted_as_a_player(monkeypatch):
    # A throwaway/spike carries a pass_attempt but no receiver_player_id.
    _stub_pbp(monkeypatch, [
        _play(pass_attempt=1, receiver_player_id="P1"),
        _play(pass_attempt=1, receiver_player_id=None),
    ])

    touches, player_targets, team_targets = _red_zone_stats(2024, False)

    assert touches.set_index("player_id").loc["P1", "red_zone_touches"] == 1
    assert player_targets.set_index("player_id").loc["P1", "player_rz_targets"] == 1
    # The unattributed throw still isn't part of the team's denominator,
    # since the team share is built from attributable targets only.
    assert team_targets.set_index("team").loc["GB", "team_rz_targets"] == 1


def test_team_red_zone_targets_are_summed_per_team_not_per_player(monkeypatch):
    _stub_pbp(monkeypatch, [
        _play(pass_attempt=1, receiver_player_id="P1", posteam="GB"),
        _play(pass_attempt=1, receiver_player_id="P2", posteam="GB"),
        _play(pass_attempt=1, receiver_player_id="P3", posteam="CHI"),
    ])

    _, player_targets, team_targets = _red_zone_stats(2024, False)

    by_team = team_targets.set_index("team")
    assert by_team.loc["GB", "team_rz_targets"] == 2
    assert by_team.loc["CHI", "team_rz_targets"] == 1
    assert player_targets.set_index("player_id").loc["P1", "player_rz_targets"] == 1


def test_red_zone_touches_is_an_integer_count(monkeypatch):
    _stub_pbp(monkeypatch, [_play(rush_attempt=1, rusher_player_id="P1")])

    touches, _, _ = _red_zone_stats(2024, False)

    assert touches["red_zone_touches"].dtype.kind == "i"


# ---- snap share ----

SNAP_COLUMNS = ["pfr_player_id", "season", "week", "game_type", "offense_pct"]


def _snap(pfr_id, week=1, offense_pct=0.5, game_type="REG", season=2024):
    return {
        "pfr_player_id": pfr_id, "season": season, "week": week,
        "game_type": game_type, "offense_pct": offense_pct,
    }


def _stub_snaps(monkeypatch, snaps, crosswalk):
    monkeypatch.setattr(
        transform.raw, "fetch_snap_counts",
        lambda season, force_refresh=False: pd.DataFrame(snaps, columns=SNAP_COLUMNS),
    )
    monkeypatch.setattr(
        transform.raw, "fetch_id_crosswalk",
        lambda force_refresh=False: pd.DataFrame(crosswalk, columns=["gsis_id", "pfr_id"]),
    )


def test_snap_counts_are_keyed_to_gsis_ids_via_the_crosswalk(monkeypatch):
    _stub_snaps(monkeypatch, [_snap("PfrA", offense_pct=0.72)], [{"gsis_id": "00-001", "pfr_id": "PfrA"}])

    out = _snap_pct(2024, False)

    assert list(out.columns) == ["player_id", "season", "week", "snap_pct"]
    assert out.iloc[0]["player_id"] == "00-001"
    assert out.iloc[0]["snap_pct"] == pytest.approx(0.72)


def test_a_player_missing_from_the_crosswalk_gets_a_null_id_not_a_dropped_row(monkeypatch):
    _stub_snaps(monkeypatch, [_snap("PfrUnknown")], [{"gsis_id": "00-001", "pfr_id": "PfrA"}])

    out = _snap_pct(2024, False)

    assert len(out) == 1
    assert pd.isna(out.iloc[0]["player_id"])


def test_duplicate_snap_rows_for_one_player_week_take_the_max_not_the_sum(monkeypatch):
    # nflverse can carry more than one row per player-week; summing them
    # would produce a snap share above 1.0.
    _stub_snaps(
        monkeypatch,
        [_snap("PfrA", offense_pct=0.30), _snap("PfrA", offense_pct=0.65)],
        [{"gsis_id": "00-001", "pfr_id": "PfrA"}],
    )

    out = _snap_pct(2024, False)

    assert len(out) == 1
    assert out.iloc[0]["snap_pct"] == pytest.approx(0.65)


def test_postseason_snap_counts_are_excluded(monkeypatch):
    _stub_snaps(
        monkeypatch,
        [_snap("PfrA", week=1), _snap("PfrA", week=19, game_type="POST")],
        [{"gsis_id": "00-001", "pfr_id": "PfrA"}],
    )

    out = _snap_pct(2024, False)

    assert list(out["week"]) == [1]


# ---- base usage schema normalisation ----

def _weekly_native(**overrides):
    """nflverse's own weekly table: abbreviated player_name + full
    player_display_name, and recent_team rather than team."""
    row = {
        "season": 2024, "week": 1, "season_type": "REG", "player_id": "00-001",
        "player_name": "J.Jacobs", "player_display_name": "Josh Jacobs",
        "position": "RB", "recent_team": "GB", "opponent_team": "CHI",
        "target_share": 0.15, "air_yards_share": 0.1, "fantasy_points_ppr": 18.0,
    }
    row.update(overrides)
    return row


def _weekly_reconstructed(**overrides):
    """pbp_fallback's output: already normalised to player_name/team."""
    row = {
        "season": 2024, "week": 1, "season_type": "REG", "player_id": "00-001",
        "player_name": "Josh Jacobs", "position": "RB", "team": "GB",
        "opponent_team": "CHI", "target_share": 0.15, "air_yards_share": 0.1,
        "fantasy_points_ppr": 18.0,
    }
    row.update(overrides)
    return row


def test_native_and_reconstructed_weekly_sources_land_on_the_same_schema(monkeypatch):
    # The fallback path used to KeyError on columns it was never going to
    # have; both sources must produce identical normalised frames.
    def _run(rows):
        monkeypatch.setattr(
            transform.raw, "fetch_weekly_data_or_reconstruct",
            lambda season, force_refresh=False: pd.DataFrame(rows),
        )
        return _base_usage(2024, False)

    native = _run([_weekly_native()])
    reconstructed = _run([_weekly_reconstructed()])

    expected = ["season", "week", "player_id", "player_name", "position", "team",
                "opponent", "target_share", "air_yards_share", "fantasy_points_ppr"]
    assert list(native.columns) == expected
    assert list(reconstructed.columns) == expected
    # The full display name wins over the abbreviated one on the native path.
    assert native.iloc[0]["player_name"] == "Josh Jacobs"
    assert native.iloc[0]["team"] == "GB"
    pd.testing.assert_frame_equal(native, reconstructed)


def test_postseason_rows_are_excluded_from_base_usage(monkeypatch):
    monkeypatch.setattr(
        transform.raw, "fetch_weekly_data_or_reconstruct",
        lambda season, force_refresh=False: pd.DataFrame([
            _weekly_native(week=1),
            _weekly_native(week=19, season_type="POST"),
        ]),
    )

    out = _base_usage(2024, False)

    assert list(out["week"]) == [1]


# ---- full table assembly ----

def test_build_player_week_table_joins_every_source_onto_the_output_contract(monkeypatch):
    monkeypatch.setattr(
        transform.raw, "fetch_weekly_data_or_reconstruct",
        lambda season, force_refresh=False: pd.DataFrame([
            _weekly_native(player_id="00-001"),
            _weekly_native(player_id="00-002", player_display_name="Other Guy"),
        ]),
    )
    _stub_snaps(
        monkeypatch,
        [_snap("PfrA", offense_pct=0.80)],
        [{"gsis_id": "00-001", "pfr_id": "PfrA"}],
    )
    _stub_pbp(monkeypatch, [
        _play(rush_attempt=1, rusher_player_id="00-001"),
        _play(pass_attempt=1, receiver_player_id="00-001"),
    ])

    out = transform.build_player_week_table(2024)

    assert list(out.columns) == OUTPUT_COLUMNS
    by_player = out.set_index("player_id")
    assert by_player.loc["00-001", "snap_pct"] == pytest.approx(0.80)
    assert by_player.loc["00-001", "red_zone_touches"] == 2
    # GB threw exactly one attributable red-zone target, all to 00-001.
    assert by_player.loc["00-001", "red_zone_target_share"] == pytest.approx(1.0)
    # A player with no snap-count row keeps a null snap_pct rather than a
    # fabricated zero, but a real zero red-zone touch count.
    assert pd.isna(by_player.loc["00-002", "snap_pct"])
    assert by_player.loc["00-002", "red_zone_touches"] == 0
