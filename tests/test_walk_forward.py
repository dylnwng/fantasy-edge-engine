import numpy as np
import pandas as pd
import pytest

from edge_engine.model.walk_forward import (
    Fold,
    bootstrap_mae_diff,
    evaluate_fold,
    run_walk_forward,
    walk_forward_folds,
)

FEAT_COLS = ["trailing_snap_pct_avg"]


# ---- fold generation ----

def test_folds_expand_and_never_include_the_validation_season():
    folds = walk_forward_folds([2018, 2019, 2020, 2021], min_train_seasons=2)

    assert [f.validation_season for f in folds] == [2020, 2021]
    assert folds[0].train_seasons == (2018, 2019)
    assert folds[1].train_seasons == (2018, 2019, 2020)
    for f in folds:
        assert f.validation_season not in f.train_seasons


def test_no_fold_trains_on_a_later_season_than_it_validates():
    # The whole point of walk-forward: a fold must never see the future.
    for f in walk_forward_folds(range(2018, 2026), min_train_seasons=2):
        assert max(f.train_seasons) < f.validation_season


def test_earliest_seasons_are_not_validated_without_enough_history():
    folds = walk_forward_folds([2018, 2019, 2020], min_train_seasons=2)
    assert [f.validation_season for f in folds] == [2020]


def test_too_few_seasons_yields_no_folds_rather_than_raising():
    # A caller with two seasons asked a reasonable question; the honest
    # answer is "not enough data", not a traceback.
    assert walk_forward_folds([2024, 2025], min_train_seasons=2) == []


def test_seasons_are_deduped_and_sorted_regardless_of_input_order():
    folds = walk_forward_folds([2021, 2019, 2020, 2019], min_train_seasons=1)
    assert [f.validation_season for f in folds] == [2020, 2021]
    assert folds[0].train_seasons == (2019,)


def test_nonsense_min_train_seasons_raises():
    with pytest.raises(ValueError, match="min_train_seasons"):
        walk_forward_folds([2018, 2019, 2020], min_train_seasons=0)


# ---- fold evaluation ----

def _table(rows):
    return pd.DataFrame(rows)


def _row(season, feat, label, trailing_points_avg):
    return {
        "season": season,
        "trailing_snap_pct_avg": feat,
        "trailing_points_avg": trailing_points_avg,
        "label_next_week_points": label,
    }


def _constant_prediction(value):
    def _fp(train_df, val_df, feat_cols):
        return np.full(len(val_df), float(value))
    return _fp


def test_fold_metrics_match_hand_calculation():
    table = _table([
        _row(2019, 0.5, 10.0, 8.0),
        # Validation rows: truth 10 and 20, baseline 8 and 12.
        _row(2020, 0.5, 10.0, 8.0),
        _row(2020, 0.5, 20.0, 12.0),
    ])
    fold = Fold(train_seasons=(2019,), validation_season=2020)

    result, model_err, baseline_err = evaluate_fold(
        table, fold, FEAT_COLS, flag_margin=3.0, fit_predict=_constant_prediction(15.0)
    )

    # Model predicts 15 for both: |15-10| = 5, |15-20| = 5 -> MAE 5.
    assert result.model_mae == pytest.approx(5.0)
    # Baseline is trailing_points_avg: |8-10| = 2, |12-20| = 8 -> MAE 5.
    assert result.baseline_mae == pytest.approx(5.0)
    assert result.mae_improvement == pytest.approx(0.0)
    assert list(model_err) == [5.0, 5.0]
    assert list(baseline_err) == [2.0, 8.0]
    assert result.n_validation_rows == 2
    assert result.n_train_rows == 1


def test_only_players_clearing_the_margin_are_flagged_and_scored_for_hit_rate():
    table = _table([
        _row(2019, 0.5, 10.0, 8.0),
        # margin = 15 - 8 = 7 > 3 -> flagged. Truth 10 > baseline 8 -> a hit.
        _row(2020, 0.5, 10.0, 8.0),
        # margin = 15 - 14 = 1, below 3 -> not flagged, excluded from hit rate.
        _row(2020, 0.5, 1.0, 14.0),
    ])
    fold = Fold(train_seasons=(2019,), validation_season=2020)

    result, _, _ = evaluate_fold(
        table, fold, FEAT_COLS, flag_margin=3.0, fit_predict=_constant_prediction(15.0)
    )

    assert result.n_flagged == 1
    assert result.hit_rate == pytest.approx(1.0)


def test_hit_rate_is_none_when_nothing_clears_the_margin():
    # None, not 0.0 -- "nothing was flagged" is a different statement from
    # "everything flagged missed", and averaging a fabricated 0.0 into the
    # pooled hit rate would understate the model.
    table = _table([_row(2019, 0.5, 10.0, 8.0), _row(2020, 0.5, 10.0, 8.0)])
    fold = Fold(train_seasons=(2019,), validation_season=2020)

    result, _, _ = evaluate_fold(
        table, fold, FEAT_COLS, flag_margin=99.0, fit_predict=_constant_prediction(15.0)
    )

    assert result.n_flagged == 0
    assert result.hit_rate is None


def test_a_fold_whose_seasons_are_missing_from_the_data_raises_clearly():
    table = _table([_row(2019, 0.5, 10.0, 8.0)])
    fold = Fold(train_seasons=(2019,), validation_season=2020)  # 2020 not ingested

    with pytest.raises(ValueError, match="empty validation set"):
        evaluate_fold(table, fold, FEAT_COLS, flag_margin=3.0, fit_predict=_constant_prediction(1.0))


def test_fit_predict_receives_only_the_folds_own_training_seasons():
    # Guards the leak this whole module exists to prevent.
    seen = {}

    def _spy(train_df, val_df, feat_cols):
        seen["train"] = sorted(train_df["season"].unique())
        seen["val"] = sorted(val_df["season"].unique())
        return np.zeros(len(val_df))

    table = _table([_row(s, 0.5, 10.0, 8.0) for s in (2018, 2019, 2020, 2021)])
    evaluate_fold(
        table, Fold(train_seasons=(2018, 2019), validation_season=2020),
        FEAT_COLS, flag_margin=3.0, fit_predict=_spy,
    )

    assert seen["train"] == [2018, 2019]
    assert seen["val"] == [2020]
    assert 2021 not in seen["train"]  # no future season leaked in


# ---- bootstrap ----

def test_bootstrap_favours_the_model_when_it_is_clearly_better():
    model_err = np.full(200, 1.0)
    baseline_err = np.full(200, 3.0)

    boot = bootstrap_mae_diff(model_err, baseline_err, n_boot=200)

    assert boot["mean_mae_improvement"] == pytest.approx(2.0)
    assert boot["fraction_favoring_model"] == 1.0
    assert boot["ci_5th"] > 0


def test_bootstrap_ci_straddles_zero_when_there_is_no_real_difference():
    rng = np.random.default_rng(1)
    err = rng.normal(5.0, 1.0, 400)

    boot = bootstrap_mae_diff(err, err.copy(), n_boot=300)

    assert boot["mean_mae_improvement"] == pytest.approx(0.0, abs=1e-9)
    assert boot["ci_5th"] <= 0 <= boot["ci_95th"]


def test_bootstrap_is_deterministic_for_a_given_seed():
    rng = np.random.default_rng(0)
    m, b = rng.normal(4, 1, 100), rng.normal(5, 1, 100)

    assert bootstrap_mae_diff(m, b, seed=7, n_boot=100) == bootstrap_mae_diff(m, b, seed=7, n_boot=100)


def test_bootstrap_rejects_misaligned_or_empty_input():
    with pytest.raises(ValueError, match="aligned"):
        bootstrap_mae_diff(np.ones(3), np.ones(4))
    with pytest.raises(ValueError, match="at least one row"):
        bootstrap_mae_diff(np.array([]), np.array([]))


# ---- end-to-end orchestration ----

def _multi_season_table():
    rows = []
    for season in (2018, 2019, 2020, 2021):
        for _ in range(20):
            rows.append(_row(season, 0.5, 10.0, 8.0))
    return _table(rows)


def test_run_walk_forward_reports_every_fold_and_pools_them():
    report = run_walk_forward(
        _multi_season_table(), FEAT_COLS, flag_margin=3.0,
        min_train_seasons=2, fit_predict=_constant_prediction(10.0),
    )

    assert report["n_folds"] == 2
    assert report["validation_seasons"] == [2020, 2021]
    assert report["pooled_n_rows"] == 40  # 20 validation rows per fold
    # Model predicts truth exactly (10.0) -> MAE 0; baseline is 8 vs 10 -> 2.
    assert report["pooled_model_mae"] == pytest.approx(0.0)
    assert report["pooled_baseline_mae"] == pytest.approx(2.0)


def test_fold_agreement_counts_seasons_not_rows():
    # The headline number: an effect that is real replicates across
    # seasons, and this is what distinguishes that from one season's
    # weather. Model is better in both folds here.
    report = run_walk_forward(
        _multi_season_table(), FEAT_COLS, flag_margin=3.0,
        min_train_seasons=2, fit_predict=_constant_prediction(10.0),
    )

    assert report["folds_favoring_model"] == 2
    assert report["fold_agreement"] == pytest.approx(1.0)


def test_fold_agreement_registers_a_model_that_is_worse():
    report = run_walk_forward(
        _multi_season_table(), FEAT_COLS, flag_margin=3.0,
        min_train_seasons=2, fit_predict=_constant_prediction(100.0),
    )

    assert report["folds_favoring_model"] == 0
    assert report["pooled_bootstrap"]["mean_mae_improvement"] < 0


def test_run_walk_forward_refuses_when_there_is_not_enough_history():
    two_seasons = _table([_row(s, 0.5, 10.0, 8.0) for s in (2024, 2025)])

    with pytest.raises(ValueError, match="No folds available"):
        run_walk_forward(two_seasons, FEAT_COLS, flag_margin=3.0, min_train_seasons=2)


def test_worst_fold_hit_rate_is_reported_not_just_the_mean():
    # A mean hit rate can hide a season where the model was useless; the
    # project's whole lesson is that the bad season is the informative one.
    report = run_walk_forward(
        _multi_season_table(), FEAT_COLS, flag_margin=3.0,
        min_train_seasons=2, fit_predict=_constant_prediction(20.0),
    )

    assert report["worst_fold_hit_rate"] is not None
    assert report["worst_fold_hit_rate"] <= report["mean_hit_rate"]
