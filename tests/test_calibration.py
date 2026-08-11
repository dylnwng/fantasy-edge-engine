import json

import numpy as np
import pytest

from edge_engine.model import calibration
from edge_engine.model.calibration import (
    MIN_FIT_ROWS,
    Calibrator,
    brier_score,
    evaluate,
    fit_calibrator,
    load_calibrator,
    no_skill_brier,
    reliability_bins,
    save_calibrator,
)


def _separable_sample(n=1000, seed=0):
    """Margins carrying real signal: bigger margin -> likelier to hit."""
    rng = np.random.default_rng(seed)
    margins = rng.normal(0.0, 4.0, n)
    true_p = 1.0 / (1.0 + np.exp(-(0.35 * margins - 0.2)))
    hits = rng.random(n) < true_p
    return margins, hits


# ---- the fitted curve ----

def test_probability_rises_with_margin():
    margins, hits = _separable_sample()
    cal = fit_calibrator(margins, hits)

    assert cal.slope > 0
    assert cal.probability(10.0) > cal.probability(0.0) > cal.probability(-10.0)


def test_recovers_a_known_relationship():
    margins, hits = _separable_sample(n=4000, seed=3)
    cal = fit_calibrator(margins, hits)

    # Generated with slope 0.35, intercept -0.2.
    assert cal.slope == pytest.approx(0.35, abs=0.08)
    assert cal.intercept == pytest.approx(-0.2, abs=0.12)


def test_probabilities_stay_inside_zero_and_one_at_absurd_margins():
    # A 500-point margin is nonsense, but a NaN or a literal 1.0 leaking
    # into the rankings is worse than a clipped number.
    cal = Calibrator(slope=2.0, intercept=0.0, n_train_rows=500, base_rate=0.5)

    for margin in (-1e6, -500.0, 0.0, 500.0, 1e6):
        p = cal.probability(margin)
        assert 0.0 < p < 1.0
        assert np.isfinite(p)


def test_probability_accepts_arrays_and_scalars():
    cal = Calibrator(slope=1.0, intercept=0.0, n_train_rows=500, base_rate=0.5)

    assert isinstance(cal.probability(1.0), float)
    out = cal.probability(np.array([-1.0, 0.0, 1.0]))
    assert out.shape == (3,)
    assert out[0] < out[1] < out[2]


def test_a_zero_margin_with_zero_intercept_is_a_coin_flip():
    cal = Calibrator(slope=1.0, intercept=0.0, n_train_rows=500, base_rate=0.5)
    assert cal.probability(0.0) == pytest.approx(0.5)


# ---- refusing to fit garbage ----

def test_refuses_to_fit_on_too_little_data():
    margins, hits = _separable_sample(n=MIN_FIT_ROWS - 1)

    with pytest.raises(ValueError, match="at least"):
        fit_calibrator(margins, hits)


def test_refuses_when_every_outcome_is_the_same():
    # A sigmoid can only answer by running off to a constant, which is a
    # fabricated certainty rather than a calibration.
    margins = np.linspace(-5, 5, 500)

    with pytest.raises(ValueError, match="all hits"):
        fit_calibrator(margins, np.ones(500, dtype=bool))
    with pytest.raises(ValueError, match="all misses"):
        fit_calibrator(margins, np.zeros(500, dtype=bool))


def test_refuses_non_finite_margins():
    margins, hits = _separable_sample(n=500)
    margins[3] = np.nan

    with pytest.raises(ValueError, match="NaN or inf"):
        fit_calibrator(margins, hits)


def test_refuses_mismatched_lengths():
    margins, hits = _separable_sample(n=500)

    with pytest.raises(ValueError, match="differ in length"):
        fit_calibrator(margins, hits[:-1])


# ---- scoring ----

def test_brier_is_zero_for_perfect_predictions():
    assert brier_score([1.0, 0.0, 1.0], [True, False, True]) == pytest.approx(0.0)


def test_brier_is_one_for_confidently_wrong_predictions():
    assert brier_score([0.0, 1.0], [True, False]) == pytest.approx(1.0)


def test_no_skill_brier_is_what_always_guessing_the_base_rate_scores():
    outcomes = [True] * 3 + [False]  # base rate 0.75
    # Variance of a Bernoulli(0.75): 0.75 * 0.25 = 0.1875
    assert no_skill_brier(outcomes) == pytest.approx(0.1875)


def test_a_real_signal_beats_the_no_skill_baseline():
    margins, hits = _separable_sample(n=3000, seed=5)
    cal = fit_calibrator(margins, hits)

    report = evaluate(cal, margins, hits)

    assert report["brier"] < report["no_skill_brier"]
    assert report["brier_improvement"] > 0


def test_pure_noise_does_not_beat_the_no_skill_baseline():
    # The guard that matters: if the margin carries no signal, the
    # calibration must not claim an improvement. main() refuses to save
    # in this case.
    rng = np.random.default_rng(11)
    margins = rng.normal(0, 4, 3000)
    hits = rng.random(3000) < 0.5  # independent of margin

    cal = fit_calibrator(margins, hits)
    report = evaluate(cal, margins, hits)

    assert report["brier_improvement"] < 0.01


def test_scoring_an_empty_set_raises_rather_than_returning_nan():
    with pytest.raises(ValueError, match="empty"):
        brier_score([], [])
    with pytest.raises(ValueError, match="empty"):
        no_skill_brier([])


# ---- reliability ----

def test_reliability_bins_track_observed_frequency():
    margins, hits = _separable_sample(n=5000, seed=7)
    cal = fit_calibrator(margins, hits)
    probs = cal.probability(margins)

    bins = reliability_bins(probs, hits, n_bins=10)

    assert bins
    for b in bins:
        # Well-calibrated on its own training sample: predicted and
        # observed should track closely.
        assert abs(b["mean_predicted"] - b["observed_rate"]) < 0.1
        assert b["n"] > 0


def test_empty_bins_are_omitted_not_reported_as_zero_percent():
    # An empty bucket reported as 0% observed looks like catastrophic
    # miscalibration where there is simply no data.
    probs = np.full(100, 0.55)
    outcomes = np.ones(100, dtype=bool)

    bins = reliability_bins(probs, outcomes, n_bins=10)

    assert len(bins) == 1
    assert bins[0]["n"] == 100


def test_a_probability_of_exactly_one_lands_in_the_final_bin():
    bins = reliability_bins([1.0, 1.0], [True, False], n_bins=10)

    assert len(bins) == 1
    assert bins[0]["n"] == 2


# ---- artifact round-trip and validation ----

def test_calibrator_survives_a_save_load_round_trip(tmp_path):
    margins, hits = _separable_sample()
    cal = fit_calibrator(margins, hits)
    path = tmp_path / "calibration.json"

    save_calibrator(cal, path)
    loaded = load_calibrator(path)

    assert loaded == cal
    assert loaded.probability(5.0) == pytest.approx(cal.probability(5.0))


def test_a_non_finite_artifact_is_rejected_loudly(tmp_path):
    # NaN parameters would silently turn every probability into NaN, which
    # renders as a blank confidence rather than an error -- the same
    # failure class the team_correlation guard exists for.
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({
        "slope": float("nan"), "intercept": 0.0, "n_train_rows": 500, "base_rate": 0.5,
    }))

    with pytest.raises(ValueError, match="non-finite"):
        load_calibrator(path)


def test_an_incomplete_artifact_names_what_is_missing(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps({"slope": 1.0}))

    with pytest.raises(ValueError, match="missing"):
        load_calibrator(path)


def test_calibrator_available_is_false_without_an_artifact(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "CALIBRATION_PATH", tmp_path / "nope.json")
    assert calibration.calibrator_available() is False


# ---- integration with the ranking output ----

def test_rankings_degrade_to_tiers_only_when_nothing_is_fitted(monkeypatch):
    from edge_engine.ranking import output

    monkeypatch.setattr(calibration, "calibrator_available", lambda: False)

    assert output.load_hit_probability_fn() is None


def test_a_corrupt_artifact_is_ignored_rather_than_breaking_the_rankings(monkeypatch, caplog):
    # An optional extra column must never cost you the weekly rankings.
    from edge_engine.ranking import output

    monkeypatch.setattr(calibration, "calibrator_available", lambda: True)

    def _raise():
        raise ValueError("calibration artifact has non-finite parameters")

    monkeypatch.setattr(calibration, "load_calibrator", _raise)

    import logging

    with caplog.at_level(logging.WARNING):
        assert output.load_hit_probability_fn() is None
    assert "Ignoring calibration artifact" in caplog.text


def test_a_fitted_calibrator_is_exposed_as_a_margin_to_probability_fn(monkeypatch):
    from edge_engine.ranking import output

    cal = Calibrator(slope=1.0, intercept=0.0, n_train_rows=500, base_rate=0.5)
    monkeypatch.setattr(calibration, "calibrator_available", lambda: True)
    monkeypatch.setattr(calibration, "load_calibrator", lambda: cal)

    fn = output.load_hit_probability_fn()

    assert fn is not None
    assert fn(0.0) == pytest.approx(0.5)
    assert fn(3.0) > 0.9
