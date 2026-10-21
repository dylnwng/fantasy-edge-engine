"""Turn the model's margin into a probability you can actually check.

The problem
-----------
`confidence_tier` sorts a candidate into High/Medium/Low by comparing
`predicted_score - baseline_score` against hand-set multiples of
`flag_margin`. Those thresholds were chosen, not measured. A tier is
also unfalsifiable: "High" makes no claim that can be wrong, so nobody
can ever discover it is miscalibrated. That is the opposite of how the
rest of this project treats its numbers -- the matchup simulator reports
a Brier score against a no-skill baseline precisely so its confidence can
be graded.

What this does
--------------
Fits a one-dimensional Platt calibration -- P(hit) = sigmoid(a*margin+b)
-- mapping the margin the model already produces onto the probability
that the player beats his own baseline next week. "Beats his own
baseline" is deliberately the same event train.py's hit_rate measures and
history.py grades, so a probability here is a prediction about something
the project already tracks and can be scored against later.

Two parameters, on purpose. Isotonic regression would fit the training
curve more closely, but with a few hundred players per season it buys
wiggle in the tails -- exactly where the interesting candidates live --
and this project has been burned by flexible fits that looked good on one
season. A monotone two-parameter curve cannot invent structure that isn't
there, and it extrapolates sanely past the largest margin ever observed.

Fit it on OUT-OF-SAMPLE margins
-------------------------------
`walk_forward.walk_forward_predictions()` supplies those. Fitting on
in-sample margins produces a curve that promises accuracy the model does
not have on new data -- the margins it trained on are too good, so the
sigmoid learns to trust them more than it should.

Absent a fitted artifact everything degrades to the existing tier
behaviour, the same silent no-op predict_qb uses when its model is
missing.

Usage:
    python -m edge_engine.model.calibration        # fit, evaluate, save
    python -m edge_engine.model.calibration --dry-run   # report, write nothing
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from edge_engine.paths import ROOT_DIR

logger = logging.getLogger(__name__)

CALIBRATION_PATH = ROOT_DIR / "models" / "calibration.json"

# Below this, a two-parameter fit is being asked to learn more than the
# data supports. Chosen to be roughly a full season of scoreable rows --
# note this fits on EVERY scored row, not just the flagged ones, so it is
# a far larger sample than the ~300-450 flagged players a season yields.
MIN_FIT_ROWS = 200

# Probabilities are clipped away from the endpoints. A reported 0.0 or
# 1.0 is a certainty this model is in no position to claim, and it makes
# log-based diagnostics infinite.
_EPS = 1e-6

# Sigmoid saturates to float precision well before this; clipping here
# keeps np.exp away from its overflow threshold (~709).
_MAX_LOGIT = 40.0


@dataclass(frozen=True)
class Calibrator:
    """P(beats own baseline) = sigmoid(slope * margin + intercept)."""

    slope: float
    intercept: float
    n_train_rows: int
    base_rate: float

    def probability(self, margin):
        """Probability for a margin (scalar or array). Returns the same
        shape it was given."""
        arr = np.asarray(margin, dtype=float)
        # Clip the logit, not just the result. np.exp overflows past ~709
        # and emits a RuntimeWarning on the way to the right answer --
        # noise in the console for a number that saturates long before
        # then anyway (sigmoid(±40) is already 1 or 0 to float precision).
        z = np.clip(self.slope * arr + self.intercept, -_MAX_LOGIT, _MAX_LOGIT)
        probs = 1.0 / (1.0 + np.exp(-z))
        probs = np.clip(probs, _EPS, 1.0 - _EPS)
        return float(probs) if np.isscalar(margin) or arr.ndim == 0 else probs

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict) -> "Calibrator":
        missing = {"slope", "intercept", "n_train_rows", "base_rate"} - set(raw)
        if missing:
            raise ValueError(f"calibration artifact is missing {sorted(missing)}")
        slope, intercept = float(raw["slope"]), float(raw["intercept"])
        if not (np.isfinite(slope) and np.isfinite(intercept)):
            # A non-finite parameter silently turns every probability into
            # NaN, which downstream would render as a blank confidence
            # rather than an error -- the same class of failure the
            # team_correlation guard in simulation/config.py exists for.
            raise ValueError(
                f"calibration artifact has non-finite parameters "
                f"(slope={slope}, intercept={intercept}) -- refit it."
            )
        return cls(
            slope=slope,
            intercept=intercept,
            n_train_rows=int(raw["n_train_rows"]),
            base_rate=float(raw["base_rate"]),
        )


def fit_calibrator(margins, outcomes) -> Calibrator:
    """Fit sigmoid(a*margin+b) against observed hit/miss outcomes.

    `margins` are predicted_score - baseline_score; `outcomes` are
    booleans for whether the player actually beat that baseline.
    """
    margins = np.asarray(margins, dtype=float)
    outcomes = np.asarray(outcomes).astype(bool)

    if len(margins) != len(outcomes):
        raise ValueError(f"margins and outcomes differ in length: {len(margins)} vs {len(outcomes)}")
    if len(margins) < MIN_FIT_ROWS:
        raise ValueError(
            f"need at least {MIN_FIT_ROWS} rows to calibrate, got {len(margins)} -- "
            "ingest more seasons or widen the walk-forward folds."
        )
    if not np.isfinite(margins).all():
        raise ValueError("margins contain NaN or inf -- drop those rows before fitting.")

    n_hits = int(outcomes.sum())
    if n_hits == 0 or n_hits == len(outcomes):
        # Every row the same class: a sigmoid can only respond by running
        # off to a constant 0 or 1, which is a fabricated certainty rather
        # than a calibration.
        raise ValueError(
            f"outcomes are all {'hits' if n_hits else 'misses'} -- "
            "nothing to calibrate against."
        )

    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression()
    model.fit(margins.reshape(-1, 1), outcomes)

    return Calibrator(
        slope=float(model.coef_[0][0]),
        intercept=float(model.intercept_[0]),
        n_train_rows=int(len(margins)),
        base_rate=float(outcomes.mean()),
    )


def brier_score(probabilities, outcomes) -> float:
    """Mean squared error of the probabilities. Lower is better."""
    probs = np.asarray(probabilities, dtype=float)
    truth = np.asarray(outcomes).astype(float)
    if len(probs) != len(truth):
        raise ValueError(f"probabilities and outcomes differ in length: {len(probs)} vs {len(truth)}")
    if len(probs) == 0:
        raise ValueError("cannot score an empty set of predictions")
    return float(np.mean((probs - truth) ** 2))


def no_skill_brier(outcomes) -> float:
    """What you'd score by ignoring the model and always predicting the
    base rate. The number to beat -- the same framing the matchup
    simulator uses when it reports 0.220 against a no-skill 0.25."""
    truth = np.asarray(outcomes).astype(float)
    if len(truth) == 0:
        raise ValueError("cannot score an empty set of predictions")
    base = float(truth.mean())
    return float(np.mean((base - truth) ** 2))


def reliability_bins(probabilities, outcomes, n_bins: int = 10) -> list[dict]:
    """Predicted vs observed frequency, bucketed. This is the actual test
    of calibration: of the candidates given ~70%, did about 70% hit?

    Empty buckets are omitted rather than reported as 0% observed, which
    would look like a catastrophic miscalibration where there is simply no
    data.
    """
    probs = np.asarray(probabilities, dtype=float)
    truth = np.asarray(outcomes).astype(float)
    if len(probs) != len(truth):
        raise ValueError(f"probabilities and outcomes differ in length: {len(probs)} vs {len(truth)}")

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Upper-inclusive on the final bin so p == 1.0 is not dropped.
        mask = (probs >= lo) & (probs < hi) if hi < 1.0 else (probs >= lo) & (probs <= hi)
        if not mask.any():
            continue
        bins.append({
            "lower": float(lo),
            "upper": float(hi),
            "n": int(mask.sum()),
            "mean_predicted": float(probs[mask].mean()),
            "observed_rate": float(truth[mask].mean()),
        })
    return bins


def evaluate(calibrator: Calibrator, margins, outcomes, n_bins: int = 10) -> dict:
    """Everything needed to decide whether to trust the calibration."""
    probs = calibrator.probability(np.asarray(margins, dtype=float))
    brier = brier_score(probs, outcomes)
    no_skill = no_skill_brier(outcomes)
    bins = reliability_bins(probs, outcomes, n_bins=n_bins)
    max_gap = max((abs(b["mean_predicted"] - b["observed_rate"]) for b in bins), default=0.0)
    return {
        "n_rows": int(len(probs)),
        "base_rate": float(np.asarray(outcomes).astype(float).mean()),
        "brier": brier,
        "no_skill_brier": no_skill,
        # Positive means the calibrated probabilities beat always
        # predicting the base rate. Negative means they are worse than
        # saying nothing, and should not ship.
        "brier_improvement": no_skill - brier,
        "max_calibration_gap": float(max_gap),
        "bins": bins,
    }


def calibrator_available() -> bool:
    """Whether a fitted artifact exists. Callers check this so a checkout
    that never fit one degrades to the existing tier behaviour instead of
    failing -- same shape as predict_qb.qb_model_available()."""
    return CALIBRATION_PATH.exists()


def load_calibrator(path: Path | None = None) -> Calibrator:
    path = path or CALIBRATION_PATH
    with open(path) as f:
        return Calibrator.from_dict(json.load(f))


def save_calibrator(calibrator: Calibrator, path: Path | None = None) -> None:
    path = path or CALIBRATION_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(calibrator.to_dict(), f, indent=2)


def _print_evaluation(report: dict) -> None:
    print(f"\nCalibrated on {report['n_rows']:,} out-of-sample rows "
          f"(base rate {report['base_rate']:.1%})\n")
    print(f"Brier score:      {report['brier']:.4f}")
    print(f"No-skill Brier:   {report['no_skill_brier']:.4f}")
    print(f"Improvement:      {report['brier_improvement']:+.4f}"
          f"{'' if report['brier_improvement'] > 0 else '   <- WORSE THAN SAYING NOTHING'}")
    print(f"Worst bin gap:    {report['max_calibration_gap']:.1%}\n")

    header = f"{'Predicted':>12}{'Observed':>11}{'N':>8}"
    print(header)
    print("-" * len(header))
    for b in report["bins"]:
        print(f"{b['mean_predicted']:>12.1%}{b['observed_rate']:>11.1%}{b['n']:>8,}")
    print()


def main() -> None:
    import argparse

    from edge_engine.model.config import load_model_config
    from edge_engine.model.walk_forward import load_evaluation_table, walk_forward_predictions

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="fit and report calibration quality without writing the artifact",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_model_config()
    table, feat_cols = load_evaluation_table(config)

    logger.info("Generating out-of-sample predictions via walk-forward folds...")
    oos = walk_forward_predictions(table, feat_cols)

    calibrator = fit_calibrator(oos["margin"], oos["beat_baseline"])
    report = evaluate(calibrator, oos["margin"], oos["beat_baseline"])
    _print_evaluation(report)

    if report["brier_improvement"] <= 0:
        # Refuse rather than ship probabilities that are worse than
        # quoting the base rate. A miscalibrated number carries more
        # authority than the tier it would replace, so shipping it would
        # be a downgrade dressed as an upgrade.
        raise SystemExit(
            "Not saving: calibrated probabilities score worse than always predicting "
            "the base rate. The margin carries no usable probability signal as fitted."
        )

    if args.dry_run:
        print("--dry-run: not written.")
        return

    save_calibrator(calibrator)
    print(f"Saved to {CALIBRATION_PATH}")


if __name__ == "__main__":
    main()
