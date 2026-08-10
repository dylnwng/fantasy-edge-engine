"""Walk-forward (rolling-origin) evaluation of the opportunity model.

Why this exists
---------------
`train.py` validates on ONE held-out season. At the ~300-450 flagged
players a single season produces, that is not enough resolution to
separate a real 1-2pp effect from noise -- which is the finding
EVALUATION.md arrived at the hard way, after a usage-persistence feature
looked like a clear win on 2024 (+1.9pp hit rate, 87% of bootstrap
resamples favouring) and then reversed sign on 2025 (-1.4pp, 20%).

The conclusion drawn there was "treat any feature that only looks good on
one season as noise." This module is the other half of that: an
evaluation that spans every season available, so an effect gets several
independent chances to replicate instead of one. It doesn't make the
model better. It makes the *verdicts* about the model trustworthy enough
to act on.

Protocol
--------
Expanding-window walk-forward. For each validation season N, train on
every ingested season strictly before N:

    train 2018-2019          -> validate 2020
    train 2018-2020          -> validate 2021
    ...                      -> ...
    train 2018-2024          -> validate 2025

No fold ever sees its own future, and no fold sees any *later* season,
so every prediction is genuinely out of sample with respect to the model
that made it.

Two verdicts, deliberately reported side by side
------------------------------------------------
1. **Pooled bootstrap CI** over all validation rows, paired on the same
   players -- matching scripts/validate_ros.py's convention so numbers
   from the two are comparable.

2. **Fold agreement**: how many individual seasons favour the model.

The second is the stricter test and the one to believe. The pooled CI
treats rows as exchangeable, but rows inside one season are correlated
(shared teams, shared scoring environment, the same player appearing
weekly), so it is optimistic -- it will report a tighter interval than
the evidence really supports. An effect that is real shows up in most
folds independently. An effect that is a single season's weather shows
up pooled and nowhere else. When the two disagree, the fold count wins.

A cluster bootstrap over whole seasons would be the textbook fix for
that correlation, but with only a handful of folds it has too few
clusters to estimate anything stable, so it is deliberately not offered
here rather than offered and quietly meaningless.

This never writes a model artifact. It trains in memory, per fold, and
leaves models/ untouched -- running it cannot disturb what predict.py
loads.

Usage:
    python -m edge_engine.model.walk_forward
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from edge_engine.model.config import ModelConfig, load_model_config
from edge_engine.model.features import feature_columns
from edge_engine.model.train import build_estimator, build_training_table

logger = logging.getLogger(__name__)

N_BOOT = 2000
DEFAULT_MIN_TRAIN_SEASONS = 2

# (train_df, val_df, feat_cols) -> predictions aligned to val_df's rows.
FitPredict = Callable[[pd.DataFrame, pd.DataFrame, list[str]], np.ndarray]


@dataclass(frozen=True)
class Fold:
    train_seasons: tuple[int, ...]
    validation_season: int


@dataclass(frozen=True)
class FoldResult:
    validation_season: int
    n_train_rows: int
    n_validation_rows: int
    model_mae: float
    baseline_mae: float
    n_flagged: int
    hit_rate: float | None

    @property
    def mae_improvement(self) -> float:
        """Positive means the model beat the naive baseline this season."""
        return self.baseline_mae - self.model_mae


def walk_forward_folds(
    seasons: Sequence[int], min_train_seasons: int = DEFAULT_MIN_TRAIN_SEASONS
) -> list[Fold]:
    """Expanding-window folds, oldest validation season first.

    A fold is only produced once at least `min_train_seasons` earlier
    seasons exist to train on -- the earliest seasons make poor training
    sets on their own, and a fold trained on one season would report a
    weak result that says more about the fold than about the model.

    Returns an empty list rather than raising when there isn't enough
    history: a caller asking to evaluate two seasons has asked a
    reasonable question with the honest answer "not enough data yet."
    """
    if min_train_seasons < 1:
        raise ValueError(f"min_train_seasons must be >= 1, got {min_train_seasons}")

    ordered = sorted(set(int(s) for s in seasons))
    folds: list[Fold] = []
    for i, season in enumerate(ordered):
        train = tuple(ordered[:i])
        if len(train) >= min_train_seasons:
            folds.append(Fold(train_seasons=train, validation_season=season))
    return folds


def _default_fit_predict(train_df: pd.DataFrame, val_df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    """Trains the SHIPPED estimator (train.build_estimator) so what's
    measured here is the model that actually ships, not a lookalike."""
    model = build_estimator()
    model.fit(train_df[feat_cols], train_df["label_next_week_points"])
    return model.predict(val_df[feat_cols])


def evaluate_fold(
    table: pd.DataFrame,
    fold: Fold,
    feat_cols: list[str],
    flag_margin: float,
    fit_predict: FitPredict | None = None,
) -> tuple[FoldResult, np.ndarray, np.ndarray]:
    """Train on the fold's seasons, score its validation season.

    Returns the fold's summary plus the per-row absolute errors for the
    model and the baseline, which the caller pools for the bootstrap.
    Both error arrays are aligned to the same rows, so the pairing the
    bootstrap depends on holds.

    Metrics mirror train.train_and_evaluate exactly -- same MAE, same
    flagging rule, same precision-oriented hit rate -- so a fold here is
    directly comparable to the single-season number train.py reports.
    """
    fit_predict = fit_predict or _default_fit_predict

    train_df = table[table["season"].isin(fold.train_seasons)]
    val_df = table[table["season"] == fold.validation_season]
    if train_df.empty or val_df.empty:
        raise ValueError(
            f"Fold validating on {fold.validation_season} has an empty "
            f"{'train' if train_df.empty else 'validation'} set -- "
            "the seasons requested aren't all present in the ingested data."
        )

    val_df = val_df.copy()
    val_df["predicted_score"] = fit_predict(train_df, val_df, feat_cols)
    val_df["baseline_score"] = val_df["trailing_points_avg"]

    truth = val_df["label_next_week_points"]
    model_err = (val_df["predicted_score"] - truth).abs().to_numpy()
    baseline_err = (val_df["baseline_score"] - truth).abs().to_numpy()

    flagged = val_df[val_df["predicted_score"] - val_df["baseline_score"] > flag_margin]
    hit_rate = (
        float((flagged["label_next_week_points"] > flagged["baseline_score"]).mean())
        if len(flagged)
        else None
    )

    result = FoldResult(
        validation_season=fold.validation_season,
        n_train_rows=int(len(train_df)),
        n_validation_rows=int(len(val_df)),
        model_mae=float(model_err.mean()),
        baseline_mae=float(baseline_err.mean()),
        n_flagged=int(len(flagged)),
        hit_rate=hit_rate,
    )
    return result, model_err, baseline_err


def bootstrap_mae_diff(
    model_err: np.ndarray, baseline_err: np.ndarray, seed: int = 0, n_boot: int = N_BOOT
) -> dict:
    """Paired bootstrap on (baseline MAE - model MAE); positive favours
    the model. Paired because both are scored on the same rows.

    Same construction and same reported keys as
    scripts/validate_ros.py's, so a CI from that script and a CI from
    this one mean the same thing.
    """
    if len(model_err) != len(baseline_err):
        raise ValueError(
            f"paired bootstrap needs aligned arrays, got {len(model_err)} and {len(baseline_err)}"
        )
    if len(model_err) == 0:
        raise ValueError("paired bootstrap needs at least one row")

    rng = np.random.default_rng(seed)
    n = len(model_err)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[i] = baseline_err[idx].mean() - model_err[idx].mean()

    return {
        "mean_mae_improvement": float(diffs.mean()),
        "ci_5th": float(np.percentile(diffs, 5)),
        "ci_95th": float(np.percentile(diffs, 95)),
        "fraction_favoring_model": float((diffs > 0).mean()),
    }


def run_walk_forward(
    table: pd.DataFrame,
    feat_cols: list[str],
    flag_margin: float,
    seasons: Sequence[int] | None = None,
    min_train_seasons: int = DEFAULT_MIN_TRAIN_SEASONS,
    fit_predict: FitPredict | None = None,
    seed: int = 0,
) -> dict:
    """Evaluate across every fold and pool the result.

    `table` is a built feature table with NaN feature/label rows already
    dropped. `fit_predict` is the extension point: pass a different one
    to put a candidate feature set or model through the identical
    protocol, which is the point of having this -- the experiments
    EVALUATION.md rejected were each judged on a single season, and
    several of those verdicts are worth revisiting under a harness that
    can actually resolve them.
    """
    seasons = sorted(table["season"].unique()) if seasons is None else seasons
    folds = walk_forward_folds(seasons, min_train_seasons=min_train_seasons)
    if not folds:
        raise ValueError(
            f"No folds available: {len(set(seasons))} season(s) present but each fold "
            f"needs {min_train_seasons} training season(s) before it. Ingest more seasons."
        )

    results: list[FoldResult] = []
    model_errs: list[np.ndarray] = []
    baseline_errs: list[np.ndarray] = []
    for fold in folds:
        result, m_err, b_err = evaluate_fold(table, fold, feat_cols, flag_margin, fit_predict)
        logger.info(
            "%s: model MAE %.3f vs baseline %.3f (%+.3f), %d flagged, hit rate %s",
            fold.validation_season, result.model_mae, result.baseline_mae,
            result.mae_improvement, result.n_flagged,
            f"{result.hit_rate:.1%}" if result.hit_rate is not None else "n/a",
        )
        results.append(result)
        model_errs.append(m_err)
        baseline_errs.append(b_err)

    pooled_model = np.concatenate(model_errs)
    pooled_baseline = np.concatenate(baseline_errs)

    folds_favoring = sum(1 for r in results if r.mae_improvement > 0)
    rated = [r.hit_rate for r in results if r.hit_rate is not None]

    return {
        "protocol": "expanding-window walk-forward",
        "n_folds": len(results),
        "validation_seasons": [r.validation_season for r in results],
        "min_train_seasons": min_train_seasons,
        "flag_margin": flag_margin,
        "folds": [asdict(r) | {"mae_improvement": r.mae_improvement} for r in results],
        # The stricter, season-level replication test. Believe this one.
        "folds_favoring_model": folds_favoring,
        "fold_agreement": folds_favoring / len(results),
        "pooled_n_rows": int(len(pooled_model)),
        "pooled_model_mae": float(pooled_model.mean()),
        "pooled_baseline_mae": float(pooled_baseline.mean()),
        "total_flagged": int(sum(r.n_flagged for r in results)),
        "mean_hit_rate": float(np.mean(rated)) if rated else None,
        "worst_fold_hit_rate": float(np.min(rated)) if rated else None,
        # Optimistic: rows within a season are correlated. See module docstring.
        "pooled_bootstrap": bootstrap_mae_diff(pooled_model, pooled_baseline, seed=seed),
    }


def load_evaluation_table(config: ModelConfig | None = None) -> tuple[pd.DataFrame, list[str]]:
    """The same feature table train.py trains on, NaN rows dropped.

    Reuses train.build_training_table so the harness can't quietly
    evaluate a differently-built table than the one that ships.
    """
    config = config or load_model_config()
    table = build_training_table(config)
    feat_cols = feature_columns(config.trailing_window)
    usable = table.dropna(subset=[*feat_cols, "label_next_week_points"])
    return usable, feat_cols


def _print_report(report: dict) -> None:
    print(f"\nWalk-forward evaluation — {report['n_folds']} folds, "
          f"{report['pooled_n_rows']:,} validation rows\n")
    header = f"{'Season':<8}{'Model':>9}{'Baseline':>10}{'Diff':>9}{'Flagged':>9}{'Hit rate':>10}"
    print(header)
    print("-" * len(header))
    for f in report["folds"]:
        hit = f"{f['hit_rate']:.1%}" if f["hit_rate"] is not None else "n/a"
        print(
            f"{f['validation_season']:<8}{f['model_mae']:>9.3f}{f['baseline_mae']:>10.3f}"
            f"{f['mae_improvement']:>+9.3f}{f['n_flagged']:>9}{hit:>10}"
        )
    print("-" * len(header))
    print(
        f"{'Pooled':<8}{report['pooled_model_mae']:>9.3f}"
        f"{report['pooled_baseline_mae']:>10.3f}"
        f"{report['pooled_baseline_mae'] - report['pooled_model_mae']:>+9.3f}"
        f"{report['total_flagged']:>9}"
        f"{(f'{report['mean_hit_rate']:.1%}' if report['mean_hit_rate'] is not None else 'n/a'):>10}"
    )

    boot = report["pooled_bootstrap"]
    print(
        f"\nFold agreement: {report['folds_favoring_model']}/{report['n_folds']} seasons favour "
        f"the model  <- the test to believe"
    )
    print(
        f"Pooled bootstrap: {boot['mean_mae_improvement']:+.3f} MAE "
        f"[90% CI {boot['ci_5th']:+.3f}, {boot['ci_95th']:+.3f}], "
        f"{boot['fraction_favoring_model']:.0%} of resamples favour the model"
    )
    print("  (pooled CI is optimistic — rows within a season are correlated)")
    if report["worst_fold_hit_rate"] is not None:
        print(f"Worst single season hit rate: {report['worst_fold_hit_rate']:.1%}")
    print()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_model_config()
    table, feat_cols = load_evaluation_table(config)
    report = run_walk_forward(table, feat_cols, flag_margin=config.flag_margin)
    _print_report(report)


if __name__ == "__main__":
    main()
