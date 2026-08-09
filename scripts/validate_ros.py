"""The §2.4 validation protocol, and the §2.5 kill gate.

Non-negotiable per the PRD, and the gate is the entire point of the
document: if the ROS model does not beat both baselines with a bootstrap
CI excluding zero on 2024 AND the same sign on 2025, Phase 4 and 5-full
do not ship.

Protocol:
  1. Origins at week 4, 8 and 12 reported SEPARATELY. A model that works
     at week 12 and fails at week 4 is not a ROS model -- it's a
     season-to-date average with extra steps.
  2. Two baselines: season-to-date PPG (the naive carry-forward), and
     the positional prior (replacement level).
  3. MAE and Spearman rank correlation against realised ROS PPG.
  4. Bootstrap CI on the MAE difference vs each baseline; must exclude
     zero, and the sign must replicate across both seasons.

Usage:
    python scripts/validate_ros.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from edge_engine.model.scoring import compute_points_for_seasons  # noqa: E402
from edge_engine.paths import PLAYER_WEEK_PATH  # noqa: E402
from edge_engine.roster.interface import get_default_source  # noqa: E402
from edge_engine.experiments.ros import build_ros_table, positional_prior, project  # noqa: E402

ORIGINS = [4, 8, 12]
N_BOOT = 2000


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation without scipy (this project deliberately has no
    scipy dependency)."""
    if len(a) < 3:
        return float("nan")
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra**2).sum() * (rb**2).sum())
    return float((ra * rb).sum() / denom) if denom else float("nan")


def _bootstrap_mae_diff(model_err, baseline_err, seed=0, n_boot=N_BOOT) -> dict:
    """Paired bootstrap on (baseline MAE - model MAE). Positive means the
    model is better. Paired because both are scored on the same players."""
    rng = np.random.default_rng(seed)
    n = len(model_err)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(float(baseline_err[idx].mean() - model_err[idx].mean()))
    diffs = np.array(diffs)
    return {
        "mean_mae_improvement": float(diffs.mean()),
        "ci_5th": float(np.percentile(diffs, 5)),
        "ci_95th": float(np.percentile(diffs, 95)),
        "fraction_favoring_model": float((diffs > 0).mean()),
    }


def evaluate_season(player_week: pd.DataFrame, scoring, season: int) -> list[dict]:
    pw = player_week[player_week["season"] == season]
    if pw.empty:
        return []
    points = compute_points_for_seasons([season], scoring)
    merged = pw.merge(points, on=["season", "week", "player_id"], how="inner")

    rows = []
    for origin in ORIGINS:
        table = build_ros_table(merged, merged["points"], origin_week=origin)
        if table.empty:
            continue
        # Require a real remaining sample; a player with one game left is
        # mostly measuring one game's noise.
        table = table[(table["ros_games"] >= 3) & (table["games_played"] >= 2)]
        if len(table) < 30:
            continue

        prior = positional_prior(table)
        projections = project(table, prior)
        by_id = {p.player_id: p for p in projections}

        actual, model, ppg_base, prior_base = [], [], [], []
        for row in table.itertuples():
            p = by_id.get(row.player_id)
            if p is None:
                continue
            actual.append(float(row.ros_ppg))
            model.append(p.estimate)
            ppg_base.append(float(row.ppg_to_date))
            prior_base.append(float(prior.get(row.position, row.ppg_to_date)))

        actual = np.array(actual)
        model_err = np.abs(np.array(model) - actual)
        ppg_err = np.abs(np.array(ppg_base) - actual)
        prior_err = np.abs(np.array(prior_base) - actual)

        rows.append({
            "season": season,
            "origin_week": origin,
            "n": len(actual),
            "model_mae": float(model_err.mean()),
            "ppg_baseline_mae": float(ppg_err.mean()),
            "prior_baseline_mae": float(prior_err.mean()),
            "model_spearman": _spearman(np.array(model), actual),
            "ppg_spearman": _spearman(np.array(ppg_base), actual),
            "vs_ppg": _bootstrap_mae_diff(model_err, ppg_err),
            "vs_prior": _bootstrap_mae_diff(model_err, prior_err),
        })
    return rows


def main() -> None:
    player_week = pd.read_parquet(PLAYER_WEEK_PATH)
    scoring = get_default_source().get_league_config().scoring

    all_rows: list[dict] = []
    for season in (2024, 2025):
        all_rows.extend(evaluate_season(player_week, scoring, season))

    if not all_rows:
        raise SystemExit("No evaluable origins; check ingested data.")

    print("ROS MODEL VALIDATION (§2.4)\n")
    header = (f"{'season':<8}{'origin':<8}{'n':<7}{'model':<9}{'stdPPG':<9}"
              f"{'prior':<9}{'rho(m)':<9}{'rho(p)':<9}")
    print(header)
    print("-" * len(header))
    for r in all_rows:
        print(f"{r['season']:<8}{r['origin_week']:<8}{r['n']:<7}"
              f"{r['model_mae']:<9.2f}{r['ppg_baseline_mae']:<9.2f}{r['prior_baseline_mae']:<9.2f}"
              f"{r['model_spearman']:<9.3f}{r['ppg_spearman']:<9.3f}")

    print("\nBOOTSTRAP: MAE improvement over each baseline (positive = model better)")
    print(f"{'season':<8}{'origin':<8}{'vs season-to-date PPG':<34}{'vs positional prior':<34}")
    print("-" * 84)
    for r in all_rows:
        a, b = r["vs_ppg"], r["vs_prior"]
        sa = f"{a['mean_mae_improvement']:+.2f} [{a['ci_5th']:+.2f},{a['ci_95th']:+.2f}] {a['fraction_favoring_model']:.0%}"
        sb = f"{b['mean_mae_improvement']:+.2f} [{b['ci_5th']:+.2f},{b['ci_95th']:+.2f}] {b['fraction_favoring_model']:.0%}"
        print(f"{r['season']:<8}{r['origin_week']:<8}{sa:<34}{sb:<34}")

    # ---- the gate ----
    def beats(rows, key):
        return all(r[key]["ci_5th"] > 0 for r in rows)

    y2024 = [r for r in all_rows if r["season"] == 2024]
    y2025 = [r for r in all_rows if r["season"] == 2025]

    gate_2024 = bool(y2024) and beats(y2024, "vs_ppg") and beats(y2024, "vs_prior")
    signs_2025_match = bool(y2025) and all(
        r["vs_ppg"]["mean_mae_improvement"] > 0 and r["vs_prior"]["mean_mae_improvement"] > 0
        for r in y2025
    )

    print("\n" + "=" * 84)
    print("§2.5 KILL GATE")
    print(f"  Beats BOTH baselines on 2024 at every origin, CI excluding zero : {gate_2024}")
    print(f"  Same sign replicated on 2025 at every origin                    : {signs_2025_match}")
    if gate_2024 and signs_2025_match:
        print("\n  PASS — Phase 4 (trade insights) may ship with ROS projections.")
    else:
        print("\n  FAIL — per §2.5, do NOT ship Phase 4 or Phase 5-full with these")
        print("  projections. Ship trade compare as a pure surfacer (divergence and")
        print("  current-season rates side by side, no ROS number) per §3.7, and")
        print("  document this as a rejected experiment in EVALUATION.md.")
    print("=" * 84)


if __name__ == "__main__":
    main()
