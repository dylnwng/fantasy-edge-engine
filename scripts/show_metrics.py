"""Print the current, real model numbers.

Exists because performance figures were duplicated by hand across README,
CLAUDE.md, SETUP.md and the dashboard footer, and promptly drifted --
CLAUDE.md spent a while quoting a MAE that a retrain had already
superseded. Docs go stale; the saved metrics files don't.

This reads `models/training_metrics*.json`, which `train` and `train_qb`
write on every run, so whatever it prints is what the shipped model
actually scored.

Usage:
    python scripts/show_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

MODELS = [
    ("Main model (WR / RB / TE)", ROOT / "models" / "training_metrics.json"),
    ("QB model", ROOT / "models" / "training_metrics_qb.json"),
]


def _pct(x: float | None) -> str:
    return f"{x * 100:.1f}%" if x is not None else "n/a"


def main() -> None:
    found = False
    for label, path in MODELS:
        if not path.exists():
            print(f"{label}: not trained yet ({path.name} missing)")
            continue
        found = True
        m = json.load(open(path))
        improvement = (1 - m["model_mae"] / m["baseline_mae"]) * 100

        print(f"\n{label}")
        print(f"  trained on      {m['train_seasons']}")
        print(f"  validated on    {m['validation_season']}  (n={m['n_validation_rows']})")
        print(f"  MAE             {m['model_mae']:.2f}  vs baseline {m['baseline_mae']:.2f}"
              f"   ({improvement:.1f}% better)")
        print(f"  flagged         {m['n_flagged']} players, hit rate {_pct(m.get('hit_rate'))}")

        gate = m.get("vs_baseline")
        if gate:
            verdict = "clears zero" if gate["ci_5th"] > 0 else "CROSSES ZERO — no longer justified"
            print(f"  bootstrap       {gate['mean_mae_gain']:+.2f} MAE "
                  f"[{gate['ci_5th']:+.2f}, {gate['ci_95th']:+.2f}] — {verdict}")

    if not found:
        print("\nNo trained models found. Run:")
        print("  python -m edge_engine.model.train")
        print("  python -m edge_engine.model.train_qb")
        sys.exit(1)

    print("\nThese are the numbers to quote. If a doc disagrees, the doc is wrong.")


if __name__ == "__main__":
    main()
