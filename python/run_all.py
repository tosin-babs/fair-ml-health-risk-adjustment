"""
Run the whole analysis, in order.

    ../.venv/bin/python python/run_all.py

Seeded from config.SEED. The benchmark step is the slow one: nine models,
five survey folds, three repeats, and the nested feature-set comparison.
Set P6_BOOT (bootstrap replicates, default 200) lower for a quick pass.

Each step runs in its own Python process. LightGBM, scikit-learn and PyTorch
each ship an OpenMP runtime; on macOS, loading them in one long-lived process
in the wrong order deadlocks or aborts. Separate processes guarantee that no
step inherits another step's threading state.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import config  # noqa: E402

STEPS = [
    ("Prospective dataset from the MEPS panels", "build_data"),
    ("Sample description", "describe"),
    ("RQ1: accuracy with survey-aware cross-validation", "benchmark"),
    ("RQ2: group compensation and the fairness frontier", "fair_frontier"),
    ("RQ3: drivers and their stability", "explain"),
    ("RQ4: coding sensitivity", "coding"),
    ("Robustness", "robustness"),
    ("Figures", "exhibits"),
    ("Calculator payload", "export_tool_data"),
]


def main():
    started, timings = time.time(), []
    for title, module in STEPS:
        script = HERE / f"{module}.py"
        if not script.exists():
            print(f"\n[skipping {module}: not written yet]")
            continue
        print("\n" + "=" * 78 + f"\n{title}\n" + "=" * 78, flush=True)
        t0 = time.time()
        r = subprocess.run([sys.executable, "-u", str(script)], cwd=HERE)
        if r.returncode != 0:
            sys.exit(f"step {module} failed with exit code {r.returncode}")
        timings.append({"step": title, "seconds": round(time.time() - t0, 1)})
        print(f"[{title}: {timings[-1]['seconds']:.0f}s]", flush=True)

    versions = {"python": platform.python_version()}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "lightgbm", "torch", "shap"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = "unavailable"
    params = {k: str(v) for k, v in vars(config).items() if k.isupper()}
    (config.ROOT / "output" / "params_used.json").write_text(json.dumps(
        {"params": params, "versions": versions, "timings": timings}, indent=2))
    print(f"\nDone in {time.time() - started:.0f}s.")


if __name__ == "__main__":
    main()
