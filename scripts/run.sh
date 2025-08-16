#!/usr/bin/env bash
set -euo pipefail

# Simple runner for a VJEPA2 test run and brain metric rendering.
# Usage: scripts/run.sh [RESULTS_DIR]
# RESULTS_DIR defaults to save_dir/results/algonauts-2025/test

RESULTS_DIR="${1:-save_dir/results/algonauts-2025/test}"

# Activate local venv if present
if [[ -d .venv ]]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Run the small test configuration (uses VJEPA2 by default in defaults.py)
python -m algonauts2025.grids.test_run

# Render basic brain metrics (pearson distribution and top parcels)
PYTHON_BIN=${PYTHON_BIN:-python}
"${PYTHON_BIN}" - <<'PY'
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path(os.environ.get('RESULTS_DIR', 'save_dir/results/algonauts-2025/test'))
pearson_path = results_dir / 'pearson.npy'
if not pearson_path.exists():
    raise SystemExit(f"pearson.npy not found at {pearson_path}. Did the run complete?")

pearson = np.load(pearson_path)
# Basic stats
mean_p = float(np.nanmean(pearson))
median_p = float(np.nanmedian(pearson))

# Save histogram
plt.figure(figsize=(6,4))
plt.hist(pearson, bins=50, color='#3b82f6', edgecolor='white')
plt.title(f'Val Pearson per parcel\nmean={mean_p:.3f}, median={median_p:.3f}')
plt.xlabel('Pearson r')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig(results_dir / 'pearson_hist.png', dpi=150)
plt.close()

# Save top-K parcels
topk = 50
idx = np.argsort(-pearson)[:topk]
with open(results_dir / 'pearson_topk.txt', 'w') as f:
    for rank, (i, r) in enumerate(zip(idx, pearson[idx]), start=1):
        f.write(f"{rank:02d}\tparcel={int(i)}\tr={float(r):.4f}\n")

print(f"Saved histogram and top-K list to {results_dir}")
PY
