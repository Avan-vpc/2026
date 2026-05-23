# Decision 2026 Highway Env

This repository currently treats the official-style `Stable-Baselines3 DQN` on `highway-fast-v0` as the only trustworthy baseline.

## Current Status

- baseline training setup: stable and reproducible
- `100k x 3 seeds` official evaluation: available
- risk metric logging: geometry-aware and usable for analysis
- `risk_score`: not yet calibrated for reward or masking

The current milestone is:

> Phase 1.1: Risk Metric Calibration on Official DQN Baseline

## Quick Start

Use the `pytorch` conda environment:

```bash
conda activate pytorch
pip install -r requirements.txt
```

Run the official baseline:

```bash
python scripts/01_train_baseline_dqn.py --stage smoke
python scripts/01_train_baseline_dqn.py --stage three_seeds
python scripts/05_evaluate_all.py
python scripts/06_export_figures.py
python scripts/07_analyze_risk_calibration.py
```

## Result Policy

- commit official metrics and figures under `results/official_baseline/`
- keep archived unreliable snapshots out of git
- keep large `.zip` model artifacts local unless explicitly needed

## More Detail

Project notes and task logs are kept under `md/`, including:

- `md/README.md`
- `md/task_phase1_risk_metrics_fix.md`
