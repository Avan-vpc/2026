# Decision 2026 Highway Env

This repository currently focuses on one trustworthy baseline:

- environment: `highway-fast-v0`
- algorithm: official-style `Stable-Baselines3 DQN`
- training: no custom reward, no risk reward, no masking, no arbitration
- evaluation: risk logging is enabled, but only after geometry-aware filtering

The current goal is to make the official baseline and its risk logging reliable before extending to `risk_reward`, `masked`, or `full` methods.

## Project Goal

Build a reproducible decision-making benchmark around `highway-env`, then add risk-aware modules on top of a stable official-style DQN baseline.

## Environment Setup

Use the `pytorch` conda environment:

```bash
conda activate pytorch
pip install -r requirements.txt
```

## Official Baseline Configuration

The official baseline follows the `highway-env` quickstart-style SB3 DQN setup:

- `policy = MlpPolicy`
- `net_arch = [256, 256]`
- `learning_rate = 5e-4`
- `buffer_size = 15000`
- `learning_starts = 200`
- `batch_size = 32`
- `gamma = 0.8`
- `train_freq = 1`
- `gradient_steps = 1`
- `target_update_interval = 50`

Config files:

- `configs/highway_fast_official.yaml`
- `configs/train_dqn_official.yaml`
- `configs/risk_params.yaml`

## Run Commands

Smoke test:

```bash
python scripts/00_smoke_test_env.py
python scripts/01_train_baseline_dqn.py --stage smoke
```

Formal baseline:

```bash
python scripts/01_train_baseline_dqn.py --stage full
python scripts/01_train_baseline_dqn.py --stage three_seeds
```

Evaluation and figure export:

```bash
python scripts/05_evaluate_all.py
python scripts/06_export_figures.py
```

Optional archive step for the smoke snapshot:

```bash
python scripts/08_freeze_official_baseline.py
```

## Results Layout

Primary outputs:

- `results/official_baseline/models/`
- `results/official_baseline/metrics/`
- `results/official_baseline/figures/`
- `results/official_baseline/videos/`

Archived unreliable outputs:

- `results/legacy_unreliable/`

## Current Trust Level

Currently acceptable as baseline statistics:

- reward
- collision rate
- average speed
- episode length
- lane change count

Currently under active repair and should not be cited as final results until regenerated:

- TTC / THW / DRAC
- `near_miss_count`
- `risk_exposure`
- `risk_score_max`
- `P_yield / P_block`

## Summary Files

The official evaluation pipeline separates:

- `official_smoke_20k_summary.csv`
- `official_100k_seed_summary.csv`
- `official_100k_across_seeds.csv`

This prevents `20k smoke` runs from being mixed into `100k` formal summaries.

## Legacy Outputs

Older `baseline / risk_reward / masked / full` results are preserved for traceability, but they are archived under `results/legacy_unreliable/` and must not be used as paper figures or tables.
