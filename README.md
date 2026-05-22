# Official Highway-Env Baseline

This repository now prioritizes reproducing the official `highway-env` SB3 DQN baseline before adding any custom risk module.

## Current Priority

- environment: `highway-fast-v0`
- algorithm: official `Stable-Baselines3 DQN`
- no custom reward
- no risk module in training
- no masking
- no arbitration
- no `merge-v0`, `highD`, `CARLA`, `MetaDrive`, or `flow matching`

## Official DQN Parameters

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

## Baseline Reproduction Commands

```bash
python scripts/00_smoke_test_env.py
python scripts/01_train_baseline_dqn.py --stage smoke
python scripts/05_evaluate_all.py
python scripts/06_export_figures.py
python scripts/08_freeze_official_baseline.py
```

## Extended Official Baseline

```bash
python scripts/01_train_baseline_dqn.py --stage full
python scripts/05_evaluate_all.py
python scripts/06_export_figures.py

python scripts/01_train_baseline_dqn.py --stage three_seeds
python scripts/05_evaluate_all.py
python scripts/06_export_figures.py
```

## Evaluation Setup

- checkpoint evaluation every `5000` steps for the `100k` run
- `20` episodes for each checkpoint evaluation
- `200` episodes for final evaluation
- deterministic evaluation enabled

## Outputs

- `results/official_baseline/models/`
- `results/official_baseline/metrics/`
- `results/official_baseline/figures/`
- `results/official_baseline/videos/`
- frozen smoke baseline copies in `results/models`, `results/metrics`, `results/figures`, `results/videos`
