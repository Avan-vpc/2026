# Phase 1.2-1.4 Risk Calibration And Baselines

## Goal

Complete `risk_score_v2` calibration, add core baselines, and stop before `risk_reward / masking / arbitration / full RIGA-LC` unless the calibration gate passes.

## Scope

- keep `official SB3 DQN` as the reference learning baseline
- add `risk_score_v2` without overwriting `risk_score_v1`
- rerun official risk calibration and export `v2` report / figure
- add rule baselines: `Rule-TTC/THW`, `Rule-DRAC`, `Rule-RiskMin`
- add `official PPO` baseline with the same env / reward / observation setting
- export unified baseline comparison across `DQN / Rule / PPO`
- block `DQN + risk_reward` if calibration fails

## Gate

Required before `risk_reward`:

1. `risk_score_v2_max AUC >= 0.65`
2. collision median `risk_score_v2_max >` non-collision median
3. `near_miss_count` should not stay close to `episode_length`

Observed gate result:

- `risk_score_v2_max AUC = 0.543954`
- collision median `risk_score_v2_max = 6.935974`
- non-collision median `risk_score_v2_max = 7.192092`
- `near_miss_count / episode_length` mean `= 0.180830`
- `near_miss_count / episode_length` p90 `= 0.333333`
- gate decision: `fail`

Conclusion:

- `risk_score_v2` is still logging-only
- do not run `scripts/12_train_dqn_risk_reward.py`

## Main Outputs

Calibration:

- `results/official_baseline/metrics/official_dqn_risk_calibration_report_v2.csv`
- `results/official_baseline/figures/official_dqn_risk_calibration_v2.png`

Rule baselines:

- `results/rule_baselines/metrics/rule_baselines_eval.csv`
- `results/rule_baselines/metrics/rule_baselines_summary.csv`
- `results/rule_baselines/metrics/rule_baselines_across_seeds.csv`
- `results/rule_baselines/figures/rule_baselines_comparison.png`

Official PPO:

- `results/official_ppo/metrics/official_ppo_seed_summary.csv`
- `results/official_ppo/metrics/official_ppo_across_seeds.csv`
- `results/official_ppo/metrics/official_ppo_risk_metrics.csv`
- `results/official_ppo/figures/official_ppo_eval_bar.png`
- `results/official_ppo/figures/official_ppo_risk_boxplots.png`
- `results/official_ppo/figures/official_ppo_training_reward.png`
- `results/official_ppo/figures/official_ppo_eval_reward.png`

Unified comparison:

- `results/baseline_comparison/metrics/baseline_comparison_summary.csv`
- `results/baseline_comparison/figures/baseline_safety_efficiency_bar.png`
- `results/baseline_comparison/figures/baseline_safety_efficiency_pareto.png`

## Baseline Comparison

| Method        | Steps  | Collision Rate | Success Rate | Avg Speed | Episode Length | Risk Score v2 Max | Near Miss Count |
|:--------------|------:|---------------:|-------------:|----------:|---------------:|------------------:|----------------:|
| Official DQN  | 100000 | 0.193333       | 0.806667     | 27.836968 | 27.031667      | 6.569388          | 5.578333        |
| Official PPO  | 100000 | 0.211667       | 0.788333     | 22.224045 | 27.323333      | 1.831157          | 0.638333        |
| Rule-TTC/THW  |      0 | 0.000000       | 1.000000     | 21.108610 | 30.000000      | 0.780819          | 0.000000        |
| Rule-DRAC     |      0 | 1.000000       | 0.000000     | 24.391477 | 13.500000      | 2.042734          | 2.000000        |
| Rule-RiskMin  |      0 | 1.000000       | 0.000000     | 23.478757 | 21.000000      | 4.668434          | 3.000000        |

## Interpretation

- `Official DQN` remains the most efficient learning baseline, but it still carries high `risk_score_v2_max` and `near_miss_count`
- `Official PPO` is the second learning baseline; it is slower than DQN but shows much lower `risk_score_v2_max` and `near_miss_count`
- `Rule-TTC/THW` is very conservative and collision-free under the current protocol
- `Rule-DRAC` and `Rule-RiskMin` are currently poor baselines and need further rule redesign if they are kept for the paper
- current `risk_score_v2` does not yet separate collision episodes well enough to support `risk_reward`, `masking`, or `arbitration`

## Script Status

Verified runnable in the current workflow:

- `python scripts/07_analyze_risk_calibration.py`
- `python scripts/09_evaluate_rule_baselines.py`
- `python scripts/10_train_official_ppo.py`
- `python scripts/11_compare_baselines.py`

Blocked by gate:

- `python scripts/12_train_dqn_risk_reward.py`

## Implementation Notes

- `scripts/09_evaluate_rule_baselines.py` now shows a single-line `tqdm` progress bar with ETA
- `scripts/10_train_official_ppo.py` already reuses the callback progress bar with `eta_min`
- `src/agents/official_sb3_ppo.py` was fixed to save under repository-absolute `results/official_ppo/` so it works under the Windows `conda run` + UNC workflow

## Submission Boundary

Recommended code/docs:

- `README.md`
- `.gitignore`
- `md/task_phase1_risk_metrics_fix.md`
- `md/task_phase12_14_baselines_and_calibration.md`
- `scripts/07_analyze_risk_calibration.py`
- `scripts/09_evaluate_rule_baselines.py`
- `scripts/10_train_official_ppo.py`
- `scripts/11_compare_baselines.py`
- `src/risk/risk_score.py`
- `src/envs/highway_custom_env.py`
- `src/agents/official_sb3_dqn.py`
- `src/agents/official_sb3_ppo.py`
- `src/baselines/`

Recommended results:

- `results/official_baseline/metrics/official_100k_seed_summary.csv`
- `results/official_baseline/metrics/official_100k_across_seeds.csv`
- `results/official_baseline/metrics/official_dqn_risk_metrics_fixed.csv`
- `results/official_baseline/metrics/official_dqn_risk_calibration_report.csv`
- `results/official_baseline/metrics/official_dqn_risk_calibration_report_v2.csv`
- `results/official_baseline/figures/official_dqn_risk_calibration.png`
- `results/official_baseline/figures/official_dqn_risk_calibration_v2.png`
- `results/rule_baselines/metrics/`
- `results/rule_baselines/figures/`
- `results/official_ppo/metrics/`
- `results/official_ppo/figures/`
- `results/baseline_comparison/metrics/`
- `results/baseline_comparison/figures/`

Keep local-only:

- `results/official_baseline/models/`
- `results/official_baseline/metrics/*_best.zip`
- `results/official_ppo/models/*.zip`
- `results/official_ppo/models/*_best.zip`
- `results/legacy_unreliable/root_results_snapshot/`
- any large local archive or checkpoint not required by the paper figures / tables
