# Phase 1.1 Risk Metrics Fix And Calibration

## Goal

Pause `full / risk_reward / masked` expansion and make the `official baseline` evaluation pipeline trustworthy and calibration-ready.

## Scope

- clean repository documentation and ignore rules
- keep legacy unreliable outputs out of the current git submission boundary
- fix official eval summary aggregation
- rewrite risk geometry in `src/risk/metrics.py`
- check `P_yield / P_block` distribution and export statistics
- add a standalone `risk calibration` report for the fixed logging output
- rerun analysis and figures only

## Acceptance Criteria

1. `20k smoke` and `100k formal` summaries are separated
2. `official_100k_across_seeds.csv` is generated from `100k` runs only
3. non-collision episodes are no longer flagged as near-miss at nearly every step
4. `p_yield / p_block` show non-degenerate distributions
5. reward / collision / avg_speed stay unchanged up to logging-only noise
6. collision vs non-collision risk distributions are summarized explicitly
7. AUC-based calibration guidance is exported without changing training logic

## Work Log

- updated `README.md` with install, run commands, trust level, and result layout
- added `.gitignore` rules for `__pycache__`, logs, tensorboard, archived legacy snapshots, and large local model artifacts
- added a root-level `README.md` so GitHub main exposes the project status directly
- rewrote `src/risk/metrics.py` with same-lane and target-lane conflict filtering
- updated `target_gap.py` and `game_risk.py` to use the new target-gap state and less-saturated yield/block scores
- updated `official_sb3_dqn.py`, `05_evaluate_all.py`, `06_export_figures.py`, and `plot_results.py`
- added `scripts/07_analyze_risk_calibration.py` to summarize percentiles and AUCs from `official_dqn_risk_metrics_fixed.csv`
- exported a dedicated calibration report and figure for Phase 1.1 review

## Current Outputs

Primary new files under `results/official_baseline/metrics/`:

- `official_smoke_20k_summary.csv`
- `official_100k_seed_summary.csv`
- `official_100k_across_seeds.csv`
- `official_dqn_risk_metrics_fixed.csv`
- `official_dqn_probability_stats.csv`
- `official_dqn_probability_histogram.csv`
- `official_dqn_risk_calibration_report.csv`

Primary figures under `results/official_baseline/figures/`:

- `official_dqn_probability_histogram.png`
- `official_dqn_risk_boxplots.png`
- `official_dqn_risk_case_timeseries.png`
- `official_dqn_risk_calibration.png`

## Observations

- `official_100k_across_seeds.csv` preserves the baseline reward / collision / speed statistics
- `near_miss_count` no longer collapses to episode length for most episodes
- `p_yield` now spans a usable range instead of saturating near `1.0`
- `risk_score_max` still deserves another review because collision episodes are not yet cleanly separated from non-collision episodes
- current interpretation is:
  - official DQN baseline: trustworthy
  - risk logging geometry: trustworthy enough for analysis
  - `P_yield / P_block`: usable as logging features
  - `risk_score`: not yet calibrated for reward, masking, or arbitration

## Submission Boundary

Recommended code-and-doc commit:

- `README.md`
- `.gitignore`
- `md/task_phase1_risk_metrics_fix.md`
- `scripts/07_analyze_risk_calibration.py`
- existing risk logging source changes already prepared for Phase 1.1

Recommended results commit:

- `results/official_baseline/metrics/official_100k_seed_summary.csv`
- `results/official_baseline/metrics/official_100k_across_seeds.csv`
- `results/official_baseline/metrics/official_dqn_risk_metrics_fixed.csv`
- `results/official_baseline/metrics/official_dqn_probability_stats.csv`
- `results/official_baseline/metrics/official_dqn_probability_histogram.csv`
- `results/official_baseline/metrics/official_dqn_risk_calibration_report.csv`
- `results/official_baseline/figures/official_dqn_probability_histogram.png`
- `results/official_baseline/figures/official_dqn_risk_boxplots.png`
- `results/official_baseline/figures/official_dqn_risk_case_timeseries.png`
- `results/official_baseline/figures/official_dqn_risk_calibration.png`

Keep local-only:

- `results/official_baseline/models/`
- `results/official_baseline/metrics/*_best.zip`
- `results/legacy_unreliable/root_results_snapshot/`
