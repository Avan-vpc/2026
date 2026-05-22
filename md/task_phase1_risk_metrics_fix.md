# Phase 1 Risk Metrics Fix

## Goal

Pause `full / risk_reward / masked` expansion and make the `official baseline` evaluation pipeline trustworthy.

## Scope

- clean repository documentation and ignore rules
- archive legacy unreliable outputs without deleting them
- fix official eval summary aggregation
- rewrite risk geometry in `src/risk/metrics.py`
- check `P_yield / P_block` distribution and export statistics
- rerun official evaluation and figures only

## Acceptance Criteria

1. `20k smoke` and `100k formal` summaries are separated
2. `official_100k_across_seeds.csv` is generated from `100k` runs only
3. non-collision episodes are no longer flagged as near-miss at nearly every step
4. `p_yield / p_block` show non-degenerate distributions
5. reward / collision / avg_speed stay unchanged up to logging-only noise

## Work Log

- updated `README.md` with install, run commands, trust level, and result layout
- added `.gitignore` for `__pycache__`, logs, and tensorboard artifacts
- archived root-level legacy outputs to `results/legacy_unreliable/root_results_snapshot/`
- rewrote `src/risk/metrics.py` with same-lane and target-lane conflict filtering
- updated `target_gap.py` and `game_risk.py` to use the new target-gap state and less-saturated yield/block scores
- updated `official_sb3_dqn.py`, `05_evaluate_all.py`, `06_export_figures.py`, and `plot_results.py`
- reran official evaluation and exported new summaries and figures

## Current Outputs

Primary new files under `results/official_baseline/metrics/`:

- `official_smoke_20k_summary.csv`
- `official_100k_seed_summary.csv`
- `official_100k_across_seeds.csv`
- `official_dqn_risk_metrics_fixed.csv`
- `official_dqn_probability_stats.csv`
- `official_dqn_probability_histogram.csv`

Primary new figure under `results/official_baseline/figures/`:

- `official_dqn_probability_histogram.png`

## Observations

- `official_100k_across_seeds.csv` preserves the baseline reward / collision / speed statistics
- `near_miss_count` no longer collapses to episode length for most episodes
- `p_yield` now spans a usable range instead of saturating near `1.0`
- `risk_score_max` still deserves another review because collision episodes are not yet cleanly separated from non-collision episodes
