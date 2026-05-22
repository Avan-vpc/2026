# Official Baseline Reset

## Goal

Reproduce the official `highway-env` SB3 DQN quickstart baseline on `highway-fast-v0`, freeze the smoke baseline, and then extend it to a stable `100k / 3 seeds` benchmark before adding any custom risk intervention.

## Scope

- `highway-fast-v0` only
- official SB3 DQN only
- official hyperparameters only
- no custom reward, no risk reward, no masking, no arbitration in training
- risk metrics logging is evaluation-only
- no `merge-v0`, `highD`, `CARLA`, `MetaDrive`, or `flow matching`

## Required Outputs

- training reward curve
- evaluation mean reward curve
- collision rate
- average speed
- episode length
- lane change count
- success rate
- saved model
- rollout GIF
- risk metrics CSV
- risk boxplots and case time-series figure

## Execution Order

1. freeze smoke baseline `20k / seed0`
2. run `100k / seed0`
3. run `3 seeds x 100k`
4. add risk logging on official baseline evaluation
5. only then move to risk-aware reward

## Current Status

- official config prepared
- smoke baseline validated on `highway-fast-v0`
- smoke baseline frozen to stable result paths
- official baseline `100k / 3 seeds` training completed
- risk logging exported for the official `100k / 3 seeds` benchmark
