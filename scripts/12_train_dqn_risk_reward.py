import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.train_dqn import train_experiment
from src.agents.official_sb3_dqn import load_yaml


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()

    train_cfg = load_yaml(ROOT / "configs" / "train_dqn_risk_reward.yaml")
    runtime = train_cfg["runtime"]
    seeds = args.seeds or list(runtime.get("seeds", [0, 1, 2]))
    total_timesteps = int(runtime.get("total_timesteps", 100000))

    for seed in seeds:
        model_path = train_experiment(
            env_config_path=ROOT / "configs" / "highway_fast_official.yaml",
            risk_config_path=ROOT / "configs" / "risk_params.yaml",
            train_config_path=ROOT / "configs" / "train_dqn_risk_reward.yaml",
            mode="risk_reward",
            seed=int(seed),
            total_timesteps=total_timesteps,
        )
        print(f"Saved model: {model_path}")
