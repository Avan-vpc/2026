import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.official_sb3_dqn import load_yaml, train_official_baseline


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "full", "three_seeds"], default="smoke")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()

    train_cfg = load_yaml(ROOT / "configs" / "train_dqn_official.yaml")
    runtime = train_cfg["runtime"]
    deterministic_eval = bool(runtime.get("deterministic_eval", True))
    if args.stage == "smoke":
        timesteps = int(runtime.get("smoke_timesteps", 20000))
        eval_freq = int(runtime.get("smoke_eval_freq", 1000))
        seeds = args.seeds or [0]
    elif args.stage == "full":
        timesteps = int(runtime.get("full_timesteps", 100000))
        eval_freq = int(runtime.get("full_eval_freq", 5000))
        seeds = args.seeds or [0]
    else:
        timesteps = int(runtime.get("full_timesteps", 100000))
        eval_freq = int(runtime.get("full_eval_freq", 5000))
        seeds = args.seeds or list(runtime.get("seeds", [0, 1, 2]))

    for seed in seeds:
        result = train_official_baseline(
            env_config_path=ROOT / "configs" / "highway_fast_official.yaml",
            train_config_path=ROOT / "configs" / "train_dqn_official.yaml",
            seed=int(seed),
            total_timesteps=int(timesteps),
            eval_freq=int(eval_freq),
            checkpoint_eval_episodes=int(runtime.get("checkpoint_eval_episodes", 20)),
            deterministic_eval=deterministic_eval,
            tag="official_sb3_dqn_highway_fast",
        )
        print(f"Saved model: {result['model_path']}")
