import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.train_dqn import train_experiment, load_yaml


if __name__ == "__main__":
    train_cfg = load_yaml(ROOT / "configs" / "train_dqn.yaml")
    total_timesteps = int(train_cfg["runtime"].get("smoke_timesteps", 20000))
    for env_config in ["highway_default.yaml", "merge_default.yaml"]:
        model_path = train_experiment(
            env_config_path=ROOT / "configs" / env_config,
            risk_config_path=ROOT / "configs" / "risk_params.yaml",
            train_config_path=ROOT / "configs" / "train_dqn.yaml",
            mode="masked",
            seed=0,
            total_timesteps=total_timesteps,
        )
        print(f"Saved model: {model_path}")
