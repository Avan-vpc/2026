from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

from src.analysis.plot_training import plot_training_curves
from src.envs.highway_custom_env import build_env


class PeriodicEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int, csv_path: Path):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.csv_path = Path(csv_path)
        self.records = []
        self.best_mean = -np.inf
        self.best_model_path = self.csv_path.with_name(self.csv_path.stem.replace("_training_curve", "_best") + ".zip")

    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            rewards, _ = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                return_episode_rewards=True,
                deterministic=True,
                warn=False,
            )
            mean_reward = float(np.mean(rewards)) if rewards else 0.0
            std_reward = float(np.std(rewards)) if rewards else 0.0
            self.records.append({"step": int(self.num_timesteps), "mean_reward": mean_reward, "std_reward": std_reward})
            if mean_reward > self.best_mean:
                self.best_mean = mean_reward
                self.model.save(self.best_model_path)
        return True

    def _on_training_end(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.records).to_csv(self.csv_path, index=False)


def load_yaml(path: Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def train_experiment(
    env_config_path: Path,
    risk_config_path: Path,
    train_config_path: Path,
    mode: str,
    seed: int,
    total_timesteps: Optional[int] = None,
) -> Path:
    env_name = Path(env_config_path).stem.replace("_default", "")
    train_cfg = load_yaml(Path(train_config_path))
    runtime_cfg = train_cfg.get("runtime", {})
    algo_cfg = train_cfg.get("algo", {})
    total_timesteps = int(total_timesteps or runtime_cfg.get("default_timesteps", 50000))
    run_name = f"{mode}_{env_name}_seed{seed}"
    results_root = Path("results")
    model_path = results_root / "models" / f"{run_name}.zip"
    curve_csv = results_root / "metrics" / f"{run_name}_training_curve.csv"
    curve_png = results_root / "figures" / f"{run_name}_training_curve.png"
    train_env = Monitor(build_env(Path(env_config_path), Path(risk_config_path), mode=mode))
    eval_env = Monitor(build_env(Path(env_config_path), Path(risk_config_path), mode=mode))
    callback = PeriodicEvalCallback(
        eval_env=eval_env,
        eval_freq=int(runtime_cfg.get("eval_freq", 5000)),
        n_eval_episodes=int(runtime_cfg.get("n_eval_episodes", 3)),
        csv_path=curve_csv,
    )
    model = DQN(
        policy=str(algo_cfg.get("policy", "MlpPolicy")),
        env=train_env,
        learning_rate=float(algo_cfg.get("learning_rate", 1e-4)),
        buffer_size=int(algo_cfg.get("buffer_size", 50000)),
        learning_starts=int(algo_cfg.get("learning_starts", 1000)),
        batch_size=int(algo_cfg.get("batch_size", 64)),
        gamma=float(algo_cfg.get("gamma", 0.99)),
        train_freq=int(algo_cfg.get("train_freq", 4)),
        target_update_interval=int(algo_cfg.get("target_update_interval", 500)),
        exploration_fraction=float(algo_cfg.get("exploration_fraction", 0.2)),
        exploration_initial_eps=float(algo_cfg.get("exploration_initial_eps", 1.0)),
        exploration_final_eps=float(algo_cfg.get("exploration_final_eps", 0.05)),
        seed=int(seed),
        verbose=1,
    )
    model.learn(total_timesteps=total_timesteps, progress_bar=False, callback=callback)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    plot_training_curves([curve_csv], curve_png)
    train_env.close()
    eval_env.close()
    return model_path
