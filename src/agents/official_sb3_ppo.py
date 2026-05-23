from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
from stable_baselines3 import PPO

from src.agents.official_sb3_dqn import (
    OfficialTrainCallback,
    _evaluate_policy_core,
    aggregate_training_curves,
    load_eval_metrics,
    load_yaml,
    make_env,
    summarize_across_seeds,
    summarize_episode_metrics,
    summarize_probability_metrics,
)


def build_model(env, train_cfg: Dict, seed: int, tensorboard_log: Optional[str] = None) -> PPO:
    algo = train_cfg["algo"]
    return PPO(
        policy=str(algo.get("policy", "MlpPolicy")),
        env=env,
        learning_rate=float(algo.get("learning_rate", 3e-4)),
        n_steps=int(algo.get("n_steps", 1024)),
        batch_size=int(algo.get("batch_size", 64)),
        n_epochs=int(algo.get("n_epochs", 10)),
        gamma=float(algo.get("gamma", 0.8)),
        gae_lambda=float(algo.get("gae_lambda", 0.95)),
        clip_range=float(algo.get("clip_range", 0.2)),
        ent_coef=float(algo.get("ent_coef", 0.0)),
        seed=int(seed),
        verbose=0,
        tensorboard_log=tensorboard_log,
    )


def train_official_ppo(
    env_config_path: Path,
    train_config_path: Path,
    seed: int,
    total_timesteps: int,
    eval_freq: int,
    checkpoint_eval_episodes: int,
    deterministic_eval: bool,
    tag: str,
) -> Dict[str, Path]:
    train_cfg = load_yaml(Path(train_config_path))
    run_name = f"{tag}_steps{int(total_timesteps)}_seed{int(seed)}"
    repo_root = Path(__file__).resolve().parents[2]
    results_root = repo_root / "results" / "official_ppo"
    model_path = results_root / "models" / f"{run_name}.zip"
    curve_csv = results_root / "metrics" / f"{run_name}_training_curve.csv"
    train_env = make_env(env_config_path, seed=seed)
    eval_env = make_env(env_config_path, seed=seed)
    callback = OfficialTrainCallback(
        eval_env=eval_env,
        eval_freq=int(eval_freq),
        n_eval_episodes=int(checkpoint_eval_episodes),
        csv_path=curve_csv,
        total_timesteps=int(total_timesteps),
        run_name=run_name,
        deterministic_eval=bool(deterministic_eval),
    )
    model = build_model(train_env, train_cfg, seed=seed, tensorboard_log=str(results_root / "tensorboard"))
    model.learn(total_timesteps=int(total_timesteps), progress_bar=False, callback=callback)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    train_env.close()
    eval_env.close()
    return {"model_path": model_path, "curve_csv": curve_csv}


def evaluate_model(
    model_path: Path,
    env_config_path: Path,
    risk_config_path: Path,
    seed: int,
    episodes: int,
    deterministic_eval: bool,
    gif_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    model = PPO.load(Path(model_path))
    return _evaluate_policy_core(
        policy_fn=lambda obs: int(model.predict(obs, deterministic=deterministic_eval)[0]),
        run_name=Path(model_path).stem,
        env_config_path=env_config_path,
        risk_config_path=risk_config_path,
        seed=seed,
        episodes=episodes,
        gif_path=gif_path,
    )


__all__ = [
    "aggregate_training_curves",
    "evaluate_model",
    "load_eval_metrics",
    "load_yaml",
    "summarize_across_seeds",
    "summarize_episode_metrics",
    "summarize_probability_metrics",
    "train_official_ppo",
]
