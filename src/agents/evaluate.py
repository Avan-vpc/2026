from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml
from stable_baselines3 import DQN

from src.analysis.metrics_logger import EPISODE_COLUMNS, write_rows
from src.envs.highway_custom_env import build_env


def load_yaml(path: Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def evaluate_model(
    model_path: Path,
    env_config_path: Path,
    risk_config_path: Path,
    mode: str,
    seed: int,
    episodes: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    model = DQN.load(Path(model_path))
    env = build_env(Path(env_config_path), Path(risk_config_path), mode=mode)
    env_name = Path(env_config_path).stem.replace("_default", "")
    method = f"{mode}_{env_name}"
    episode_rows: List[Dict] = []
    case_rows: List[Dict] = []
    for episode_idx in range(int(episodes)):
        obs, info = env.reset(seed=int(seed) + episode_idx)
        terminated = False
        truncated = False
        step_idx = 0
        source_counter = Counter()
        while not (terminated or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            source_counter[str(info.get("chosen_source", "rl"))] += 1
            if episode_idx == 0:
                case_rows.append({
                    "method": method,
                    "episode": episode_idx,
                    "step": step_idx,
                    "reward": float(reward),
                    "ttc_min": float(info.get("ttc_min", 0.0)),
                    "thw_min": float(info.get("thw_min", 0.0)),
                    "drac_max": float(info.get("drac_max", 0.0)),
                    "risk_score": float(info.get("risk_score", 0.0)),
                    "raw_action": int(info.get("raw_action", int(action))),
                    "final_action": int(info.get("final_action", int(action))),
                    "chosen_source": str(info.get("chosen_source", "rl")),
                })
            step_idx += 1
        metrics = dict(info.get("episode_metrics", {}))
        row = {column: None for column in EPISODE_COLUMNS}
        row.update({
            "method": method,
            "seed": int(seed),
            "episode": int(episode_idx),
        })
        row.update(metrics)
        if source_counter:
            row["chosen_source"] = source_counter.most_common(1)[0][0]
        episode_rows.append(row)
    env.close()
    return pd.DataFrame(episode_rows, columns=EPISODE_COLUMNS), pd.DataFrame(case_rows)


def evaluate_and_save(
    model_path: Path,
    env_config_path: Path,
    risk_config_path: Path,
    mode: str,
    seed: int,
    episodes: int,
) -> Tuple[Path, Path]:
    env_name = Path(env_config_path).stem.replace("_default", "")
    run_name = f"{mode}_{env_name}_seed{seed}"
    episode_df, case_df = evaluate_model(model_path, env_config_path, risk_config_path, mode, seed, episodes)
    metrics_path = Path("results") / "metrics" / f"{run_name}_eval.csv"
    case_path = Path("results") / "metrics" / f"{run_name}_case.csv"
    write_rows(metrics_path, episode_df.to_dict(orient="records"))
    write_rows(case_path, case_df.to_dict(orient="records"))
    return metrics_path, case_path
