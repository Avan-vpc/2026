import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import gymnasium as gym
import highway_env  # noqa: F401
import imageio.v2 as imageio
import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from tqdm.auto import tqdm

from src.risk.game_risk import compute_game_risk
from src.risk.metrics import _lane_id, _safe_position, _safe_speed, collect_step_metrics, is_near_miss
from src.risk.risk_score import compute_risk_score
from src.risk.target_gap import identify_target_gap


RUN_NAME_PATTERN = re.compile(r"^(?P<method>.+?)_steps(?P<steps>\d+)_seed(?P<seed>\d+)$")
EPISODE_METRICS = [
    "reward",
    "collision",
    "success",
    "avg_speed",
    "episode_length",
    "lane_change_count",
    "ttc_min",
    "thw_min",
    "drac_max",
    "risk_exposure",
    "risk_score_max",
    "near_miss_count",
    "target_gap_size",
    "target_rear_distance",
    "target_rear_relative_speed",
    "p_yield",
    "p_block",
]


def load_yaml(path: Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def make_env(env_config_path: Path, seed: Optional[int] = None, render_mode: Optional[str] = None):
    env_cfg = load_yaml(Path(env_config_path))
    env = gym.make(
        str(env_cfg["env"]["id"]),
        config=env_cfg["env"].get("config", {}),
        render_mode=render_mode,
    )
    if seed is not None:
        env.reset(seed=int(seed))
        env.action_space.seed(int(seed))
    return env


def _safe_collision(env) -> int:
    vehicle = getattr(env.unwrapped, "vehicle", None)
    if vehicle is None:
        return 0
    crashed = getattr(vehicle, "crashed", False)
    if callable(crashed):
        crashed = crashed()
    return int(bool(crashed))


def parse_run_metadata(name: str) -> Dict[str, object]:
    stem = Path(name).stem
    match = RUN_NAME_PATTERN.match(stem)
    if match is None:
        return {
            "run_name": stem,
            "method": stem,
            "steps": -1,
            "seed": -1,
            "stage": "unknown",
            "method_label": stem,
        }
    method = str(match.group("method"))
    steps = int(match.group("steps"))
    seed = int(match.group("seed"))
    stage = "smoke" if steps <= 20000 else "formal"
    return {
        "run_name": stem,
        "method": method,
        "steps": steps,
        "seed": seed,
        "stage": stage,
        "method_label": f"{method} ({steps // 1000}k)",
    }


def _risk_step(env, risk_params: Dict) -> Dict[str, float]:
    metrics = collect_step_metrics(env)
    gap = identify_target_gap(env)
    game = compute_game_risk(env, risk_params)
    risk_score = compute_risk_score(metrics, game, risk_params)
    rear = gap.get("target_rear_vehicle")
    ego = getattr(env.unwrapped, "vehicle", None)
    rear_distance = 0.0
    if ego is not None and rear is not None:
        rear_distance = float(abs(_safe_position(ego)[0] - _safe_position(rear)[0]))
    step = {
        **metrics,
        "risk_score": float(risk_score),
        "target_gap_size": float(metrics.get("target_gap_size", gap.get("gap_size", 0.0))),
        "target_rear_distance": float(metrics.get("target_rear_distance", rear_distance)),
        "target_rear_relative_speed": float(metrics.get("target_rear_relative_speed", gap.get("rear_relative_speed", 0.0))),
        "p_yield": float(game.get("P_yield", 0.0)),
        "p_block": float(game.get("P_block", 0.0)),
        "near_miss_flag": float(is_near_miss(metrics, risk_params.get("thresholds", {}))),
    }
    return step


class OfficialTrainCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq: int, n_eval_episodes: int, csv_path: Path, total_timesteps: int, run_name: str, deterministic_eval: bool):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.csv_path = Path(csv_path)
        self.total_timesteps = int(total_timesteps)
        self.run_name = run_name
        self.deterministic_eval = bool(deterministic_eval)
        self.records: List[Dict] = []
        self.best_mean = -np.inf
        self.best_model_path = self.csv_path.parents[1] / "models" / (self.csv_path.stem.replace("_training_curve", "_best") + ".zip")
        self.progress_bar = None
        self.previous_timesteps = 0
        self.start_time = None

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.progress_bar = tqdm(total=self.total_timesteps, desc=self.run_name, dynamic_ncols=True, leave=True)

    def _evaluate(self) -> Tuple[float, float]:
        rewards = []
        for episode_idx in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset(seed=episode_idx)
            terminated = truncated = False
            episode_reward = 0.0
            while not (terminated or truncated):
                action, _ = self.model.predict(obs, deterministic=self.deterministic_eval)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                episode_reward += float(reward)
            rewards.append(episode_reward)
        return float(np.mean(rewards)), float(np.std(rewards))

    def _on_step(self) -> bool:
        delta = int(self.num_timesteps - self.previous_timesteps)
        if self.progress_bar is not None and delta > 0:
            self.progress_bar.update(delta)
            self.previous_timesteps = int(self.num_timesteps)
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            train_rewards = [ep_info["r"] for ep_info in list(self.model.ep_info_buffer)] if self.model.ep_info_buffer else []
            train_mean = float(np.mean(train_rewards)) if train_rewards else 0.0
            train_std = float(np.std(train_rewards)) if train_rewards else 0.0
            eval_mean, eval_std = self._evaluate()
            elapsed = time.time() - self.start_time if self.start_time else 0.0
            eta = max((elapsed / max(self.num_timesteps, 1)) * max(self.total_timesteps - self.num_timesteps, 0), 0.0)
            self.records.append({
                "step": int(self.num_timesteps),
                "train_mean_reward": train_mean,
                "train_std_reward": train_std,
                "eval_mean_reward": eval_mean,
                "eval_std_reward": eval_std,
                "elapsed_sec": float(elapsed),
                "eta_sec": float(eta),
            })
            if self.progress_bar is not None:
                self.progress_bar.set_postfix({
                    "train_r": f"{train_mean:.2f}",
                    "eval_r": f"{eval_mean:.2f}",
                    "eta_min": f"{eta / 60.0:.1f}",
                })
            if eval_mean > self.best_mean:
                self.best_mean = eval_mean
                self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
                self.model.save(self.best_model_path)
        return True

    def _on_training_end(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.records).to_csv(self.csv_path, index=False)
        if self.progress_bar is not None:
            remaining = self.total_timesteps - self.progress_bar.n
            if remaining > 0:
                self.progress_bar.update(remaining)
            self.progress_bar.close()


def build_model(env, train_cfg: Dict, seed: int, tensorboard_log: Optional[str] = None) -> DQN:
    algo = train_cfg["algo"]
    return DQN(
        policy=str(algo.get("policy", "MlpPolicy")),
        env=env,
        policy_kwargs=dict(net_arch=list(algo.get("net_arch", [256, 256]))),
        learning_rate=float(algo.get("learning_rate", 5e-4)),
        buffer_size=int(algo.get("buffer_size", 15000)),
        learning_starts=int(algo.get("learning_starts", 200)),
        batch_size=int(algo.get("batch_size", 32)),
        gamma=float(algo.get("gamma", 0.8)),
        train_freq=int(algo.get("train_freq", 1)),
        gradient_steps=int(algo.get("gradient_steps", 1)),
        target_update_interval=int(algo.get("target_update_interval", 50)),
        seed=int(seed),
        verbose=0,
        tensorboard_log=tensorboard_log,
    )


def train_official_baseline(
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
    results_root = Path("results") / "official_baseline"
    model_path = results_root / "models" / f"{run_name}.zip"
    curve_csv = results_root / "metrics" / f"{run_name}_training_curve.csv"
    train_env = Monitor(make_env(env_config_path, seed=seed))
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
    model = DQN.load(Path(model_path))
    env_cfg = load_yaml(Path(env_config_path))
    risk_params = load_yaml(Path(risk_config_path))
    meta = parse_run_metadata(Path(model_path).stem)
    rows: List[Dict] = []
    trace_rows: List[Dict] = []
    risk_rows: List[Dict] = []
    frames = []
    for episode_idx in range(int(episodes)):
        render_mode = "rgb_array" if gif_path and episode_idx == 0 else None
        env = make_env(env_config_path, seed=seed + episode_idx, render_mode=render_mode)
        obs, info = env.reset(seed=seed + episode_idx)
        terminated = truncated = False
        step_idx = 0
        reward_sum = 0.0
        speed_values: List[float] = []
        lane_change_count = 0
        prev_lane = None
        ttc_min = 1e6
        thw_min = 1e6
        drac_max = 0.0
        risk_exposure = 0.0
        risk_score_max = 0.0
        near_miss_count = 0
        target_gap_values = []
        target_rear_distance_values = []
        target_rear_relative_speed_values = []
        p_yield_values = []
        p_block_values = []
        while not (terminated or truncated):
            if render_mode == "rgb_array":
                frame = env.render()
                if frame is not None:
                    frames.append(frame)
            action, _ = model.predict(obs, deterministic=deterministic_eval)
            obs, reward, terminated, truncated, info = env.step(action)
            reward_sum += float(reward)
            step_risk = _risk_step(env, risk_params)
            speed_now = _safe_speed(getattr(env, "unwrapped", env).vehicle) if getattr(getattr(env, "unwrapped", env), "vehicle", None) is not None else 0.0
            speed_values.append(speed_now)
            lane_now = _lane_id(getattr(env.unwrapped, "vehicle", None)) if getattr(env.unwrapped, "vehicle", None) is not None else None
            if prev_lane is not None and lane_now is not None and lane_now != prev_lane:
                lane_change_count += 1
            prev_lane = lane_now
            ttc_min = min(ttc_min, float(step_risk.get("ttc_min", 1e6)))
            thw_min = min(thw_min, float(step_risk.get("thw_min", 1e6)))
            drac_max = max(drac_max, float(step_risk.get("drac_max", 0.0)))
            risk_exposure += float(step_risk.get("risk_score", 0.0))
            risk_score_max = max(risk_score_max, float(step_risk.get("risk_score", 0.0)))
            if float(step_risk.get("near_miss_flag", 0.0)) > 0.5:
                near_miss_count += 1
            target_gap_values.append(float(step_risk.get("target_gap_size", 0.0)))
            target_rear_distance_values.append(float(step_risk.get("target_rear_distance", 0.0)))
            target_rear_relative_speed_values.append(float(step_risk.get("target_rear_relative_speed", 0.0)))
            p_yield_values.append(float(step_risk.get("p_yield", 0.0)))
            p_block_values.append(float(step_risk.get("p_block", 0.0)))
            if episode_idx == 0:
                trace_rows.append({
                    **meta,
                    "episode": int(episode_idx),
                    "step": int(step_idx),
                    "reward": float(reward),
                    "speed": float(speed_now),
                    "collision": int(_safe_collision(env)),
                    "action": int(action),
                    **step_risk,
                })
            step_idx += 1
        collision = int(_safe_collision(env))
        success = int(collision == 0)
        row = {
            **meta,
            "episode": int(episode_idx),
            "reward": float(reward_sum),
            "collision": collision,
            "success": success,
            "avg_speed": float(np.mean(speed_values) if speed_values else 0.0),
            "episode_length": int(step_idx),
            "lane_change_count": int(lane_change_count),
            "ttc_min": float(ttc_min),
            "thw_min": float(thw_min),
            "drac_max": float(drac_max),
            "risk_exposure": float(risk_exposure),
            "risk_score_max": float(risk_score_max),
            "near_miss_count": int(near_miss_count),
            "target_gap_size": float(np.mean(target_gap_values) if target_gap_values else 0.0),
            "target_rear_distance": float(np.mean(target_rear_distance_values) if target_rear_distance_values else 0.0),
            "target_rear_relative_speed": float(np.mean(target_rear_relative_speed_values) if target_rear_relative_speed_values else 0.0),
            "p_yield": float(np.mean(p_yield_values) if p_yield_values else 0.0),
            "p_block": float(np.mean(p_block_values) if p_block_values else 0.0),
        }
        rows.append(row)
        risk_row = dict(row)
        risk_row["collision_flag"] = collision
        risk_rows.append(risk_row)
        env.close()
    if gif_path is not None and frames:
        gif_path.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(gif_path, frames, fps=int(env_cfg.get("gif_fps", 15)))
    return pd.DataFrame(rows), pd.DataFrame(trace_rows), pd.DataFrame(risk_rows)


def aggregate_training_curves(csv_paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        if csv_path.exists():
            frames.append(pd.read_csv(csv_path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    grouped = df.groupby("step", as_index=False).agg(
        train_mean_reward_mean=("train_mean_reward", "mean"),
        train_mean_reward_std=("train_mean_reward", "std"),
        eval_mean_reward_mean=("eval_mean_reward", "mean"),
        eval_mean_reward_std=("eval_mean_reward", "std"),
    )
    return grouped.fillna(0.0)


def load_eval_metrics(csv_paths: Iterable[Path]) -> pd.DataFrame:
    frames = []
    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        if csv_path.exists():
            frames.append(pd.read_csv(csv_path))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    if "run_name" not in df.columns:
        metadata = pd.DataFrame([parse_run_metadata(str(name)) for name in df["method"].tolist()])
        for column in metadata.columns:
            if column not in df.columns:
                df[column] = metadata[column]
    return df


def summarize_episode_metrics(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    agg_spec = {f"{metric}_mean": (metric, "mean") for metric in EPISODE_METRICS}
    agg_spec.update({f"{metric}_std": (metric, "std") for metric in EPISODE_METRICS})
    summary = df.groupby(group_cols, as_index=False).agg(**agg_spec)
    return summary.fillna(0.0)


def summarize_across_seeds(seed_summary: pd.DataFrame) -> pd.DataFrame:
    if seed_summary.empty:
        return pd.DataFrame()
    grouped = seed_summary.groupby(["method", "steps"], as_index=False).agg(
        seed_count=("seed", "nunique"),
        reward_mean=("reward_mean", "mean"),
        reward_std=("reward_mean", "std"),
        collision_rate_mean=("collision_mean", "mean"),
        collision_rate_std=("collision_mean", "std"),
        success_rate_mean=("success_mean", "mean"),
        success_rate_std=("success_mean", "std"),
        avg_speed_mean=("avg_speed_mean", "mean"),
        avg_speed_std=("avg_speed_mean", "std"),
        episode_length_mean=("episode_length_mean", "mean"),
        episode_length_std=("episode_length_mean", "std"),
        lane_change_count_mean=("lane_change_count_mean", "mean"),
        lane_change_count_std=("lane_change_count_mean", "std"),
        near_miss_count_mean=("near_miss_count_mean", "mean"),
        near_miss_count_std=("near_miss_count_mean", "std"),
        p_yield_mean=("p_yield_mean", "mean"),
        p_yield_std=("p_yield_mean", "std"),
        p_block_mean=("p_block_mean", "mean"),
        p_block_std=("p_block_mean", "std"),
    )
    grouped["method_label"] = grouped.apply(lambda row: f"{row['method']} ({int(row['steps']) // 1000}k)", axis=1)
    return grouped.fillna(0.0)


def summarize_probability_metrics(risk_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if risk_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    stats_rows = []
    hist_rows = []
    bins = np.linspace(0.0, 1.0, 11)
    for column in ["p_yield", "p_block"]:
        if column not in risk_df.columns:
            continue
        values = risk_df[column].astype(float).clip(0.0, 1.0)
        stats_rows.append({
            "metric": column,
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "q25": float(values.quantile(0.25)),
            "median": float(values.median()),
            "q75": float(values.quantile(0.75)),
            "max": float(values.max()),
        })
        counts, edges = np.histogram(values, bins=bins)
        for left, right, count in zip(edges[:-1], edges[1:], counts):
            hist_rows.append({
                "metric": column,
                "bin_left": float(left),
                "bin_right": float(right),
                "count": int(count),
            })
    return pd.DataFrame(stats_rows), pd.DataFrame(hist_rows)


def aggregate_eval_metrics(csv_paths: Iterable[Path]) -> pd.DataFrame:
    df = load_eval_metrics(csv_paths)
    if df.empty:
        return pd.DataFrame()
    return summarize_episode_metrics(df, ["method", "steps", "seed"])
