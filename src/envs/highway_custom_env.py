from pathlib import Path
from typing import Dict, Optional

import gymnasium as gym
import highway_env  # noqa: F401
import numpy as np
import yaml
from gymnasium import spaces

from src.arbitration.arbitrator import ArbitrationWrapper
from src.masking.action_mask import ActionMaskWrapper
from src.risk.game_risk import compute_game_risk
from src.risk.metrics import collect_step_metrics, is_near_miss, _safe_position, _safe_speed, _lane_id
from src.risk.risk_score import compute_risk_score


class FlatObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        low = np.full(int(np.prod(env.observation_space.shape)), -np.inf, dtype=np.float32)
        high = np.full(int(np.prod(env.observation_space.shape)), np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        return np.asarray(observation, dtype=np.float32).reshape(-1)


class RiskObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, risk_params: Dict):
        super().__init__(env)
        self.risk_params = risk_params
        base_dim = int(np.prod(env.observation_space.shape))
        extra_dim = 7
        low = np.full(base_dim + extra_dim, -np.inf, dtype=np.float32)
        high = np.full(base_dim + extra_dim, np.inf, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, observation):
        base = np.asarray(observation, dtype=np.float32).reshape(-1)
        metrics = collect_step_metrics(self.env)
        game = compute_game_risk(self.env, self.risk_params)
        risk_score = compute_risk_score(metrics, game, self.risk_params)
        extra = np.asarray([
            metrics["ttc_min"],
            metrics["thw_min"],
            metrics["drac_max"],
            metrics["density"],
            risk_score,
            game["P_yield"],
            game["P_block"],
        ], dtype=np.float32)
        extra = np.nan_to_num(extra, nan=0.0, posinf=1e3, neginf=-1e3)
        return np.concatenate([base, extra], axis=0)


class RiskAwareRewardWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, env_config: Dict, risk_params: Dict):
        super().__init__(env)
        self.env_config = env_config
        self.risk_params = risk_params
        self.prev_lane = None
        self.waiting_steps = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        ego = getattr(self.unwrapped, "vehicle", None)
        self.prev_lane = _lane_id(ego) if ego is not None else None
        self.waiting_steps = 0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        metrics = collect_step_metrics(self.env)
        game = compute_game_risk(self.env, self.risk_params)
        risk_score = compute_risk_score(metrics, game, self.risk_params)
        reward_cfg = self.risk_params.get("reward", {})
        shaped_reward = float(reward)
        shaped_reward -= float(reward_cfg.get("risk_penalty_scale", 0.15)) * risk_score
        shaped_reward -= float(reward_cfg.get("drac_penalty_scale", 0.1)) * metrics["drac_max"]
        ego = getattr(self.unwrapped, "vehicle", None)
        desired_speed = float(self.env_config.get("desired_speed", 22.0))
        desired_speed_ratio = float(reward_cfg.get("desired_speed_ratio", 0.85))
        lane_now = _lane_id(ego) if ego is not None else self.prev_lane
        if ego is not None and _safe_speed(ego) < desired_speed * desired_speed_ratio and lane_now == self.prev_lane:
            self.waiting_steps += 1
            shaped_reward -= float(reward_cfg.get("waiting_penalty_scale", 0.02))
        if self.prev_lane is not None and lane_now is not None and lane_now != self.prev_lane:
            shaped_reward -= float(reward_cfg.get("oscillation_penalty_scale", 0.01))
        self.prev_lane = lane_now
        info = dict(info)
        info["reward_raw"] = float(reward)
        info["reward_shaped"] = float(shaped_reward)
        info["risk_score"] = float(risk_score)
        return obs, shaped_reward, terminated, truncated, info


class MetricsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, env_config: Dict, risk_params: Dict):
        super().__init__(env)
        self.env_config = env_config
        self.risk_params = risk_params
        self.reset_episode_state()

    def reset_episode_state(self):
        self.episode_return = 0.0
        self.episode_length = 0
        self.speed_sum = 0.0
        self.lane_change_count = 0
        self.waiting_steps = 0
        self.risk_exposure = 0.0
        self.risk_score_max = 0.0
        self.near_miss_count = 0
        self.ttc_min_episode = 1e6
        self.thw_min_episode = 1e6
        self.drac_max_episode = 0.0
        self.prev_lane = None
        self.initial_x = 0.0
        self.last_raw_action = -1
        self.last_final_action = -1
        self.last_source = "rl"

    def reset(self, **kwargs):
        self.reset_episode_state()
        obs, info = self.env.reset(**kwargs)
        ego = getattr(self.unwrapped, "vehicle", None)
        self.prev_lane = _lane_id(ego) if ego is not None else None
        self.initial_x = float(_safe_position(ego)[0]) if ego is not None else 0.0
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        metrics = collect_step_metrics(self.env)
        game = compute_game_risk(self.env, self.risk_params)
        risk_score = compute_risk_score(metrics, game, self.risk_params)
        ego = getattr(self.unwrapped, "vehicle", None)
        lane_now = _lane_id(ego) if ego is not None else self.prev_lane
        speed_now = _safe_speed(ego) if ego is not None else 0.0
        desired_speed = float(self.env_config.get("desired_speed", 22.0))
        if self.prev_lane is not None and lane_now is not None and lane_now != self.prev_lane:
            self.lane_change_count += 1
        if lane_now == self.prev_lane and speed_now < desired_speed:
            self.waiting_steps += 1
        self.prev_lane = lane_now
        self.episode_return += float(reward)
        self.episode_length += 1
        self.speed_sum += speed_now
        self.risk_exposure += risk_score
        self.risk_score_max = max(self.risk_score_max, risk_score)
        self.ttc_min_episode = min(self.ttc_min_episode, float(metrics.get("ttc_min", 1e6)))
        self.thw_min_episode = min(self.thw_min_episode, float(metrics.get("thw_min", 1e6)))
        self.drac_max_episode = max(self.drac_max_episode, float(metrics.get("drac_max", 0.0)))
        thresholds = self.risk_params.get("thresholds", {})
        if is_near_miss(metrics, thresholds):
            self.near_miss_count += 1
        collision = int(bool(getattr(ego, "crashed", False))) if ego is not None else 0
        self.last_raw_action = int(info.get("raw_action", action))
        self.last_final_action = int(info.get("final_action", info.get("raw_action", action)))
        self.last_source = str(info.get("chosen_source", "rl"))
        info.update(metrics)
        info.update(game)
        info.update({
            "risk_score": risk_score,
            "collision": collision,
            "raw_action": self.last_raw_action,
            "final_action": self.last_final_action,
            "chosen_source": self.last_source,
            "masked_action_count": int(info.get("masked_action_count", 0)),
        })
        if terminated or truncated:
            current_x = float(_safe_position(ego)[0]) if ego is not None else self.initial_x
            progress = current_x - self.initial_x
            success = int(
                collision == 0
                and self.episode_return >= float(self.env_config.get("success_reward_threshold", 12.0))
                and progress >= float(self.env_config.get("progress_min_distance", 100.0))
            )
            info["episode_metrics"] = {
                "reward": self.episode_return,
                "collision": collision,
                "success": success,
                "episode_length": self.episode_length,
                "avg_speed": self.speed_sum / max(self.episode_length, 1),
                "lane_change_count": self.lane_change_count,
                "waiting_steps": self.waiting_steps,
                "ttc_min": self.ttc_min_episode,
                "thw_min": self.thw_min_episode,
                "drac_max": self.drac_max_episode,
                "risk_exposure": self.risk_exposure,
                "risk_score_max": self.risk_score_max,
                "near_miss_count": self.near_miss_count,
                "raw_action": self.last_raw_action,
                "final_action": self.last_final_action,
                "masked_action_count": int(info.get("masked_action_count", 0)),
                "chosen_source": self.last_source,
            }
        return obs, reward, terminated, truncated, info


def load_yaml(path: Path) -> Dict:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_env(
    env_config_path: Path,
    risk_config_path: Path,
    mode: str = "baseline",
    render_mode: Optional[str] = None,
):
    env_config = load_yaml(Path(env_config_path))
    risk_params = load_yaml(Path(risk_config_path))
    env = gym.make(env_config["env"]["id"], render_mode=render_mode)
    env.unwrapped.configure(env_config["env"].get("config", {}))
    env.reset()
    if mode == "masked":
        env = ActionMaskWrapper(env, risk_params)
    elif mode == "full":
        env = ArbitrationWrapper(env, risk_params)
    if mode in {"risk_reward", "masked", "full"}:
        env = RiskAwareRewardWrapper(env, env_config, risk_params)
    if mode == "full":
        env = RiskObservationWrapper(env, risk_params)
    else:
        env = FlatObservationWrapper(env)
    env = MetricsWrapper(env, env_config, risk_params)
    return env
