import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.official_sb3_dqn import load_yaml, make_env, parse_run_metadata, summarize_across_seeds, summarize_episode_metrics, _risk_step, _safe_collision, _safe_speed, _lane_id
from src.baselines.rule_policy import RuleDRACPolicy, RuleRiskMinPolicy, RuleTTCThwPolicy


sns.set_theme(style="whitegrid")


def _evaluate_rule_policy(
    run_name: str,
    policy,
    env_config_path: Path,
    risk_config_path: Path,
    seed: int,
    episodes: int,
    progress_bar=None,
) -> pd.DataFrame:
    meta = parse_run_metadata(run_name)
    risk_params = load_yaml(risk_config_path)
    rows: List[Dict] = []
    for episode_idx in range(int(episodes)):
        env = make_env(env_config_path, seed=seed + episode_idx)
        obs, info = env.reset(seed=seed + episode_idx)
        terminated = truncated = False
        step_idx = 0
        reward_sum = 0.0
        speed_values = []
        lane_change_count = 0
        prev_lane = None
        ttc_min = 1e6
        thw_min = 1e6
        drac_max = 0.0
        risk_exposure_v1 = 0.0
        risk_exposure_v2 = 0.0
        risk_score_v1_max = 0.0
        risk_score_v2_max = 0.0
        near_miss_count = 0
        target_gap_values = []
        target_rear_distance_values = []
        target_rear_relative_speed_values = []
        p_yield_values = []
        p_block_values = []
        same_lane_front_risk_values = []
        target_rear_risk_values = []
        target_front_risk_values = []
        while not (terminated or truncated):
            action = int(policy.suggest_action(env))
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
            risk_exposure_v1 += float(step_risk.get("risk_score_v1", 0.0))
            risk_exposure_v2 += float(step_risk.get("risk_score_v2", 0.0))
            risk_score_v1_max = max(risk_score_v1_max, float(step_risk.get("risk_score_v1", 0.0)))
            risk_score_v2_max = max(risk_score_v2_max, float(step_risk.get("risk_score_v2", 0.0)))
            if float(step_risk.get("near_miss_flag", 0.0)) > 0.5:
                near_miss_count += 1
            target_gap_values.append(float(step_risk.get("target_gap_size", 0.0)))
            target_rear_distance_values.append(float(step_risk.get("target_rear_distance", 0.0)))
            target_rear_relative_speed_values.append(float(step_risk.get("target_rear_relative_speed", 0.0)))
            p_yield_values.append(float(step_risk.get("p_yield", 0.0)))
            p_block_values.append(float(step_risk.get("p_block", 0.0)))
            same_lane_front_risk_values.append(float(step_risk.get("same_lane_front_risk", 0.0)))
            target_rear_risk_values.append(float(step_risk.get("target_rear_risk", 0.0)))
            target_front_risk_values.append(float(step_risk.get("target_front_risk", 0.0)))
            step_idx += 1
        collision = int(_safe_collision(env))
        rows.append({
            **meta,
            "episode": int(episode_idx),
            "reward": float(reward_sum),
            "collision": collision,
            "success": int(collision == 0),
            "avg_speed": float(sum(speed_values) / max(len(speed_values), 1)),
            "episode_length": int(step_idx),
            "lane_change_count": int(lane_change_count),
            "ttc_min": float(ttc_min),
            "thw_min": float(thw_min),
            "drac_max": float(drac_max),
            "risk_exposure": float(risk_exposure_v2),
            "risk_exposure_v1": float(risk_exposure_v1),
            "risk_exposure_v2": float(risk_exposure_v2),
            "risk_score_max": float(risk_score_v2_max),
            "risk_score_v1_max": float(risk_score_v1_max),
            "risk_score_v2_max": float(risk_score_v2_max),
            "same_lane_front_risk": float(sum(same_lane_front_risk_values) / max(len(same_lane_front_risk_values), 1)),
            "target_rear_risk": float(sum(target_rear_risk_values) / max(len(target_rear_risk_values), 1)),
            "target_front_risk": float(sum(target_front_risk_values) / max(len(target_front_risk_values), 1)),
            "near_miss_count": int(near_miss_count),
            "target_gap_size": float(sum(target_gap_values) / max(len(target_gap_values), 1)),
            "target_rear_distance": float(sum(target_rear_distance_values) / max(len(target_rear_distance_values), 1)),
            "target_rear_relative_speed": float(sum(target_rear_relative_speed_values) / max(len(target_rear_relative_speed_values), 1)),
            "p_yield": float(sum(p_yield_values) / max(len(p_yield_values), 1)),
            "p_block": float(sum(p_block_values) / max(len(p_block_values), 1)),
        })
        env.close()
        if progress_bar is not None:
            progress_bar.update(1)
            progress_bar.set_postfix_str(
                f"method={meta['method']} seed={seed} ep={episode_idx + 1}/{episodes} collision={collision}",
                refresh=False,
            )
    return pd.DataFrame(rows)


def _plot_rule_comparison(summary_df: pd.DataFrame, output_path: Path) -> None:
    if summary_df.empty:
        return
    metrics = ["collision_rate_mean", "avg_speed_mean", "success_rate_mean", "risk_score_v2_max_mean"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4))
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics):
        if metric not in summary_df.columns:
            ax.axis("off")
            continue
        sns.barplot(data=summary_df, x="method_label", y=metric, ax=ax, errorbar=None)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    env_config = ROOT / "configs" / "highway_fast_official.yaml"
    risk_config = ROOT / "configs" / "risk_params.yaml"
    train_cfg = load_yaml(ROOT / "configs" / "train_dqn_official.yaml")
    runtime = train_cfg["runtime"]
    seeds = [0] if args.stage == "smoke" else list(runtime.get("seeds", [0, 1, 2]))
    episodes = int(args.episodes) if args.episodes is not None else (20 if args.stage == "smoke" else 200)

    risk_params = load_yaml(risk_config)
    policies: List[Tuple[str, object]] = [
        ("rule_ttc_thw", RuleTTCThwPolicy(risk_params)),
        ("rule_drac", RuleDRACPolicy(risk_params)),
        ("rule_riskmin", RuleRiskMinPolicy(risk_params)),
    ]

    metrics_dir = ROOT / "results" / "rule_baselines" / "metrics"
    figures_dir = ROOT / "results" / "rule_baselines" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    total_runs = len(policies) * len(seeds) * int(episodes)
    frames = []
    with tqdm(total=total_runs, desc="Rule baseline evaluation", dynamic_ncols=True) as pbar:
        for method_name, policy in policies:
            for seed in seeds:
                run_name = f"{method_name}_steps0_seed{seed}"
                eval_df = _evaluate_rule_policy(
                    run_name,
                    policy,
                    env_config,
                    risk_config,
                    seed,
                    episodes,
                    progress_bar=pbar,
                )
                frames.append(eval_df)
                print(f"Evaluated {method_name} seed={seed} episodes={episodes}")

    eval_df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    eval_df.to_csv(metrics_dir / "rule_baselines_eval.csv", index=False)
    seed_summary = summarize_episode_metrics(eval_df, ["method", "steps", "seed"])
    across = summarize_across_seeds(seed_summary)
    seed_summary.to_csv(metrics_dir / "rule_baselines_summary.csv", index=False)
    across.to_csv(metrics_dir / "rule_baselines_across_seeds.csv", index=False)
    _plot_rule_comparison(across, figures_dir / "rule_baselines_comparison.png")
