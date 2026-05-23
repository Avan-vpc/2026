import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.official_sb3_ppo import (
    aggregate_training_curves,
    evaluate_model,
    load_eval_metrics,
    load_yaml,
    summarize_across_seeds,
    summarize_episode_metrics,
    summarize_probability_metrics,
    train_official_ppo,
)
from src.analysis.plot_results import plot_risk_boxplots, plot_summary_bar
from src.analysis.plot_training import plot_mean_std_curve


def select_models(model_dir: Path, total_timesteps: int, seeds):
    selected = []
    for seed in seeds:
        model_path = model_dir / f"official_sb3_ppo_highway_fast_steps{int(total_timesteps)}_seed{int(seed)}.zip"
        if model_path.exists():
            selected.append(model_path)
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["smoke", "full", "three_seeds"], default="smoke")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    args = parser.parse_args()

    env_config = ROOT / "configs" / "highway_fast_official.yaml"
    risk_config = ROOT / "configs" / "risk_params.yaml"
    train_cfg = load_yaml(ROOT / "configs" / "train_ppo_official.yaml")
    runtime = train_cfg["runtime"]
    deterministic_eval = bool(runtime.get("deterministic_eval", True))

    if args.stage == "smoke":
        total_timesteps = int(runtime.get("smoke_timesteps", 20000))
        seeds = args.seeds or [0]
    elif args.stage == "full":
        total_timesteps = int(runtime.get("total_timesteps", 100000))
        seeds = args.seeds or [0]
    else:
        total_timesteps = int(runtime.get("total_timesteps", 100000))
        seeds = args.seeds or list(runtime.get("seeds", [0, 1, 2]))

    eval_freq = int(runtime.get("eval_freq", 5000))
    checkpoint_eval_episodes = int(runtime.get("checkpoint_eval_episodes", 20))
    final_eval_episodes = int(runtime.get("final_eval_episodes", 200)) if args.stage != "smoke" else 20

    for seed in seeds:
        result = train_official_ppo(
            env_config_path=env_config,
            train_config_path=ROOT / "configs" / "train_ppo_official.yaml",
            seed=int(seed),
            total_timesteps=int(total_timesteps),
            eval_freq=int(eval_freq),
            checkpoint_eval_episodes=int(checkpoint_eval_episodes),
            deterministic_eval=deterministic_eval,
            tag="official_sb3_ppo_highway_fast",
        )
        print(f"Saved model: {result['model_path']}")

    results_root = ROOT / "results" / "official_ppo"
    metrics_dir = results_root / "metrics"
    figures_dir = results_root / "figures"
    videos_dir = results_root / "videos"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)

    selected_models = select_models(results_root / "models", total_timesteps=total_timesteps, seeds=seeds)
    if not selected_models:
        raise FileNotFoundError(f"No PPO models found for steps={total_timesteps}, seeds={seeds}")

    eval_paths = []
    risk_frames = []
    for model_path in selected_models:
        stem = model_path.stem
        seed = int(stem.split("_seed")[-1])
        eval_df, trace_df, risk_df = evaluate_model(
            model_path=model_path,
            env_config_path=env_config,
            risk_config_path=risk_config,
            seed=seed,
            episodes=final_eval_episodes,
            deterministic_eval=deterministic_eval,
            gif_path=None,
        )
        eval_path = metrics_dir / f"{stem}_eval.csv"
        trace_path = metrics_dir / f"{stem}_trace.csv"
        risk_path = metrics_dir / f"{stem}_risk_metrics.csv"
        eval_df.to_csv(eval_path, index=False)
        trace_df.to_csv(trace_path, index=False)
        risk_df.to_csv(risk_path, index=False)
        eval_paths.append(eval_path)
        risk_frames.append(risk_df)

    eval_df = load_eval_metrics(eval_paths)
    if not eval_df.empty:
        seed_summary = summarize_episode_metrics(eval_df, ["method", "steps", "seed"])
        across = summarize_across_seeds(seed_summary)
        seed_summary.to_csv(metrics_dir / "official_ppo_seed_summary.csv", index=False)
        across.to_csv(metrics_dir / "official_ppo_across_seeds.csv", index=False)
        plot_summary_bar(metrics_dir / "official_ppo_across_seeds.csv", figures_dir / "official_ppo_eval_bar.png")

    training_paths = [
        metrics_dir / f"official_sb3_ppo_highway_fast_steps{int(total_timesteps)}_seed{int(seed)}_training_curve.csv"
        for seed in seeds
        if (metrics_dir / f"official_sb3_ppo_highway_fast_steps{int(total_timesteps)}_seed{int(seed)}_training_curve.csv").exists()
    ]
    agg = aggregate_training_curves(training_paths)
    if not agg.empty:
        agg.to_csv(metrics_dir / "official_ppo_training_agg.csv", index=False)
        plot_mean_std_curve(
            agg,
            x_col="step",
            mean_col="train_mean_reward_mean",
            std_col="train_mean_reward_std",
            output_path=figures_dir / "official_ppo_training_reward.png",
            ylabel="training mean reward",
            title="Official PPO on highway-fast-v0",
        )
        plot_mean_std_curve(
            agg,
            x_col="step",
            mean_col="eval_mean_reward_mean",
            std_col="eval_mean_reward_std",
            output_path=figures_dir / "official_ppo_eval_reward.png",
            ylabel="evaluation mean reward",
            title="Official PPO Evaluation on highway-fast-v0",
        )

    if risk_frames:
        risk_df = load_eval_metrics([metrics_dir / f"{path.stem}_risk_metrics.csv" for path in selected_models])
        risk_df.to_csv(metrics_dir / "official_ppo_risk_metrics.csv", index=False)
        stats_df, hist_df = summarize_probability_metrics(risk_df)
        if not stats_df.empty:
            stats_df.to_csv(metrics_dir / "official_ppo_probability_stats.csv", index=False)
        if not hist_df.empty:
            hist_df.to_csv(metrics_dir / "official_ppo_probability_histogram.csv", index=False)
        plot_risk_boxplots(metrics_dir / "official_ppo_risk_metrics.csv", figures_dir / "official_ppo_risk_boxplots.png")
