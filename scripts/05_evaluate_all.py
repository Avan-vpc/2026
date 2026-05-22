import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.official_sb3_dqn import (
    evaluate_model,
    load_eval_metrics,
    load_yaml,
    summarize_across_seeds,
    summarize_episode_metrics,
    summarize_probability_metrics,
)


def select_models(model_dir: Path):
    models = sorted(model_dir.glob("official_sb3_dqn_highway_fast_steps*_seed*.zip"))
    if models:
        return models
    return sorted(model_dir.glob("official_sb3_dqn_highway_fast_*.zip"))


if __name__ == "__main__":
    env_config = ROOT / "configs" / "highway_fast_official.yaml"
    risk_config = ROOT / "configs" / "risk_params.yaml"
    train_cfg = load_yaml(ROOT / "configs" / "train_dqn_official.yaml")
    runtime = train_cfg["runtime"]
    final_eval_episodes = int(runtime.get("final_eval_episodes", 200))
    deterministic_eval = bool(runtime.get("deterministic_eval", True))
    model_dir = ROOT / "results" / "official_baseline" / "models"
    metrics_dir = ROOT / "results" / "official_baseline" / "metrics"
    videos_dir = ROOT / "results" / "official_baseline" / "videos"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    videos_dir.mkdir(parents=True, exist_ok=True)
    eval_paths = []
    risk_frames = []
    selected_models = select_models(model_dir)
    for model_path in selected_models:
        stem = model_path.stem
        seed = int(stem.split("_seed")[-1])
        eval_path = metrics_dir / f"{stem}_eval.csv"
        trace_path = metrics_dir / f"{stem}_trace.csv"
        risk_path = metrics_dir / f"{stem}_risk_metrics.csv"
        gif_path = videos_dir / f"{stem}.gif"
        eval_df, trace_df, risk_df = evaluate_model(
            model_path=model_path,
            env_config_path=env_config,
            risk_config_path=risk_config,
            seed=seed,
            episodes=final_eval_episodes,
            deterministic_eval=deterministic_eval,
            gif_path=gif_path if seed == 0 else None,
        )
        eval_df.to_csv(eval_path, index=False)
        trace_df.to_csv(trace_path, index=False)
        risk_df.to_csv(risk_path, index=False)
        eval_paths.append(eval_path)
        risk_frames.append(risk_df)
        print(f"Evaluated: {model_path.name}")

    eval_df = load_eval_metrics(eval_paths)
    if not eval_df.empty:
        smoke_df = eval_df[eval_df["steps"] <= 20000].copy()
        formal_df = eval_df[eval_df["steps"] >= 100000].copy()
        if not smoke_df.empty:
            smoke_summary = summarize_episode_metrics(smoke_df, ["method", "steps", "seed"])
            smoke_summary.to_csv(metrics_dir / "official_smoke_20k_summary.csv", index=False)
        if not formal_df.empty:
            seed_summary = summarize_episode_metrics(formal_df, ["method", "steps", "seed"])
            across_seeds = summarize_across_seeds(seed_summary)
            seed_summary.to_csv(metrics_dir / "official_100k_seed_summary.csv", index=False)
            across_seeds.to_csv(metrics_dir / "official_100k_across_seeds.csv", index=False)

    if risk_frames:
        risk_df = pd.concat(risk_frames, ignore_index=True)
        risk_df.to_csv(metrics_dir / "official_dqn_risk_metrics_fixed.csv", index=False)
        probability_stats, probability_hist = summarize_probability_metrics(risk_df)
        if not probability_stats.empty:
            probability_stats.to_csv(metrics_dir / "official_dqn_probability_stats.csv", index=False)
        if not probability_hist.empty:
            probability_hist.to_csv(metrics_dir / "official_dqn_probability_histogram.csv", index=False)
