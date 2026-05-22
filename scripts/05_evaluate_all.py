import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.official_sb3_dqn import aggregate_eval_metrics, evaluate_model, load_yaml


def select_models(model_dir: Path):
    hundred_k = sorted(model_dir.glob("official_sb3_dqn_highway_fast_steps100000_seed*.zip"))
    if hundred_k:
        return hundred_k
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
    summary = aggregate_eval_metrics(eval_paths)
    if not summary.empty:
        summary.to_csv(metrics_dir / "official_sb3_dqn_highway_fast_eval_summary.csv", index=False)
    if risk_frames:
        pd.concat(risk_frames, ignore_index=True).to_csv(metrics_dir / "official_dqn_risk_metrics.csv", index=False)
