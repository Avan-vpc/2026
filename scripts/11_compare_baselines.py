import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.official_sb3_dqn import summarize_across_seeds


sns.set_theme(style="whitegrid")


def _load_summary(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    if df.empty:
        return df
    if "seed" in df.columns:
        return summarize_across_seeds(df)
    return df


def _normalize_method_labels(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "method_label" not in df.columns:
        df["method_label"] = df["method"]
    df["method_label"] = df["method_label"].replace({
        "rule_ttc_thw (0k)": "Rule-TTC/THW",
        "rule_drac (0k)": "Rule-DRAC",
        "rule_riskmin (0k)": "Rule-RiskMin",
        "official_sb3_dqn_highway_fast (100k)": "Official DQN",
        "official_sb3_ppo_highway_fast (100k)": "Official PPO",
    })
    return df


def _plot_bar(df: pd.DataFrame, output_path: Path) -> None:
    metrics = [
        "collision_rate_mean",
        "avg_speed_mean",
        "success_rate_mean",
        "episode_length_mean",
        "risk_score_v2_max_mean",
        "near_miss_count_mean",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for ax, metric in zip(axes, metrics):
        if metric not in df.columns:
            ax.axis("off")
            continue
        sns.barplot(data=df, x="method_label", y=metric, ax=ax, errorbar=None)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_pareto(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df["collision_rate_mean"], df["avg_speed_mean"], s=90)
    for _, row in df.iterrows():
        ax.annotate(str(row["method_label"]), (row["collision_rate_mean"], row["avg_speed_mean"]), xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("collision rate")
    ax.set_ylabel("average speed")
    ax.set_title("Safety-Efficiency Pareto")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    dqn_path = ROOT / "results" / "official_baseline" / "metrics" / "official_100k_seed_summary.csv"
    rule_path = ROOT / "results" / "rule_baselines" / "metrics" / "rule_baselines_summary.csv"
    ppo_path = ROOT / "results" / "official_ppo" / "metrics" / "official_ppo_seed_summary.csv"

    frames = [df for df in [_load_summary(dqn_path), _load_summary(rule_path), _load_summary(ppo_path)] if not df.empty]
    if not frames:
        raise FileNotFoundError("No baseline summary CSVs found")

    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = _normalize_method_labels(merged)

    metrics_dir = ROOT / "results" / "baseline_comparison" / "metrics"
    figures_dir = ROOT / "results" / "baseline_comparison" / "figures"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    merged.to_csv(metrics_dir / "baseline_comparison_summary.csv", index=False)
    _plot_bar(merged, figures_dir / "baseline_safety_efficiency_bar.png")
    _plot_pareto(merged, figures_dir / "baseline_safety_efficiency_pareto.png")
