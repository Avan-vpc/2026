import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.official_sb3_dqn import aggregate_training_curves
from src.analysis.plot_results import plot_risk_boxplots, plot_risk_case, plot_summary_bar
from src.analysis.plot_training import plot_mean_std_curve


def select_training_curves(metrics_dir: Path):
    hundred_k = sorted(metrics_dir.glob("official_sb3_dqn_highway_fast_steps100000_seed*_training_curve.csv"))
    if hundred_k:
        return hundred_k
    return sorted(metrics_dir.glob("official_sb3_dqn_highway_fast_*_training_curve.csv"))


def select_case_paths(metrics_dir: Path):
    hundred_k = sorted(metrics_dir.glob("official_sb3_dqn_highway_fast_steps100000_seed*_trace.csv"))
    if hundred_k:
        return hundred_k
    return sorted(metrics_dir.glob("official_sb3_dqn_highway_fast*_trace.csv"))


if __name__ == "__main__":
    metrics_dir = ROOT / "results" / "official_baseline" / "metrics"
    figures_dir = ROOT / "results" / "official_baseline" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    training_paths = select_training_curves(metrics_dir)
    agg = aggregate_training_curves(training_paths)
    if not agg.empty:
        agg.to_csv(metrics_dir / "official_sb3_dqn_highway_fast_training_agg.csv", index=False)
        plot_mean_std_curve(
            agg,
            x_col="step",
            mean_col="train_mean_reward_mean",
            std_col="train_mean_reward_std",
            output_path=figures_dir / "official_sb3_dqn_training_reward.png",
            ylabel="training mean reward",
            title="Official SB3 DQN on highway-fast-v0 (100k, 3 seeds)",
        )
        plot_mean_std_curve(
            agg,
            x_col="step",
            mean_col="eval_mean_reward_mean",
            std_col="eval_mean_reward_std",
            output_path=figures_dir / "official_sb3_dqn_eval_reward.png",
            ylabel="evaluation mean reward",
            title="Official SB3 DQN Evaluation on highway-fast-v0 (100k, 3 seeds)",
        )
    plot_summary_bar(
        metrics_dir / "official_sb3_dqn_highway_fast_eval_summary.csv",
        figures_dir / "official_sb3_dqn_eval_bar.png",
    )
    plot_risk_boxplots(
        metrics_dir / "official_dqn_risk_metrics.csv",
        figures_dir / "official_dqn_risk_boxplots.png",
    )
    case_paths = select_case_paths(metrics_dir)
    if case_paths:
        plot_risk_case(case_paths[0], figures_dir / "official_dqn_risk_case_timeseries.png")
