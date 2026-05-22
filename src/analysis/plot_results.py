from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_summary_bar(metrics_csv: Path, output_path: Path) -> None:
    metrics_csv = Path(metrics_csv)
    if not metrics_csv.exists():
        return
    df = pd.read_csv(metrics_csv)
    if df.empty:
        return
    metrics = [
        ("collision_rate_mean", "collision rate"),
        ("avg_speed_mean", "average speed"),
        ("episode_length_mean", "episode length"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (column, title) in zip(axes, metrics):
        if column not in df.columns:
            continue
        sns.barplot(data=df, x="method", y=column, ax=ax, errorbar=None)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_risk_boxplots(risk_csv: Path, output_path: Path) -> None:
    risk_csv = Path(risk_csv)
    if not risk_csv.exists():
        return
    df = pd.read_csv(risk_csv)
    if df.empty or "collision_flag" not in df.columns:
        return
    df = df.copy()
    df["collision_group"] = df["collision_flag"].map({0: "non_collision", 1: "collision"})
    metrics = [
        ("ttc_min", "TTC min"),
        ("drac_max", "DRAC max"),
        ("risk_score_max", "risk score max"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (column, title) in zip(axes, metrics):
        if column not in df.columns:
            continue
        sns.boxplot(data=df, x="collision_group", y=column, ax=ax)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_risk_case(trace_csv: Path, output_path: Path) -> None:
    trace_csv = Path(trace_csv)
    if not trace_csv.exists():
        return
    df = pd.read_csv(trace_csv)
    if df.empty:
        return
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(df["step"], df["ttc_min"], label="TTC", color="tab:blue")
    axes[0].legend()
    axes[1].plot(df["step"], df["drac_max"], label="DRAC", color="tab:orange")
    axes[1].legend()
    axes[2].plot(df["step"], df["risk_score"], label="risk_score", color="tab:red")
    axes[2].legend()
    axes[3].plot(df["step"], df["action"], label="action", color="tab:green")
    axes[3].legend()
    axes[3].set_xlabel("step")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
