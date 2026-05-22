from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


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
    label_col = "method_label" if "method_label" in df.columns else "method"
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (column, title) in zip(axes, metrics):
        if column not in df.columns:
            continue
        sns.barplot(data=df, x=label_col, y=column, ax=ax, errorbar=None)
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


def plot_probability_histogram(hist_csv: Path, output_path: Path) -> None:
    hist_csv = Path(hist_csv)
    if not hist_csv.exists():
        return
    df = pd.read_csv(hist_csv)
    if df.empty or "metric" not in df.columns:
        return
    metrics = list(df["metric"].dropna().unique())
    if not metrics:
        return
    fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 4), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        sub = df[df["metric"] == metric].copy()
        centers = 0.5 * (sub["bin_left"] + sub["bin_right"])
        widths = sub["bin_right"] - sub["bin_left"]
        ax.bar(centers, sub["count"], width=widths * 0.95, align="center")
        ax.set_xlim(0.0, 1.0)
        ax.set_title(metric)
        ax.set_xlabel("value")
        ax.set_ylabel("count")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
