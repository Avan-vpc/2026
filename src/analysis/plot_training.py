from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import pandas as pd


def _save(fig, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_training_curves(csv_paths: Iterable[Path], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    for csv_path in csv_paths:
        csv_path = Path(csv_path)
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        if {"step", "train_mean_reward"}.issubset(df.columns):
            ax.plot(df["step"], df["train_mean_reward"], label=csv_path.stem)
    ax.set_xlabel("training steps")
    ax.set_ylabel("training mean reward")
    ax.legend()
    _save(fig, output_path)


def plot_mean_std_curve(
    df: pd.DataFrame,
    x_col: str,
    mean_col: str,
    std_col: Optional[str],
    output_path: Path,
    ylabel: str,
    title: str,
) -> None:
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df[x_col], df[mean_col], label="mean")
    if std_col and std_col in df.columns:
        lower = df[mean_col] - df[std_col]
        upper = df[mean_col] + df[std_col]
        ax.fill_between(df[x_col], lower, upper, alpha=0.2, label="std")
    ax.set_xlabel("training steps")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    _save(fig, output_path)
