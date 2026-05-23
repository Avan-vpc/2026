from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results" / "official_baseline" / "metrics"
FIGURES_DIR = ROOT / "results" / "official_baseline" / "figures"
INPUT_CSV = METRICS_DIR / "official_dqn_risk_metrics_fixed.csv"
REPORT_CSV = METRICS_DIR / "official_dqn_risk_calibration_report.csv"
REPORT_CSV_V2 = METRICS_DIR / "official_dqn_risk_calibration_report_v2.csv"
FIGURE_PNG = FIGURES_DIR / "official_dqn_risk_calibration.png"
FIGURE_PNG_V2 = FIGURES_DIR / "official_dqn_risk_calibration_v2.png"

PERCENTILES = [0.10, 0.25, 0.50, 0.75, 0.90]
PERCENTILE_NAMES = ["p10", "p25", "median", "p75", "p90"]
PERCENTILE_METRICS = ["risk_score_v1_max", "risk_score_v2_max", "drac_max", "ttc_min", "near_miss_count"]
AUC_METRICS = ["risk_score_v1_max", "risk_score_v2_max", "drac_max", "inverse_ttc_min", "near_miss_count"]


def _prepare_binary_labels(df: pd.DataFrame) -> pd.Series:
    if "collision_flag" in df.columns:
        return df["collision_flag"].fillna(0).astype(int)
    if "collision" in df.columns:
        return df["collision"].fillna(0).astype(int)
    raise KeyError("Missing collision label column")


def _safe_auc(labels: pd.Series, scores: pd.Series) -> float:
    valid = pd.DataFrame({"label": labels, "score": scores}).dropna()
    if valid.empty or valid["label"].nunique() < 2:
        return float("nan")
    valid = valid.sort_values("score", ascending=True).reset_index(drop=True)
    ranks = valid["score"].rank(method="average")
    pos = int((valid["label"] == 1).sum())
    neg = int((valid["label"] == 0).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    rank_sum = float(ranks[valid["label"] == 1].sum())
    auc = (rank_sum - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def _inverse_ttc(ttc: pd.Series) -> pd.Series:
    return 1.0 / ttc.clip(lower=1e-3, upper=10.0)


def _metric_series(df: pd.DataFrame, metric: str) -> pd.Series:
    if metric == "inverse_ttc_min":
        return _inverse_ttc(pd.to_numeric(df["ttc_min"], errors="coerce"))
    return pd.to_numeric(df[metric], errors="coerce")


def _near_miss_ratio(df: pd.DataFrame) -> pd.Series:
    near_miss = pd.to_numeric(df["near_miss_count"], errors="coerce")
    episode_length = pd.to_numeric(df["episode_length"], errors="coerce").clip(lower=1.0)
    return near_miss / episode_length


def _build_percentile_rows(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    rows = []
    groups = {
        "non_collision": df.loc[labels == 0],
        "collision": df.loc[labels == 1],
    }
    for group_name, group_df in groups.items():
        for metric in PERCENTILE_METRICS:
            if metric not in group_df.columns:
                continue
            values = pd.to_numeric(group_df[metric], errors="coerce").dropna()
            row = {"section": "percentiles", "group": group_name, "metric": metric}
            if values.empty:
                for name in PERCENTILE_NAMES:
                    row[name] = float("nan")
            else:
                quantiles = values.quantile(PERCENTILES).tolist()
                for name, value in zip(PERCENTILE_NAMES, quantiles):
                    row[name] = float(value)
            rows.append(row)
    return pd.DataFrame(rows)


def _build_auc_rows(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    rows = []
    for metric in AUC_METRICS:
        if metric != "inverse_ttc_min" and metric not in df.columns:
            continue
        rows.append(
            {
                "section": "auc",
                "group": "all",
                "metric": metric,
                "auc": _safe_auc(labels, _metric_series(df, metric)),
            }
        )
    return pd.DataFrame(rows)


def _build_recommendation_rows(df: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    collision_median_v2 = df.loc[labels == 1, "risk_score_v2_max"].median() if "risk_score_v2_max" in df.columns else float("nan")
    non_collision_median_v2 = df.loc[labels == 0, "risk_score_v2_max"].median() if "risk_score_v2_max" in df.columns else float("nan")
    auc_v1 = _safe_auc(labels, _metric_series(df, "risk_score_v1_max")) if "risk_score_v1_max" in df.columns else float("nan")
    auc_v2 = _safe_auc(labels, _metric_series(df, "risk_score_v2_max")) if "risk_score_v2_max" in df.columns else float("nan")
    auc_drac = _safe_auc(labels, _metric_series(df, "drac_max")) if "drac_max" in df.columns else float("nan")
    auc_ttc = _safe_auc(labels, _metric_series(df, "inverse_ttc_min")) if "ttc_min" in df.columns else float("nan")
    auc_near_miss = _safe_auc(labels, _metric_series(df, "near_miss_count")) if "near_miss_count" in df.columns else float("nan")

    near_miss_ratio_mean = float("nan")
    near_miss_ratio_p90 = float("nan")
    near_miss_ratio_ok = False
    if {"near_miss_count", "episode_length"}.issubset(df.columns):
        ratio = _near_miss_ratio(df).dropna()
        if not ratio.empty:
            near_miss_ratio_mean = float(ratio.mean())
            near_miss_ratio_p90 = float(ratio.quantile(0.90))
            near_miss_ratio_ok = near_miss_ratio_p90 < 0.50

    separation_ok = (
        pd.notna(collision_median_v2)
        and pd.notna(non_collision_median_v2)
        and collision_median_v2 > non_collision_median_v2
    )
    auc_ok = pd.notna(auc_v2) and auc_v2 >= 0.65

    if auc_ok and separation_ok and near_miss_ratio_ok:
        gate = "pass"
        recommendation = "risk_score_v2 passes the gate; DQN + risk_reward is allowed in the next step."
    elif pd.notna(auc_v2) and auc_v2 < 0.60:
        gate = "fail"
        reasons = []
        if not separation_ok:
            reasons.append("collision median risk_score_v2_max does not exceed non-collision median")
        if not near_miss_ratio_ok:
            reasons.append("near_miss_count remains too dense relative to episode_length")
        recommendation = (
            "risk_score_v2 still fails calibration; "
            + "; ".join(
                reasons
                + [
                    "increase target rear conflict sensitivity",
                    "reduce over-dominant same-lane term",
                    "retune clipped TTC/DRAC term weights before reward or masking",
                ]
            )
            + "."
        )
    else:
        gate = "borderline"
        reasons = []
        if not separation_ok:
            reasons.append("collision median separation is still weak")
        if not near_miss_ratio_ok:
            reasons.append("near_miss density is still too high")
        recommendation = (
            "risk_score_v2 improves but does not yet satisfy the reward gate; keep it as logging only"
            + (f" ({'; '.join(reasons)})." if reasons else ".")
        )

    return pd.DataFrame(
        [
            {
                "section": "recommendation",
                "group": "all",
                "metric": "risk_score_v2_gate",
                "gate": gate,
                "risk_score_v1_auc": auc_v1,
                "risk_score_v2_auc": auc_v2,
                "drac_auc": auc_drac,
                "inverse_ttc_auc": auc_ttc,
                "near_miss_auc": auc_near_miss,
                "risk_score_v2_median_collision": collision_median_v2,
                "risk_score_v2_median_non_collision": non_collision_median_v2,
                "near_miss_ratio_mean": near_miss_ratio_mean,
                "near_miss_ratio_p90": near_miss_ratio_p90,
                "recommended_formula": recommendation,
            }
        ]
    )


def _plot_calibration(df: pd.DataFrame, labels: pd.Series, output_path: Path) -> None:
    plot_df = df.copy()
    plot_df["collision_group"] = labels.map({0: "non_collision", 1: "collision"})
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    plot_specs = [
        ("risk_score_v1_max", "Risk score v1 max"),
        ("risk_score_v2_max", "Risk score v2 max"),
        ("drac_max", "DRAC max"),
        ("ttc_min", "TTC min"),
        ("near_miss_count", "Near-miss count"),
        ("target_rear_risk", "Target rear risk"),
    ]
    for ax, (column, title) in zip(axes, plot_specs):
        if column not in plot_df.columns:
            ax.axis("off")
            continue
        sns.boxplot(data=plot_df, x="collision_group", y=column, ax=ax)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=10)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Missing input CSV: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    if df.empty:
        raise ValueError("Input CSV is empty")

    labels = _prepare_binary_labels(df)
    percentile_df = _build_percentile_rows(df, labels)
    auc_df = _build_auc_rows(df, labels)
    recommendation_df = _build_recommendation_rows(df, labels)
    report_df = pd.concat([percentile_df, auc_df, recommendation_df], ignore_index=True, sort=False)

    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(REPORT_CSV, index=False)
    report_df.to_csv(REPORT_CSV_V2, index=False)

    _plot_calibration(df, labels, FIGURE_PNG)
    _plot_calibration(df, labels, FIGURE_PNG_V2)

    print(f"Wrote report: {REPORT_CSV}")
    print(f"Wrote report: {REPORT_CSV_V2}")
    print(f"Wrote figure: {FIGURE_PNG}")
    print(f"Wrote figure: {FIGURE_PNG_V2}")


if __name__ == "__main__":
    main()
