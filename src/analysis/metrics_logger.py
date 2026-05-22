from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd


EPISODE_COLUMNS = [
    "method",
    "seed",
    "episode",
    "reward",
    "collision",
    "success",
    "episode_length",
    "avg_speed",
    "lane_change_count",
    "waiting_steps",
    "ttc_min",
    "thw_min",
    "drac_max",
    "risk_exposure",
    "risk_score_max",
    "near_miss_count",
    "raw_action",
    "final_action",
    "masked_action_count",
    "chosen_source",
]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_rows(path: Path, rows: Iterable[Mapping]) -> pd.DataFrame:
    ensure_parent(path)
    df = pd.DataFrame(list(rows))
    df.to_csv(path, index=False)
    return df
