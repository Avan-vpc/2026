import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"copied: {src.name} -> {dst}")


if __name__ == "__main__":
    copy_if_exists(
        ROOT / "results" / "official_baseline" / "models" / "official_sb3_dqn_highway_fast_steps20000_seed0.zip",
        ROOT / "results" / "models" / "official_sb3_dqn_highway_fast_seed0.zip",
    )
    copy_if_exists(
        ROOT / "results" / "official_baseline" / "metrics" / "official_sb3_dqn_highway_fast_eval_summary.csv",
        ROOT / "results" / "metrics" / "official_sb3_dqn_highway_fast_eval_summary.csv",
    )
    copy_if_exists(
        ROOT / "results" / "official_baseline" / "figures" / "official_sb3_dqn_training_reward.png",
        ROOT / "results" / "figures" / "official_sb3_dqn_training_reward.png",
    )
    copy_if_exists(
        ROOT / "results" / "official_baseline" / "figures" / "official_sb3_dqn_eval_reward.png",
        ROOT / "results" / "figures" / "official_sb3_dqn_eval_reward.png",
    )
    copy_if_exists(
        ROOT / "results" / "official_baseline" / "videos" / "official_sb3_dqn_highway_fast_steps20000_seed0.gif",
        ROOT / "results" / "videos" / "official_sb3_dqn_rollout.gif",
    )
