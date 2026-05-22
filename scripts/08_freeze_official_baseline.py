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
    archive_root = ROOT / "results" / "legacy_unreliable" / "frozen_official_smoke"
    copy_if_exists(
        ROOT / "results" / "official_baseline" / "models" / "official_sb3_dqn_highway_fast_steps20000_seed0.zip",
        archive_root / "models" / "official_sb3_dqn_highway_fast_seed0.zip",
    )
    copy_if_exists(
        ROOT / "results" / "official_baseline" / "metrics" / "official_smoke_20k_summary.csv",
        archive_root / "metrics" / "official_smoke_20k_summary.csv",
    )
    copy_if_exists(
        ROOT / "results" / "official_baseline" / "figures" / "official_sb3_dqn_training_reward.png",
        archive_root / "figures" / "official_sb3_dqn_training_reward.png",
    )
    copy_if_exists(
        ROOT / "results" / "official_baseline" / "figures" / "official_sb3_dqn_eval_reward.png",
        archive_root / "figures" / "official_sb3_dqn_eval_reward.png",
    )
    copy_if_exists(
        ROOT / "results" / "official_baseline" / "videos" / "official_sb3_dqn_highway_fast_steps20000_seed0.gif",
        archive_root / "videos" / "official_sb3_dqn_rollout.gif",
    )
