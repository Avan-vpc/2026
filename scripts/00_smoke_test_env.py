import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agents.official_sb3_dqn import load_yaml, make_env


if __name__ == "__main__":
    config_path = ROOT / "configs" / "highway_fast_official.yaml"
    env_cfg = load_yaml(config_path)
    env = make_env(config_path, seed=int(env_cfg.get("seed", 0)))
    obs, info = env.reset(seed=int(env_cfg.get("seed", 0)))
    total_reward = 0.0
    for step in range(100):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if terminated or truncated:
            break
    env.close()
    print(f"SMOKE {env_cfg['env']['id']}: steps={step + 1}, reward={total_reward:.3f}")
