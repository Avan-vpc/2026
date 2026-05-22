from typing import Any, Dict, Tuple

import gymnasium as gym

from src.masking.safety_verifier import SafetyVerifier


class ActionMaskWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, risk_params: Dict):
        super().__init__(env)
        self.verifier = SafetyVerifier(risk_params)
        self.episode_masked_actions = 0

    def reset(self, **kwargs):
        self.episode_masked_actions = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        raw_action = int(action)
        final_action = raw_action
        masked = False
        if not self.verifier.is_safe(self.env, raw_action):
            final_action = self.verifier.fallback_action(self.env)
            masked = True
            self.episode_masked_actions += 1
        obs, reward, terminated, truncated, info = self.env.step(final_action)
        info = dict(info)
        info.update({
            "raw_action": raw_action,
            "final_action": final_action,
            "masked_action": int(masked),
            "masked_action_count": self.episode_masked_actions,
            "chosen_source": "rl" if not masked else "mask",
        })
        return obs, reward, terminated, truncated, info
