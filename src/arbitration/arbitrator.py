from typing import Dict

import gymnasium as gym

from src.arbitration.rule_policy import RulePolicy
from src.arbitration.risk_min_policy import RiskMinPolicy
from src.masking.safety_verifier import SafetyVerifier


class ArbitrationWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, risk_params: Dict):
        super().__init__(env)
        self.verifier = SafetyVerifier(risk_params)
        self.rule_policy = RulePolicy(risk_params)
        self.risk_min_policy = RiskMinPolicy(risk_params)
        self.episode_masked_actions = 0

    def reset(self, **kwargs):
        self.episode_masked_actions = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        raw_action = int(action)
        rl_safe = self.verifier.is_safe(self.env, raw_action)
        current = self.verifier.score_state(self.env)
        rule_action = self.rule_policy.suggest_action(self.env)
        risk_action = self.risk_min_policy.suggest_action(self.env)
        accept_threshold = float(self.verifier.risk_params.get("arbitration", {}).get("risk_accept_threshold", 0.9))
        if rl_safe and current["risk_score"] < accept_threshold:
            final_action = raw_action
            chosen_source = "rl"
        elif self.verifier.is_safe(self.env, rule_action):
            final_action = rule_action
            chosen_source = "rule"
        else:
            final_action = risk_action
            chosen_source = "risk_min"
        masked = int(final_action != raw_action)
        self.episode_masked_actions += masked
        if not self.verifier.is_safe(self.env, final_action):
            final_action = self.verifier.fallback_action(self.env)
            chosen_source = "fallback"
            self.episode_masked_actions += 1
        obs, reward, terminated, truncated, info = self.env.step(final_action)
        info = dict(info)
        info.update({
            "raw_action": raw_action,
            "final_action": int(final_action),
            "masked_action": int(final_action != raw_action),
            "masked_action_count": self.episode_masked_actions,
            "chosen_source": chosen_source,
            "rule_action": int(rule_action),
            "risk_min_action": int(risk_action),
        })
        return obs, reward, terminated, truncated, info
