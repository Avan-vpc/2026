from typing import Any, Dict

from src.masking.safety_verifier import get_action_map, SafetyVerifier


class RiskMinPolicy:
    def __init__(self, risk_params: Dict):
        self.verifier = SafetyVerifier(risk_params)

    def suggest_action(self, env: Any) -> int:
        action_map = get_action_map(env)
        metrics = self.verifier.score_state(env)
        if metrics["risk_score"] > 1.0 or metrics["drac_max"] > 2.5:
            return int(action_map.get("SLOWER", action_map.get("IDLE", 1)))
        if metrics["ttc_min"] < 3.0:
            return int(action_map.get("IDLE", 1))
        return int(action_map.get("FASTER", action_map.get("IDLE", 1)))
