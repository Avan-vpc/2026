from typing import Any, Dict, Tuple

from src.risk.game_risk import compute_game_risk
from src.risk.metrics import collect_step_metrics
from src.risk.risk_score import compute_risk_scores


DEFAULT_ACTIONS = {
    "LANE_LEFT": 0,
    "IDLE": 1,
    "LANE_RIGHT": 2,
    "FASTER": 3,
    "SLOWER": 4,
}


def get_action_map(env: Any) -> Dict[str, int]:
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    action_type = getattr(unwrapped, "action_type", None)
    actions = getattr(action_type, "actions_indexes", None)
    if isinstance(actions, dict) and actions:
        return {str(k): int(v) for k, v in actions.items()}
    return DEFAULT_ACTIONS.copy()


def inverse_action_map(action_map: Dict[str, int]) -> Dict[int, str]:
    return {int(v): str(k) for k, v in action_map.items()}


def _unsafe_action_names() -> Tuple[str, ...]:
    return ("LANE_LEFT", "LANE_RIGHT", "FASTER")


class SafetyVerifier:
    def __init__(self, risk_params: Dict):
        self.risk_params = risk_params

    def score_state(self, env: Any) -> Dict[str, float]:
        metrics = collect_step_metrics(env)
        game_risk = compute_game_risk(env, self.risk_params)
        risk_bundle = compute_risk_scores(metrics, game_risk, self.risk_params)
        metrics.update(game_risk)
        metrics.update(risk_bundle)
        return metrics

    def is_safe(self, env: Any, action: int) -> bool:
        metrics = self.score_state(env)
        thresholds = self.risk_params.get("thresholds", {})
        action_map = get_action_map(env)
        inverse_map = inverse_action_map(action_map)
        action_name = inverse_map.get(int(action), "IDLE")
        if metrics["ttc_min"] < float(thresholds.get("ttc_mask", 2.5)) and action_name in _unsafe_action_names():
            return False
        if metrics["drac_max"] > float(thresholds.get("drac_mask", 3.0)) and action_name != "SLOWER":
            return False
        if metrics["risk_score"] > float(thresholds.get("risk_mask_threshold", 1.2)) and action_name in _unsafe_action_names():
            return False
        return True

    def fallback_action(self, env: Any) -> int:
        action_map = get_action_map(env)
        metrics = self.score_state(env)
        if metrics["risk_score"] > float(self.risk_params.get("thresholds", {}).get("risk_mask_threshold", 1.2)):
            return int(action_map.get("SLOWER", action_map.get("IDLE", 1)))
        return int(action_map.get("IDLE", 1))
