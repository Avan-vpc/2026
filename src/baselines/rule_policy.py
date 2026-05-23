from typing import Any, Dict

from src.masking.safety_verifier import get_action_map
from src.risk.game_risk import compute_game_risk
from src.risk.metrics import _lane_id, collect_step_metrics, identify_target_gap_state
from src.risk.risk_score import compute_risk_scores


def _get_action(env: Any, action_name: str) -> int:
    action_map = get_action_map(env)
    return int(action_map.get(action_name, action_map.get("IDLE", 1)))


def _best_lane_change_action(env: Any, target_lane) -> int:
    ego = getattr(env.unwrapped, "vehicle", None)
    ego_lane = _lane_id(ego) if ego is not None else tuple()
    if not ego_lane or target_lane is None:
        return _get_action(env, "IDLE")
    ego_lane_num = ego_lane[-1]
    target_lane_num = target_lane[-1]
    if target_lane_num < ego_lane_num:
        return _get_action(env, "LANE_LEFT")
    if target_lane_num > ego_lane_num:
        return _get_action(env, "LANE_RIGHT")
    return _get_action(env, "IDLE")


class RuleTTCThwPolicy:
    def __init__(self, risk_params: Dict):
        self.risk_params = risk_params

    def suggest_action(self, env: Any) -> int:
        metrics = collect_step_metrics(env)
        gap = identify_target_gap_state(env)
        cfg = self.risk_params.get("risk_score_v2", {})
        ttc_threshold = float(cfg.get("ttc_threshold", 3.0))
        thw_threshold = float(cfg.get("thw_threshold", 1.0))
        safe_gap = float(self.risk_params.get("arbitration", {}).get("safe_gap", 18.0))

        if float(metrics.get("same_lane_front_ttc", 1e6)) < ttc_threshold or float(metrics.get("same_lane_front_thw", 1e6)) < thw_threshold:
            return _get_action(env, "SLOWER")

        lane_change_safe = (
            float(metrics.get("target_gap_size", 0.0)) >= safe_gap
            and float(metrics.get("target_rear_ttc", 1e6)) >= ttc_threshold
            and float(metrics.get("target_front_ttc", 1e6)) >= ttc_threshold
        )
        if lane_change_safe:
            return _best_lane_change_action(env, gap.get("target_lane"))

        if float(metrics.get("same_lane_front_ttc", 1e6)) > 2.0 * ttc_threshold:
            return _get_action(env, "FASTER")
        return _get_action(env, "IDLE")


class RuleDRACPolicy:
    def __init__(self, risk_params: Dict):
        self.risk_params = risk_params

    def suggest_action(self, env: Any) -> int:
        metrics = collect_step_metrics(env)
        gap = identify_target_gap_state(env)
        cfg = self.risk_params.get("risk_score_v2", {})
        drac_threshold = float(cfg.get("drac_threshold", 3.0))
        safe_gap = float(self.risk_params.get("arbitration", {}).get("safe_gap", 18.0))

        if float(metrics.get("drac_max", 0.0)) > drac_threshold:
            return _get_action(env, "SLOWER")

        lane_change_allowed = (
            float(metrics.get("target_rear_drac", 0.0)) <= drac_threshold
            and float(metrics.get("target_gap_size", 0.0)) >= safe_gap
        )
        if lane_change_allowed:
            return _best_lane_change_action(env, gap.get("target_lane"))

        if float(metrics.get("same_lane_front_drac", 0.0)) < 0.5 * drac_threshold:
            return _get_action(env, "FASTER")
        return _get_action(env, "IDLE")


class RuleRiskMinPolicy:
    def __init__(self, risk_params: Dict):
        self.risk_params = risk_params

    def _score_action(self, env: Any, action_name: str, metrics: Dict[str, float], risk_bundle: Dict[str, float]) -> float:
        score = float(risk_bundle.get("risk_score_v2", 0.0))
        if action_name == "SLOWER":
            score -= 0.35 * float(risk_bundle.get("same_lane_front_risk", 0.0))
            score -= 0.20 * float(risk_bundle.get("drac_term", 0.0))
        elif action_name in {"LANE_LEFT", "LANE_RIGHT"}:
            score += 0.50 * float(risk_bundle.get("target_rear_risk", 0.0))
            score += 0.30 * float(risk_bundle.get("target_front_risk", 0.0))
            score -= 0.25 * float(risk_bundle.get("same_lane_front_risk", 0.0))
        elif action_name == "FASTER":
            score += 0.30 * float(risk_bundle.get("same_lane_front_risk", 0.0))
            score += 0.20 * float(metrics.get("drac_max", 0.0))
            score -= 0.10
        else:
            score += 0.05
        return float(score)

    def suggest_action(self, env: Any) -> int:
        action_map = get_action_map(env)
        metrics = collect_step_metrics(env)
        game = compute_game_risk(env, self.risk_params)
        risk_bundle = compute_risk_scores(metrics, game, self.risk_params)
        candidates = [name for name in ["LANE_LEFT", "IDLE", "LANE_RIGHT", "FASTER", "SLOWER"] if name in action_map]
        best_name = min(candidates, key=lambda name: self._score_action(env, name, metrics, risk_bundle))
        return int(action_map[best_name])
