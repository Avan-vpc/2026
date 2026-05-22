from typing import Any, Dict

from src.masking.safety_verifier import get_action_map
from src.risk.game_risk import compute_game_risk


class RulePolicy:
    def __init__(self, risk_params: Dict):
        self.risk_params = risk_params

    def suggest_action(self, env: Any) -> int:
        action_map = get_action_map(env)
        game = compute_game_risk(env, self.risk_params)
        safe_gap = float(self.risk_params.get("arbitration", {}).get("safe_gap", 18.0))
        if game.get("gap_size", 0.0) >= safe_gap and game.get("P_yield", 0.0) >= 0.5:
            if action_map.get("LANE_LEFT") is not None:
                return int(action_map["LANE_LEFT"])
        if game.get("P_block", 0.0) > 0.6:
            return int(action_map.get("SLOWER", action_map.get("IDLE", 1)))
        return int(action_map.get("IDLE", 1))
