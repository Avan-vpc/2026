import math
from typing import Any, Dict

from src.risk.target_gap import identify_target_gap


def _sigmoid(value: float) -> float:
    value = max(min(value, 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(-value))


def compute_game_risk(env: Any, risk_params: Dict) -> Dict[str, float]:
    gap = identify_target_gap(env)
    coeffs = risk_params.get("sigmoid", {})
    gap_size = float(gap.get("gap_size", 0.0))
    ttc_rear = float(gap.get("rear_ttc", 1e6))
    rear_closing_speed = max(float(gap.get("rear_relative_speed", 0.0)), 0.0)
    density = max(float(risk_params.get("density_override", 0.0)), 0.0)
    aggressive_indicator = 1.0 if float(gap.get("rear_acceleration", 0.0)) > 0.5 else 0.0

    gap_feature = (gap_size - 25.0) / 10.0
    ttc_feature = (min(ttc_rear, 8.0) - 4.0) / 2.0
    closing_feature = rear_closing_speed / 5.0
    density_feature = density * 20.0

    score = (
        float(coeffs.get("a0", -0.5))
        + float(coeffs.get("a1", 0.12)) * gap_feature
        + float(coeffs.get("a2", 0.30)) * ttc_feature
        - float(coeffs.get("a3", 0.25)) * closing_feature
        - float(coeffs.get("a4", 0.08)) * density_feature
        - float(coeffs.get("a5", 0.60)) * aggressive_indicator
    )
    p_yield = _sigmoid(score)
    p_block = 1.0 - p_yield
    rear_response_risk = p_block * max(1.0 - gap_size / 40.0, 0.0)
    target_gap_risk = p_block * (1.0 + rear_closing_speed / 10.0)
    return {
        "P_yield": float(p_yield),
        "P_block": float(p_block),
        "rear_response_risk": float(rear_response_risk),
        "target_gap_risk": float(target_gap_risk),
        "gap_size": float(gap_size),
    }
