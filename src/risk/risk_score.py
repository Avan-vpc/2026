import math
from typing import Dict


def compute_risk_score(metrics: Dict[str, float], game_risk: Dict[str, float], risk_params: Dict) -> float:
    weights = risk_params.get("weights", {})
    ttc = min(float(metrics.get("ttc_min", 1e6)), 10.0)
    thw = min(float(metrics.get("thw_min", 1e6)), 10.0)
    drac = min(float(metrics.get("drac_max", 0.0)), 10.0)
    density = float(metrics.get("density", 0.0))
    game = float(game_risk.get("target_gap_risk", 0.0))
    score = (
        float(weights.get("ttc", 1.0)) * (1.0 / max(ttc, 0.1))
        + float(weights.get("thw", 0.8)) * (1.0 / max(thw, 0.1))
        + float(weights.get("drac", 0.6)) * drac
        + float(weights.get("density", 0.3)) * density
        + float(weights.get("game", 0.5)) * game
    )
    if not math.isfinite(score):
        return 0.0
    return float(score)
