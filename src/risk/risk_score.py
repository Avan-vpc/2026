import math
from typing import Dict


def _clip_inverse_term(value: float, threshold: float) -> float:
    clipped = min(float(value), float(threshold))
    return max(0.0, (float(threshold) - clipped) / max(float(threshold), 1e-6))


def _clip_drac_term(value: float, threshold: float, max_term: float) -> float:
    return min(max(float(value), 0.0) / max(float(threshold), 1e-6), float(max_term))


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


def compute_risk_score_v2(metrics: Dict[str, float], game_risk: Dict[str, float], risk_params: Dict) -> Dict[str, float]:
    cfg = risk_params.get("risk_score_v2", {})
    ttc_threshold = float(cfg.get("ttc_threshold", 3.0))
    thw_threshold = float(cfg.get("thw_threshold", 1.0))
    drac_threshold = float(cfg.get("drac_threshold", 3.0))
    max_term = float(cfg.get("max_term", 3.0))

    same_front_ttc_term = _clip_inverse_term(float(metrics.get("same_lane_front_ttc", 1e6)), ttc_threshold)
    same_front_thw_term = _clip_inverse_term(float(metrics.get("same_lane_front_thw", 1e6)), thw_threshold)
    same_front_drac_term = _clip_drac_term(float(metrics.get("same_lane_front_drac", 0.0)), drac_threshold, max_term)

    target_rear_ttc_term = _clip_inverse_term(float(metrics.get("target_rear_ttc", 1e6)), ttc_threshold)
    target_rear_drac_term = _clip_drac_term(float(metrics.get("target_rear_drac", 0.0)), drac_threshold, max_term)
    target_front_ttc_term = _clip_inverse_term(float(metrics.get("target_front_ttc", 1e6)), ttc_threshold)
    target_front_drac_term = _clip_drac_term(float(metrics.get("target_front_drac", 0.0)), drac_threshold, max_term)
    drac_term = _clip_drac_term(float(metrics.get("drac_max", 0.0)), drac_threshold, max_term)

    p_block = float(game_risk.get("P_block", 0.0))
    target_conflict_active = 1.0 if float(metrics.get("target_conflict_active", 0.0)) > 0.5 else 0.0

    same_lane_front_risk = 0.55 * same_front_ttc_term + 0.20 * same_front_thw_term + 0.25 * same_front_drac_term
    target_rear_risk = target_conflict_active * (
        0.60 * target_rear_ttc_term + 0.40 * target_rear_drac_term
    ) * (0.5 + 0.5 * p_block)
    target_front_risk = target_conflict_active * (
        0.65 * target_front_ttc_term + 0.35 * target_front_drac_term
    )

    score = (
        float(cfg.get("w_same_front", 2.0)) * same_lane_front_risk
        + float(cfg.get("w_target_rear", 2.5)) * target_rear_risk
        + float(cfg.get("w_target_front", 1.5)) * target_front_risk
        + float(cfg.get("w_drac", 1.5)) * drac_term
        + float(cfg.get("w_p_block", 0.5)) * p_block
    )
    if not math.isfinite(score):
        score = 0.0

    return {
        "risk_score_v2": float(score),
        "ttc_term": float(_clip_inverse_term(float(metrics.get("ttc_min", 1e6)), ttc_threshold)),
        "thw_term": float(_clip_inverse_term(float(metrics.get("thw_min", 1e6)), thw_threshold)),
        "drac_term": float(drac_term),
        "same_lane_front_risk": float(same_lane_front_risk),
        "target_rear_risk": float(target_rear_risk),
        "target_front_risk": float(target_front_risk),
        "p_block_term": float(p_block),
        "same_front_ttc_term": float(same_front_ttc_term),
        "same_front_thw_term": float(same_front_thw_term),
        "same_front_drac_term": float(same_front_drac_term),
        "target_rear_ttc_term": float(target_rear_ttc_term),
        "target_rear_drac_term": float(target_rear_drac_term),
        "target_front_ttc_term": float(target_front_ttc_term),
        "target_front_drac_term": float(target_front_drac_term),
        "target_conflict_active": float(target_conflict_active),
    }


def compute_risk_scores(metrics: Dict[str, float], game_risk: Dict[str, float], risk_params: Dict) -> Dict[str, float]:
    risk_v1 = compute_risk_score(metrics, game_risk, risk_params)
    risk_v2 = compute_risk_score_v2(metrics, game_risk, risk_params)
    return {
        "risk_score_v1": float(risk_v1),
        "risk_score": float(risk_v2["risk_score_v2"]),
        **risk_v2,
    }
