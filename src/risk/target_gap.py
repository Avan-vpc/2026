from typing import Any, Dict, Tuple

from src.risk.metrics import _lane_id, _safe_position, _safe_scalar, find_lane_neighbors, compute_gap_size, compute_ttc


def _neighbor_lanes(env: Any, ego_lane: Tuple[Any, ...]):
    lane_num = ego_lane[-1] if ego_lane else 0
    left = ego_lane[:-1] + (lane_num - 1,) if lane_num > 0 else None
    right = ego_lane[:-1] + (lane_num + 1,)
    return left, right


def identify_target_gap(env: Any) -> Dict[str, Any]:
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    ego = getattr(unwrapped, "vehicle", None)
    if ego is None:
        return {
            "target_lane": None,
            "target_front_vehicle": None,
            "target_rear_vehicle": None,
            "gap_size": 0.0,
            "rear_relative_speed": 0.0,
            "rear_acceleration": 0.0,
        }
    ego_lane = _lane_id(ego)
    left_lane, right_lane = _neighbor_lanes(env, ego_lane)
    candidates = []
    for lane in [left_lane, right_lane]:
        if lane is None:
            continue
        front, rear = find_lane_neighbors(env, lane)
        gap = compute_gap_size(ego, front, rear)
        candidates.append((gap, lane, front, rear))
    if not candidates:
        return {
            "target_lane": ego_lane,
            "target_front_vehicle": None,
            "target_rear_vehicle": None,
            "gap_size": 0.0,
            "rear_relative_speed": 0.0,
            "rear_acceleration": 0.0,
        }
    _, lane, front, rear = max(candidates, key=lambda item: item[0])
    ego_x = _safe_position(ego)[0]
    rear_speed = _safe_scalar(rear, "speed", 0.0)
    ego_speed = _safe_scalar(ego, "speed", 0.0)
    rear_acc = _safe_scalar(rear, "acceleration", 0.0)
    return {
        "target_lane": lane,
        "target_front_vehicle": front,
        "target_rear_vehicle": rear,
        "gap_size": compute_gap_size(ego, front, rear),
        "rear_relative_speed": rear_speed - ego_speed,
        "rear_acceleration": rear_acc,
        "rear_ttc": compute_ttc(ego, rear) if rear is not None else 1e6,
        "ego_x": ego_x,
    }
