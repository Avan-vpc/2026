import math
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


INF = 1e6
DEFAULT_LANE_WIDTH = 4.0


def _safe_scalar(vehicle: Any, name: str, default: float = 0.0) -> float:
    if vehicle is None:
        return float(default)
    value = getattr(vehicle, name, default)
    if callable(value):
        try:
            value = value()
        except TypeError:
            return float(default)
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _safe_speed(vehicle: Any) -> float:
    return _safe_scalar(vehicle, "speed", 0.0)


def _safe_position(vehicle: Any) -> np.ndarray:
    if vehicle is None:
        return np.zeros(2, dtype=float)
    pos = getattr(vehicle, "position", None)
    if callable(pos):
        try:
            pos = pos()
        except TypeError:
            pos = None
    if pos is None:
        return np.zeros(2, dtype=float)
    return np.asarray(pos, dtype=float)


def _lane_id(vehicle: Any) -> Tuple[Any, ...]:
    lane_index = getattr(vehicle, "lane_index", None)
    if isinstance(lane_index, tuple):
        return lane_index
    if lane_index is None:
        return tuple()
    return (lane_index,)


def _target_lane_id(vehicle: Any) -> Tuple[Any, ...]:
    lane_index = getattr(vehicle, "target_lane_index", None)
    if isinstance(lane_index, tuple):
        return lane_index
    if lane_index is None:
        return _lane_id(vehicle)
    return (lane_index,)


def _ego_and_vehicles(env: Any) -> Tuple[Any, Iterable[Any]]:
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    ego = getattr(unwrapped, "vehicle", None)
    road = getattr(unwrapped, "road", None)
    vehicles = getattr(road, "vehicles", []) if road is not None else []
    return ego, vehicles


def _infer_lane_width(env: Any, fallback: float = DEFAULT_LANE_WIDTH) -> float:
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    road = getattr(unwrapped, "road", None)
    network = getattr(road, "network", None)
    graph = getattr(network, "graph", {}) if network is not None else {}
    for outgoing in graph.values():
        for lanes in outgoing.values():
            if not lanes:
                continue
            lane = lanes[0]
            width = getattr(lane, "width", None)
            if callable(width):
                try:
                    width = width(0.0)
                except TypeError:
                    width = None
            if width is not None:
                try:
                    width_value = float(width)
                    if width_value > 0.0:
                        return width_value
                except (TypeError, ValueError):
                    continue
    return float(fallback)


def _neighbor_lanes(ego_lane: Tuple[Any, ...]) -> Tuple[Optional[Tuple[Any, ...]], Optional[Tuple[Any, ...]]]:
    if not ego_lane:
        return None, None
    lane_num = ego_lane[-1]
    left = ego_lane[:-1] + (lane_num - 1,) if lane_num > 0 else None
    right = ego_lane[:-1] + (lane_num + 1,)
    return left, right


def _is_lane_change_active(ego: Any) -> bool:
    lane_now = _lane_id(ego)
    target_lane = _target_lane_id(ego)
    return bool(lane_now and target_lane and lane_now != target_lane)


def _lateral_distance(ego: Any, veh: Any) -> float:
    return float(abs(_safe_position(veh)[1] - _safe_position(ego)[1]))


def compute_relative_speed(ego: Any, veh: Any) -> float:
    dx = _safe_position(veh)[0] - _safe_position(ego)[0]
    if dx >= 0.0:
        return _safe_speed(ego) - _safe_speed(veh)
    return _safe_speed(veh) - _safe_speed(ego)


def compute_ttc(ego: Any, veh: Any) -> float:
    dx = _safe_position(veh)[0] - _safe_position(ego)[0]
    distance = abs(dx)
    if distance < 1e-6:
        return 0.0
    closing_speed = compute_relative_speed(ego, veh)
    if closing_speed <= 1e-6:
        return math.inf
    return distance / closing_speed


def compute_thw(ego: Any, veh: Any) -> float:
    dx = _safe_position(veh)[0] - _safe_position(ego)[0]
    distance = abs(dx)
    follower_speed = _safe_speed(ego) if dx >= 0.0 else _safe_speed(veh)
    if follower_speed <= 1e-6:
        return math.inf
    return distance / follower_speed


def compute_drac(ego: Any, veh: Any) -> float:
    dx = _safe_position(veh)[0] - _safe_position(ego)[0]
    distance = max(abs(dx), 1e-6)
    closing_speed = compute_relative_speed(ego, veh)
    if closing_speed <= 1e-6:
        return 0.0
    return (closing_speed ** 2) / (2.0 * distance)


def compute_gap_size(ego: Any, front: Optional[Any], rear: Optional[Any]) -> float:
    ego_x = _safe_position(ego)[0]
    front_x = _safe_position(front)[0] if front is not None else ego_x + 80.0
    rear_x = _safe_position(rear)[0] if rear is not None else ego_x - 80.0
    return max(front_x - rear_x, 0.0)


def compute_local_density(vehicles: Iterable[Any], ego: Any, radius: float = 80.0) -> float:
    ego_pos = _safe_position(ego)
    count = 0
    for veh in vehicles:
        if veh is ego:
            continue
        if np.linalg.norm(_safe_position(veh) - ego_pos) <= radius:
            count += 1
    return count / max(radius, 1.0)


def find_lane_neighbors(env: Any, lane: Optional[Tuple[Any, ...]] = None) -> Tuple[Optional[Any], Optional[Any]]:
    ego, vehicles = _ego_and_vehicles(env)
    if ego is None:
        return None, None
    lane = lane or _lane_id(ego)
    ego_x = _safe_position(ego)[0]
    front = None
    rear = None
    best_front_dx = math.inf
    best_rear_dx = math.inf
    for veh in vehicles:
        if veh is ego or _lane_id(veh) != lane:
            continue
        dx = _safe_position(veh)[0] - ego_x
        if dx >= 0.0 and dx < best_front_dx:
            best_front_dx = dx
            front = veh
        if dx < 0.0 and abs(dx) < best_rear_dx:
            best_rear_dx = abs(dx)
            rear = veh
    return front, rear


def identify_target_gap_state(env: Any) -> Dict[str, Any]:
    ego, _ = _ego_and_vehicles(env)
    if ego is None:
        return {
            "target_lane": None,
            "target_front_vehicle": None,
            "target_rear_vehicle": None,
            "gap_size": 0.0,
            "rear_relative_speed": 0.0,
            "rear_acceleration": 0.0,
            "rear_ttc": INF,
            "rear_drac": 0.0,
            "front_ttc": INF,
            "front_thw": INF,
            "front_drac": 0.0,
            "target_conflict_active": 0.0,
            "lane_change_active": 0.0,
            "target_front_distance": INF,
            "target_rear_distance": INF,
        }
    ego_lane = _lane_id(ego)
    lane_change_active = _is_lane_change_active(ego)
    desired_lane = _target_lane_id(ego)
    lane_width = _infer_lane_width(env)
    if desired_lane and desired_lane != ego_lane:
        candidate_lanes = [desired_lane]
    else:
        left_lane, right_lane = _neighbor_lanes(ego_lane)
        candidate_lanes = [lane for lane in (left_lane, right_lane) if lane is not None]
    candidates = []
    for lane in candidate_lanes:
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
            "rear_ttc": INF,
            "rear_drac": 0.0,
            "front_ttc": INF,
            "front_thw": INF,
            "front_drac": 0.0,
            "target_conflict_active": 0.0,
            "lane_change_active": float(lane_change_active),
            "target_front_distance": INF,
            "target_rear_distance": INF,
        }
    _, lane, front, rear = max(candidates, key=lambda item: item[0])
    ego_x = _safe_position(ego)[0]
    rear_speed = _safe_scalar(rear, "speed", 0.0)
    ego_speed = _safe_scalar(ego, "speed", 0.0)
    rear_acc = _safe_scalar(rear, "acceleration", 0.0)
    target_front_distance = abs(_safe_position(front)[0] - ego_x) if front is not None else INF
    target_rear_distance = abs(_safe_position(rear)[0] - ego_x) if rear is not None else INF
    lateral_front = _lateral_distance(ego, front) if front is not None else INF
    lateral_rear = _lateral_distance(ego, rear) if rear is not None else INF
    target_conflict_active = lane_change_active or lateral_front <= lane_width or lateral_rear <= lane_width
    return {
        "target_lane": lane,
        "target_front_vehicle": front,
        "target_rear_vehicle": rear,
        "gap_size": compute_gap_size(ego, front, rear),
        "rear_relative_speed": rear_speed - ego_speed,
        "rear_acceleration": rear_acc,
        "rear_ttc": compute_ttc(ego, rear) if rear is not None else INF,
        "rear_drac": compute_drac(ego, rear) if rear is not None else 0.0,
        "front_ttc": compute_ttc(ego, front) if front is not None else INF,
        "front_thw": compute_thw(ego, front) if front is not None else INF,
        "front_drac": compute_drac(ego, front) if front is not None else 0.0,
        "target_conflict_active": float(target_conflict_active),
        "lane_change_active": float(lane_change_active),
        "target_front_distance": float(target_front_distance),
        "target_rear_distance": float(target_rear_distance),
    }


def collect_step_metrics(env: Any) -> Dict[str, float]:
    ego, vehicles = _ego_and_vehicles(env)
    if ego is None:
        return {
            "ttc_min": INF,
            "thw_min": INF,
            "drac_max": 0.0,
            "density": 0.0,
            "avg_speed_context": 0.0,
            "same_lane_front_ttc": INF,
            "same_lane_front_thw": INF,
            "same_lane_front_drac": 0.0,
            "same_lane_rear_ttc": INF,
            "same_lane_rear_drac": 0.0,
            "target_front_ttc": INF,
            "target_front_thw": INF,
            "target_front_drac": 0.0,
            "target_rear_ttc": INF,
            "target_rear_drac": 0.0,
            "target_gap_size": 0.0,
            "target_rear_distance": INF,
            "target_front_distance": INF,
            "target_rear_relative_speed": 0.0,
            "lane_change_active": 0.0,
            "target_conflict_active": 0.0,
        }

    same_front, same_rear = find_lane_neighbors(env, _lane_id(ego))
    target = identify_target_gap_state(env)

    same_lane_front_ttc = compute_ttc(ego, same_front) if same_front is not None else INF
    same_lane_front_thw = compute_thw(ego, same_front) if same_front is not None else INF
    same_lane_front_drac = compute_drac(ego, same_front) if same_front is not None else 0.0
    same_lane_rear_ttc = compute_ttc(ego, same_rear) if same_rear is not None else INF
    same_lane_rear_drac = compute_drac(ego, same_rear) if same_rear is not None else 0.0

    target_front_ttc = float(target.get("front_ttc", INF))
    target_front_thw = float(target.get("front_thw", INF))
    target_front_drac = float(target.get("front_drac", 0.0))
    target_rear_ttc = float(target.get("rear_ttc", INF))
    target_rear_drac = float(target.get("rear_drac", 0.0))
    target_conflict_active = bool(target.get("target_conflict_active", 0.0))

    ttc_candidates = [same_lane_front_ttc]
    thw_candidates = [same_lane_front_thw]
    drac_candidates = [same_lane_front_drac, same_lane_rear_drac]
    if target_conflict_active:
        ttc_candidates.extend([target_front_ttc, target_rear_ttc])
        thw_candidates.append(target_front_thw)
        drac_candidates.extend([target_front_drac, target_rear_drac])

    density = compute_local_density(vehicles, ego)
    avg_speed_context = float(np.mean([_safe_speed(v) for v in vehicles])) if vehicles else 0.0

    return {
        "ttc_min": float(min(ttc_candidates) if ttc_candidates else INF),
        "thw_min": float(min(thw_candidates) if thw_candidates else INF),
        "drac_max": float(max(drac_candidates) if drac_candidates else 0.0),
        "density": float(density),
        "avg_speed_context": float(avg_speed_context),
        "same_lane_front_ttc": float(same_lane_front_ttc),
        "same_lane_front_thw": float(same_lane_front_thw),
        "same_lane_front_drac": float(same_lane_front_drac),
        "same_lane_rear_ttc": float(same_lane_rear_ttc),
        "same_lane_rear_drac": float(same_lane_rear_drac),
        "target_front_ttc": float(target_front_ttc),
        "target_front_thw": float(target_front_thw),
        "target_front_drac": float(target_front_drac),
        "target_rear_ttc": float(target_rear_ttc),
        "target_rear_drac": float(target_rear_drac),
        "target_gap_size": float(target.get("gap_size", 0.0)),
        "target_rear_distance": float(target.get("target_rear_distance", INF)),
        "target_front_distance": float(target.get("target_front_distance", INF)),
        "target_rear_relative_speed": float(target.get("rear_relative_speed", 0.0)),
        "lane_change_active": float(target.get("lane_change_active", 0.0)),
        "target_conflict_active": float(target.get("target_conflict_active", 0.0)),
    }


def is_near_miss(metrics: Dict[str, float], thresholds: Dict[str, float]) -> bool:
    ttc_threshold = float(thresholds.get("ttc_near_miss", 2.0))
    drac_threshold = float(thresholds.get("drac_near_miss", 3.0))
    same_lane_front_hit = float(metrics.get("same_lane_front_ttc", INF)) < ttc_threshold
    target_active = bool(metrics.get("lane_change_active", 0.0) > 0.5 or metrics.get("target_conflict_active", 0.0) > 0.5)
    target_front_hit = target_active and float(metrics.get("target_front_ttc", INF)) < ttc_threshold
    target_rear_ttc_hit = target_active and float(metrics.get("target_rear_ttc", INF)) < ttc_threshold
    target_rear_drac_hit = target_active and float(metrics.get("target_rear_drac", 0.0)) > drac_threshold
    return bool(same_lane_front_hit or target_front_hit or target_rear_ttc_hit or target_rear_drac_hit)
