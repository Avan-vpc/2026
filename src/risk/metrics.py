import math
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np


INF = 1e6


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


def _ego_and_vehicles(env: Any) -> Tuple[Any, Iterable[Any]]:
    unwrapped = env.unwrapped if hasattr(env, "unwrapped") else env
    ego = getattr(unwrapped, "vehicle", None)
    road = getattr(unwrapped, "road", None)
    vehicles = getattr(road, "vehicles", []) if road is not None else []
    return ego, vehicles


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
    front_x = _safe_position(front)[0] if front is not None else ego_x + 100.0
    rear_x = _safe_position(rear)[0] if rear is not None else ego_x - 100.0
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


def collect_step_metrics(env: Any) -> Dict[str, float]:
    ego, vehicles = _ego_and_vehicles(env)
    if ego is None:
        return {
            "ttc_min": INF,
            "thw_min": INF,
            "drac_max": 0.0,
            "density": 0.0,
            "avg_speed_context": 0.0,
        }
    ttc_values = []
    thw_values = []
    drac_values = []
    for veh in vehicles:
        if veh is ego:
            continue
        ttc_values.append(compute_ttc(ego, veh))
        thw_values.append(compute_thw(ego, veh))
        drac_values.append(compute_drac(ego, veh))
    density = compute_local_density(vehicles, ego)
    avg_speed_context = float(np.mean([_safe_speed(v) for v in vehicles])) if vehicles else 0.0
    return {
        "ttc_min": float(min(ttc_values) if ttc_values else INF),
        "thw_min": float(min(thw_values) if thw_values else INF),
        "drac_max": float(max(drac_values) if drac_values else 0.0),
        "density": float(density),
        "avg_speed_context": avg_speed_context,
    }
