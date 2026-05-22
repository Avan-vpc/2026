from typing import Any, Dict

from src.risk.metrics import identify_target_gap_state


def identify_target_gap(env: Any) -> Dict[str, Any]:
    return identify_target_gap_state(env)
