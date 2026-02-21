"""Signal controller utilities for automatic traffic light decision-making.

This module provides:
- compute_lane_densities: Convert lane vehicle counts to density percentages.
- select_green_lane: Decide which lane should get the green signal (highest density).
- assign_green_duration: Convert density to a green-time category (low/med/high -> seconds).
- generate_signal_plan: Produce a round-robin schedule using densities and durations.
"""
from typing import Dict, Tuple, List

# Default maximum vehicles per lane used to normalize densities.
MAX_VEHICLES_PER_LANE = 15

# Green durations (seconds) for low/medium/high traffic
GREEN_DURATION_LOW = 8
GREEN_DURATION_MED = 18
GREEN_DURATION_HIGH = 30


def compute_lane_densities(
    lane_counts: Dict[str, int],
    max_per_lane: int = MAX_VEHICLES_PER_LANE,
) -> Dict[str, float]:
    """Compute density percentage for each lane.

    Density = min(100, (count / max_per_lane) * 100).
    """
    densities = {}
    for lane, cnt in lane_counts.items():
        density = min(100.0, (cnt / float(max_per_lane)) * 100.0) if max_per_lane > 0 else 0.0
        densities[lane] = float(density)
    return densities


def select_green_lane(lane_densities: Dict[str, float]) -> str:
    """Select the lane that should get the green signal (highest density)."""
    if not lane_densities:
        return ''
    max_density = max(lane_densities.values())
    candidates = [lane for lane, d in lane_densities.items() if d == max_density]
    candidates.sort()
    return candidates[0]


def assign_green_duration(density: float) -> Tuple[str, int]:
    """Assign a duration label and green-time seconds for a density value.

    Returns (label, seconds).
    """
    d = max(0.0, min(100.0, float(density)))
    secs = int(round(GREEN_DURATION_LOW + (d / 100.0) * (GREEN_DURATION_HIGH - GREEN_DURATION_LOW)))
    if d < 30.0:
        label = 'low'
    elif d < 70.0:
        label = 'medium'
    else:
        label = 'high'
    return (label, int(secs))


def generate_signal_plan(lane_densities: Dict[str, float]) -> List[Dict]:
    """Generate a cyclic signal plan covering all lanes.

    Lanes are ordered by decreasing density (most congested served first).
    """
    ordered = sorted(lane_densities.items(), key=lambda kv: (-kv[1], kv[0]))
    plan = []
    for direction, dens in ordered:
        label, secs = assign_green_duration(dens)
        plan.append({
            'direction': direction,
            'density': float(dens),
            'duration_label': label,
            'duration_seconds': int(secs),
        })
    return plan
