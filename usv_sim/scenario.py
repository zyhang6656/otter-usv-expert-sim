from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .config import SimConfig
from .types import Obstacle, Scenario


def make_demo_scenario() -> Scenario:
    return make_demo_scenario_with_seed(0)


def make_demo_scenario_with_seed(
    seed: int,
    config: SimConfig | None = None,
    min_distance: float = 45.0,
    max_distance: float = 65.0,
    static_count: int = 6,
    dynamic_count: int = 3,
) -> Scenario:
    rng = np.random.default_rng(seed)
    return sample_scenario(
        rng,
        difficulty="medium",
        config=config or SimConfig(),
        min_distance=min_distance,
        max_distance=max_distance,
        static_count=static_count,
        dynamic_count=dynamic_count,
        name="random_otter_demo",
    )


def env_vector(scenario: Scenario, config: SimConfig | None = None) -> np.ndarray:
    config = config or SimConfig()
    parts: list[float] = []
    for i in range(config.max_static_obstacles):
        if i < len(scenario.static_obstacles):
            obs = scenario.static_obstacles[i]
            parts.extend([float(obs.center[0]), float(obs.center[1]), float(obs.radius)])
        else:
            parts.extend([0.0, 0.0, 0.0])
    for i in range(config.max_dynamic_obstacles):
        if i < len(scenario.dynamic_obstacles):
            obs = scenario.dynamic_obstacles[i]
            parts.extend(
                [
                    float(obs.center[0]),
                    float(obs.center[1]),
                    float(obs.radius),
                    float(obs.velocity[0]),
                    float(obs.velocity[1]),
                ]
            )
        else:
            parts.extend([0.0, 0.0, 0.0, 0.0, 0.0])
    parts.extend([float(scenario.goal[0]), float(scenario.goal[1])])
    out = np.asarray(parts, dtype=float)
    if out.shape != (57,):
        raise ValueError(f"env vector must be 57D, got {out.shape}")
    return out


def obstacle_clearance(point: np.ndarray, obstacles: Iterable[Obstacle], t: float, safety_margin: float = 0.0) -> float:
    clearances = []
    for obs in obstacles:
        if not obs.active:
            continue
        dist = np.linalg.norm(point - obs.position_at(t))
        clearances.append(float(dist - obs.radius - safety_margin))
    if not clearances:
        return float("inf")
    return min(clearances)


def _line_static_conflicts(
    start: np.ndarray,
    goal: np.ndarray,
    static_obstacles: list[Obstacle],
    clearance: float,
) -> int:
    seg = goal - start
    seg_norm_sq = float(np.dot(seg, seg))
    count = 0
    for obs in static_obstacles:
        if seg_norm_sq <= 1e-9:
            closest = start
        else:
            alpha = float(np.clip(np.dot(obs.center - start, seg) / seg_norm_sq, 0.0, 1.0))
            closest = start + alpha * seg
        if np.linalg.norm(closest - obs.center) <= obs.radius + clearance:
            count += 1
    return count


def _sample_point_clear(
    rng: np.random.Generator,
    obstacles: list[Obstacle],
    radius: float,
    workspace_size: float,
    max_tries: int = 1000,
) -> np.ndarray:
    for _ in range(max_tries):
        p = rng.uniform(8.0, workspace_size - 8.0, size=2)
        if all(np.linalg.norm(p - obs.center) > obs.radius + radius for obs in obstacles):
            return p
    raise RuntimeError("failed to sample a clear point")


def _sample_goal_with_distance(
    rng: np.random.Generator,
    start: np.ndarray,
    min_dist: float,
    max_dist: float,
    workspace_size: float,
    margin: float = 8.0,
    max_tries: int = 1000,
) -> np.ndarray:
    for _ in range(max_tries):
        distance = float(rng.uniform(min_dist, max_dist))
        angle = float(rng.uniform(-math.pi, math.pi))
        goal = start + distance * np.array([math.cos(angle), math.sin(angle)])
        if margin <= goal[0] <= workspace_size - margin and margin <= goal[1] <= workspace_size - margin:
            return goal
    raise RuntimeError("failed to sample a goal at the requested distance")


def _sample_near_segment(
    rng: np.random.Generator,
    start: np.ndarray,
    goal: np.ndarray,
    radius: float,
    workspace_size: float,
    lateral_min: float,
    lateral_max: float,
) -> np.ndarray:
    seg = goal - start
    seg_norm = float(np.linalg.norm(seg))
    if seg_norm < 1e-6:
        raise RuntimeError("degenerate start-goal segment")
    direction = seg / seg_norm
    normal = np.array([-direction[1], direction[0]])
    for _ in range(200):
        alpha = float(rng.uniform(0.18, 0.82))
        sign = -1.0 if rng.random() < 0.5 else 1.0
        offset = sign * float(rng.uniform(lateral_min, lateral_max))
        center = start + alpha * seg + offset * normal
        margin = radius + 4.0
        if margin <= center[0] <= workspace_size - margin and margin <= center[1] <= workspace_size - margin:
            return center
    return rng.uniform(10.0, workspace_size - 10.0, size=2)


def sample_scenario(
    rng: np.random.Generator,
    difficulty: str = "medium",
    config: SimConfig | None = None,
    min_distance: float | None = None,
    max_distance: float | None = None,
    static_count: int | None = None,
    dynamic_count: int | None = None,
    name: str | None = None,
) -> Scenario:
    config = config or SimConfig()
    specs = {
        "easy": ((2, 4), (1, 2), (2.0, 4.0), (0.2, 0.6), 30.0, 45.0, 1),
        "medium": ((5, 7), (2, 3), (2.0, 5.5), (0.2, 0.9), 45.0, 65.0, 2),
        "hard": ((8, 10), (3, 5), (2.5, 6.5), (0.2, 1.2), 60.0, 80.0, 2),
    }
    if difficulty not in specs:
        raise ValueError(f"unknown difficulty: {difficulty}")
    ns_range, nd_range, radius_range, speed_range, min_dist, max_dist, min_conflicts = specs[difficulty]
    if min_distance is not None:
        min_dist = float(min_distance)
    if max_distance is not None:
        max_dist = float(max_distance)

    for _ in range(300):
        static_obstacles: list[Obstacle] = []
        start_xy = rng.uniform(10.0, config.workspace_size - 10.0, size=2)
        goal = _sample_goal_with_distance(rng, start_xy, min_dist, max_dist, config.workspace_size)

        ns = int(static_count if static_count is not None else rng.integers(ns_range[0], ns_range[1] + 1))
        for _j in range(ns):
            radius = float(rng.uniform(radius_range[0], radius_range[1]))
            if _j < min_conflicts:
                center = _sample_near_segment(
                    rng,
                    start_xy,
                    goal,
                    radius,
                    config.workspace_size,
                    lateral_min=0.0,
                    lateral_max=radius + config.grid_clearance * 0.8,
                )
            else:
                center = _sample_near_segment(
                    rng,
                    start_xy,
                    goal,
                    radius,
                    config.workspace_size,
                    lateral_min=radius + 2.5,
                    lateral_max=12.0,
                )
            if np.linalg.norm(center - start_xy) < radius + 7.0 or np.linalg.norm(center - goal) < radius + 7.0:
                center = _sample_point_clear(rng, static_obstacles, radius + 2.0, config.workspace_size)
            if any(np.linalg.norm(center - obs.center) <= radius + obs.radius + 2.0 for obs in static_obstacles):
                center = _sample_point_clear(rng, static_obstacles, radius + 2.0, config.workspace_size)
            static_obstacles.append(Obstacle(center, radius, kind="static"))

        if obstacle_clearance(start_xy, static_obstacles, 0.0, 5.0) < 0.0:
            continue
        if obstacle_clearance(goal, static_obstacles, 0.0, 5.0) < 0.0:
            continue
        if _line_static_conflicts(start_xy, goal, static_obstacles, 2.0) < min_conflicts:
            continue

        dynamic_obstacles: list[Obstacle] = []
        nd = int(dynamic_count if dynamic_count is not None else rng.integers(nd_range[0], nd_range[1] + 1))
        seg = goal - start_xy
        seg_norm = max(float(np.linalg.norm(seg)), 1e-6)
        direction = seg / seg_norm
        normal = np.array([-direction[1], direction[0]])
        for _j in range(nd):
            radius = float(rng.uniform(1.0, max(1.4, radius_range[1] * 0.65)))
            alpha = float(rng.uniform(0.15, 0.85))
            sign = -1.0 if rng.random() < 0.5 else 1.0
            center = start_xy + alpha * seg + sign * float(rng.uniform(8.0, 18.0)) * normal
            center = np.clip(center, 10.0, config.workspace_size - 10.0)
            speed = float(rng.uniform(speed_range[0], speed_range[1]))
            velocity = -sign * speed * normal + float(rng.uniform(-0.15, 0.15)) * direction
            dynamic_obstacles.append(Obstacle(center, radius, velocity, kind="dynamic"))

        heading = float(math.atan2(goal[1] - start_xy[1], goal[0] - start_xy[0]))
        start = np.array([start_xy[0], start_xy[1], heading, 0.0, 0.0, 0.0], dtype=float)
        return Scenario(
            start=start,
            goal=goal,
            static_obstacles=static_obstacles,
            dynamic_obstacles=dynamic_obstacles,
            workspace=(0.0, config.workspace_size, 0.0, config.workspace_size),
            name=name or f"random_{difficulty}",
        )

    raise RuntimeError(f"failed to sample a {difficulty} scenario")
