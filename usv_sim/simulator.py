from __future__ import annotations

import time
from typing import Any

import numpy as np

from .config import ControllerConfig, SimConfig
from .controller import SamplingNMPCController
from .dynamics import OtterUSV
from .planner import theta_star_path
from .scenario import env_vector, obstacle_clearance
from .types import Scenario, TrajectorySample, state6_to_state7


def simulate_scenario(
    scenario: Scenario,
    sim_config: SimConfig | None = None,
    controller_config: ControllerConfig | None = None,
    seed: int = 0,
    record_trace: bool = True,
    stop_on_reach: bool = False,
) -> TrajectorySample:
    sim_config = sim_config or SimConfig(rng_seed=seed)
    rng = np.random.default_rng(seed)
    ship = OtterUSV()
    controller = SamplingNMPCController(ship, sim_config, controller_config, rng)
    path = theta_star_path(scenario, sim_config)

    states6 = np.zeros((sim_config.n_steps, 6), dtype=float)
    controls = np.zeros((sim_config.n_steps - 1, 2), dtype=float)
    states6[0] = scenario.start
    prev_control = np.zeros(2, dtype=float)
    subtarget_idx = 1 if len(path) > 1 else 0
    reached = False
    collided = False
    min_clearance = float("inf")
    saturation_count = 0
    trace: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    obstacles = scenario.all_obstacles()
    for k in range(sim_config.n_steps - 1):
        t = k * sim_config.dt
        state = states6[k]
        if subtarget_idx < len(path) - 1 and np.linalg.norm(state[:2] - path[subtarget_idx]) < sim_config.subtarget_radius:
            subtarget_idx += 1
        subtarget = path[subtarget_idx]
        control, info = controller.control(state, prev_control, subtarget, obstacles, t)
        controls[k] = control
        states6[k + 1] = ship.rk4_step(state, control, sim_config.dt)
        prev_control = control

        clearance = obstacle_clearance(
            states6[k + 1, :2],
            obstacles,
            t + sim_config.dt,
            sim_config.obstacle_clearance_margin,
        )
        min_clearance = min(min_clearance, clearance)
        if clearance < 0.0:
            collided = True
        if np.any(control <= 0.98 * ship.n_min) or np.any(control >= 0.98 * ship.n_max):
            saturation_count += 1
        dist_goal = float(np.linalg.norm(states6[k + 1, :2] - scenario.goal))
        if dist_goal <= sim_config.goal_radius:
            reached = True
            if stop_on_reach:
                if k + 2 < sim_config.n_steps:
                    states6[k + 2 :] = states6[k + 1]
                    controls[k + 1 :] = 0.0
                break

        if record_trace:
            trace.append(
                {
                    "step": k,
                    "time": t,
                    "subtarget_idx": int(subtarget_idx),
                    "subtarget": subtarget.tolist(),
                    "clearance": float(clearance),
                    "dist_goal": dist_goal,
                    "control_cost": float(info["cost"]),
                }
            )

    final_distance = float(np.linalg.norm(states6[-1, :2] - scenario.goal))
    reached = reached or final_distance <= sim_config.goal_radius
    controls_saturated_ratio = saturation_count / max(1, len(controls))
    success = bool(reached and not collided)
    metadata = {
        "success": success,
        "expert_filter_passed": bool(success and controls_saturated_ratio <= 0.9),
        "reached": bool(reached),
        "collided": bool(collided),
        "final_distance": final_distance,
        "min_clearance": float(min_clearance),
        "vessel_collision_radius": float(sim_config.vessel_collision_radius),
        "obstacle_clearance_margin": float(sim_config.obstacle_clearance_margin),
        "control_saturation_ratio": float(controls_saturated_ratio),
        "initial_distance": float(np.linalg.norm(scenario.goal - scenario.start[:2])),
        "static_obstacle_count": int(len(scenario.static_obstacles)),
        "dynamic_obstacle_count": int(len(scenario.dynamic_obstacles)),
        "vehicle": "OtterUSV",
        "dt": sim_config.dt,
        "n_steps": sim_config.n_steps,
        "runtime_sec": float(time.perf_counter() - t0),
    }
    if record_trace:
        metadata["trace"] = trace
    return TrajectorySample(
        states=state6_to_state7(states6),
        controls=controls,
        env57=env_vector(scenario, sim_config),
        path=path,
        metadata=metadata,
        scenario=scenario,
    )
