from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SimConfig:
    workspace_size: float = 100.0
    dt: float = 0.5
    n_steps: int = 181
    safety_margin: float = 1.0
    vessel_collision_radius: float = 3.3
    obstacle_influence: float = 12.0
    max_static_obstacles: int = 10
    max_dynamic_obstacles: int = 5
    grid_resolution: float = 2.0
    grid_clearance: float = 4.5
    subtarget_radius: float = 3.0
    goal_radius: float = 5.0
    rng_seed: int = 0

    @property
    def obstacle_clearance_margin(self) -> float:
        return self.vessel_collision_radius + self.safety_margin


@dataclass(frozen=True)
class ControllerConfig:
    horizon: int = 12
    candidates: int = 320
    iterations: int = 3
    elites: int = 40
    delta_n_limit: float = 10.0
    n_min: float = -45.0
    n_max: float = 76.0
    target_weight: float = 9.0
    running_target_weight: float = 0.08
    heading_weight: float = 1.8
    delta_weight_n: float = 0.18
    apf_weight: float = 12.0
    collision_weight: float = 2.5e5
    bounds_weight: float = 2.0e5
    reverse_weight: float = 450.0
    speed_weight: float = 8.0
    desired_speed: float = 1.25
    cem_std_n: float = 5.5


@dataclass(frozen=True)
class OtterConfig:
    # Maritime Robotics Otter USV data from public specifications and the
    # MSS/Fossen otter.m model. The 3-DOF implementation below keeps the
    # horizontal surge-sway-yaw dynamics and the twin-propeller actuation.
    mass: float = 62.0
    length: float = 2.0
    breadth: float = 1.08
    rho: float = 1025.0
    gravity: float = 9.81
    y_pontoon: float = 0.395
    r66_ratio: float = 0.25
    t_sway: float = 1.0
    t_yaw: float = 1.0
    u_max: float = 6.0 * 0.5144
    k_pos: float = 0.02216 / 2.0
    k_neg: float = 0.01289 / 2.0
    max_positive_bollard_pull: float = 24.4
    max_negative_bollard_pull: float = 13.6


# Backward-compatible name for modules that only need dimensions.
ShipConfig = OtterConfig
