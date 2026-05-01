from __future__ import annotations

import math

import numpy as np

from .config import ControllerConfig, SimConfig
from .dynamics import OtterUSV
from .types import Obstacle, wrap_angle


class SamplingNMPCController:
    def __init__(
        self,
        ship: OtterUSV,
        sim_config: SimConfig | None = None,
        controller_config: ControllerConfig | None = None,
        rng: np.random.Generator | None = None,
    ):
        self.ship = ship
        self.sim_config = sim_config or SimConfig()
        self.config = controller_config or ControllerConfig()
        self.rng = rng or np.random.default_rng(self.sim_config.rng_seed)
        self.mean = np.zeros((self.config.horizon, 2), dtype=float)

    def reset(self) -> None:
        self.mean.fill(0.0)

    def control(
        self,
        state: np.ndarray,
        prev_control: np.ndarray,
        subtarget: np.ndarray,
        obstacles: list[Obstacle],
        current_time: float,
    ) -> tuple[np.ndarray, dict[str, float]]:
        cfg = self.config
        std = np.tile(np.array([cfg.cem_std_n, cfg.cem_std_n], dtype=float), (cfg.horizon, 1))
        mean = self._seed_mean_from_guidance(state, prev_control, subtarget)

        best_seq = mean.copy()
        best_cost = float("inf")
        costs = np.empty(cfg.candidates + 2, dtype=float)

        for _ in range(cfg.iterations):
            samples = self.rng.normal(mean[None, :, :], std[None, :, :], size=(cfg.candidates, cfg.horizon, 2))
            samples = np.clip(
                samples,
                np.array([-cfg.delta_n_limit, -cfg.delta_n_limit]),
                np.array([cfg.delta_n_limit, cfg.delta_n_limit]),
            )
            zeros = np.zeros((1, cfg.horizon, 2), dtype=float)
            guided = mean[None, :, :]
            sequences = np.concatenate([samples, zeros, guided], axis=0)
            costs[:] = self._evaluate_sequences(state, prev_control, subtarget, obstacles, current_time, sequences)
            elite_idx = np.argpartition(costs, cfg.elites)[: cfg.elites]
            elites = sequences[elite_idx]
            elite_costs = costs[elite_idx]
            order = np.argsort(elite_costs)
            if float(elite_costs[order[0]]) < best_cost:
                best_cost = float(elite_costs[order[0]])
                best_seq = elites[order[0]].copy()
            mean = elites.mean(axis=0)
            std = np.maximum(elites.std(axis=0), np.array([1.5, 1.5]))

        control = self._controls_from_deltas(prev_control, best_seq[None, :, :], state)[0, 0]
        self.mean[:-1] = best_seq[1:]
        self.mean[-1] = 0.0
        return control, {"cost": best_cost}

    def _seed_mean_from_guidance(self, state: np.ndarray, prev_control: np.ndarray, subtarget: np.ndarray) -> np.ndarray:
        cfg = self.config
        vec = subtarget - state[:2]
        desired_heading = math.atan2(float(vec[1]), float(vec[0]))
        heading_error = float(wrap_angle(desired_heading - state[2]))
        dist = float(np.linalg.norm(vec))
        desired_speed = min(cfg.desired_speed, max(0.45, dist / 18.0))
        base_n = self.ship.shaft_speed_for_speed(desired_speed)
        diff_n = np.clip(9.0 * heading_error - 6.0 * state[5], -24.0, 24.0)
        desired = np.array([base_n + diff_n, base_n - diff_n], dtype=float)
        desired = np.clip(desired, cfg.n_min, cfg.n_max)
        prev = np.asarray(prev_control, dtype=float)
        first_delta = np.clip(
            desired - prev,
            np.array([-cfg.delta_n_limit, -cfg.delta_n_limit]),
            np.array([cfg.delta_n_limit, cfg.delta_n_limit]),
        )
        mean = self.mean.copy()
        mean *= 0.65
        mean[0] = 0.7 * mean[0] + 0.3 * first_delta
        for i in range(1, cfg.horizon):
            mean[i] = 0.75 * mean[i] + 0.25 * (first_delta * (0.85**i))
        return mean

    def _controls_from_deltas(self, prev_control: np.ndarray, sequences: np.ndarray, state: np.ndarray) -> np.ndarray:
        cfg = self.config
        prev = np.asarray(prev_control, dtype=float)
        controls = prev[None, None, :] + np.cumsum(sequences, axis=1)
        controls[:, :, 0] = np.clip(controls[:, :, 0], cfg.n_min, cfg.n_max)
        controls[:, :, 1] = np.clip(controls[:, :, 1], cfg.n_min, cfg.n_max)
        return controls

    def _evaluate_sequences(
        self,
        state: np.ndarray,
        prev_control: np.ndarray,
        subtarget: np.ndarray,
        obstacles: list[Obstacle],
        current_time: float,
        sequences: np.ndarray,
    ) -> np.ndarray:
        cfg = self.config
        sim_cfg = self.sim_config
        n = sequences.shape[0]
        states = np.repeat(state[None, :], n, axis=0)
        controls = self._controls_from_deltas(prev_control, sequences, state)
        cost = np.zeros(n, dtype=float)
        prev_two = np.repeat(np.asarray(prev_control, dtype=float)[None, :], n, axis=0)

        for h in range(cfg.horizon):
            ctrl = controls[:, h, :]
            delta = ctrl - prev_two
            prev_two = ctrl
            states = self.ship.rk4_step_batch(states, ctrl, sim_cfg.dt)
            invalid = ~np.all(np.isfinite(states), axis=1)
            if np.any(invalid):
                cost[invalid] += 1.0e12
                states[invalid] = 1.0e6
            t = current_time + (h + 1) * sim_cfg.dt
            pos = states[:, :2]

            dist_target = np.linalg.norm(pos - subtarget[None, :], axis=1)
            cost += cfg.running_target_weight * dist_target * dist_target
            desired_heading = np.arctan2(subtarget[1] - pos[:, 1], subtarget[0] - pos[:, 0])
            heading_error = wrap_angle(desired_heading - states[:, 2])
            cost += cfg.heading_weight * heading_error * heading_error
            cost += cfg.delta_weight_n * np.sum((delta / max(cfg.delta_n_limit, 1e-6)) ** 2, axis=1)
            cost += cfg.speed_weight * (states[:, 3] - cfg.desired_speed) ** 2
            cost += cfg.reverse_weight * np.maximum(0.0, -states[:, 3]) ** 2

            min_clearance = self._clearances(pos, obstacles, t)
            near = min_clearance < sim_cfg.obstacle_influence
            safe_clear = np.maximum(min_clearance, 0.05)
            apf = (1.0 / safe_clear - 1.0 / sim_cfg.obstacle_influence) ** 2
            cost += cfg.apf_weight * np.where(near, apf, 0.0)
            cost += cfg.collision_weight * np.maximum(0.0, -min_clearance + 0.05) ** 2
            cost += cfg.bounds_weight * self._bounds_violation(pos)

        final_dist = np.linalg.norm(states[:, :2] - subtarget[None, :], axis=1)
        cost += cfg.target_weight * final_dist * final_dist
        return cost

    def _clearances(self, positions: np.ndarray, obstacles: list[Obstacle], t: float) -> np.ndarray:
        if not obstacles:
            return np.full(positions.shape[0], np.inf, dtype=float)
        clearance = np.full(positions.shape[0], np.inf, dtype=float)
        for obs in obstacles:
            if not obs.active:
                continue
            center = obs.position_at(t)
            c = np.linalg.norm(positions - center[None, :], axis=1) - obs.radius - self.sim_config.safety_margin
            clearance = np.minimum(clearance, c)
        return clearance

    def _bounds_violation(self, positions: np.ndarray) -> np.ndarray:
        w = self.sim_config.workspace_size
        low = np.maximum(0.0, -positions)
        high = np.maximum(0.0, positions - w)
        return np.sum(low * low + high * high, axis=1)
