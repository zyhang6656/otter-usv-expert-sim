from __future__ import annotations

import numpy as np

from .config import OtterConfig
from .types import wrap_angle


class OtterUSV:
    """Planar 3-DOF approximation of the Maritime Robotics Otter USV.

    The model follows the MSS/Fossen `otter.m` structure in the horizontal
    plane: surge, sway, yaw, twin propellers, linear sway/yaw damping, and
    nonlinear yaw damping. State is [x, y, psi, u, v, r].
    """

    def __init__(self, config: OtterConfig | None = None):
        self.config = config or OtterConfig()
        c = self.config
        iz = c.mass * (c.r66_ratio * c.length) ** 2
        x_added = self._added_mass_surge(c.mass, c.length, c.rho)
        y_added = 1.5 * c.mass
        n_added = 1.7 * iz
        self.mass_matrix = np.diag([c.mass + x_added, c.mass + y_added, iz + n_added]).astype(float)
        self.x_u = -c.max_positive_bollard_pull * c.gravity / c.u_max
        self.y_v = -self.mass_matrix[1, 1] / c.t_sway
        self.n_r = -self.mass_matrix[2, 2] / c.t_yaw
        self.n_max = float(np.sqrt((0.5 * c.max_positive_bollard_pull * c.gravity) / c.k_pos))
        self.n_min = float(-np.sqrt((0.5 * c.max_negative_bollard_pull * c.gravity) / c.k_neg))

    @staticmethod
    def _added_mass_surge(mass: float, length: float, rho: float) -> float:
        volume = mass / rho
        return float(2.7 * rho * volume ** (5.0 / 3.0) / (length * length))

    def shaft_speed_for_speed(self, speed: float) -> float:
        speed = max(0.0, float(speed))
        required_total_thrust = max(0.0, -self.x_u * speed)
        per_prop = 0.5 * required_total_thrust
        return float(np.sqrt(per_prop / self.config.k_pos)) if per_prop > 0 else 0.0

    def propeller_forces_batch(self, shaft_speeds: np.ndarray) -> np.ndarray:
        c = self.config
        n = np.asarray(shaft_speeds, dtype=float)
        n = np.clip(n, self.n_min, self.n_max)
        thrust = np.where(n >= 0.0, c.k_pos * n * np.abs(n), c.k_neg * n * np.abs(n))
        tau = np.zeros((n.shape[0], 3), dtype=float)
        tau[:, 0] = thrust[:, 0] + thrust[:, 1]
        tau[:, 2] = c.y_pontoon * (thrust[:, 0] - thrust[:, 1])
        return tau

    def coriolis_batch(self, nu: np.ndarray) -> np.ndarray:
        u = nu[:, 0]
        v = nu[:, 1]
        m11 = self.mass_matrix[0, 0]
        m22 = self.mass_matrix[1, 1]
        out = np.zeros((nu.shape[0], 3, 3), dtype=float)
        out[:, 0, 2] = -m22 * v
        out[:, 1, 2] = m11 * u
        out[:, 2, 0] = m22 * v
        out[:, 2, 1] = -m11 * u
        return out

    def damping_forces_batch(self, nu: np.ndarray) -> np.ndarray:
        tau_damp = np.zeros_like(nu)
        tau_damp[:, 0] = self.x_u * nu[:, 0]
        tau_damp[:, 1] = self.y_v * nu[:, 1]
        tau_damp[:, 2] = self.n_r * (1.0 + 10.0 * np.abs(nu[:, 2])) * nu[:, 2]
        return tau_damp

    def derivative_batch(self, states: np.ndarray, shaft_speeds: np.ndarray) -> np.ndarray:
        states = np.asarray(states, dtype=float)
        shaft_speeds = np.asarray(shaft_speeds, dtype=float)
        if states.ndim == 1:
            states = states[None, :]
        if shaft_speeds.ndim == 1:
            shaft_speeds = np.broadcast_to(shaft_speeds[None, :], (states.shape[0], 2))

        psi = states[:, 2]
        nu = states[:, 3:6]
        u = nu[:, 0]
        v = nu[:, 1]
        r = nu[:, 2]

        d_eta = np.empty((states.shape[0], 3), dtype=float)
        d_eta[:, 0] = u * np.cos(psi) - v * np.sin(psi)
        d_eta[:, 1] = u * np.sin(psi) + v * np.cos(psi)
        d_eta[:, 2] = r

        tau_prop = self.propeller_forces_batch(shaft_speeds)
        tau_damp = self.damping_forces_batch(nu)
        coriolis = self.coriolis_batch(nu)
        rhs = tau_prop + tau_damp - np.einsum("nij,nj->ni", coriolis, nu)
        d_nu = np.linalg.solve(self.mass_matrix, rhs.T).T

        deriv = np.empty_like(states)
        deriv[:, 0:3] = d_eta
        deriv[:, 3:6] = d_nu
        return deriv

    def derivative(self, state: np.ndarray, shaft_speeds: np.ndarray) -> np.ndarray:
        return self.derivative_batch(state[None, :], shaft_speeds[None, :])[0]

    def rk4_step_batch(self, states: np.ndarray, shaft_speeds: np.ndarray, dt: float) -> np.ndarray:
        k1 = self.derivative_batch(states, shaft_speeds)
        k2 = self.derivative_batch(states + 0.5 * dt * k1, shaft_speeds)
        k3 = self.derivative_batch(states + 0.5 * dt * k2, shaft_speeds)
        k4 = self.derivative_batch(states + dt * k3, shaft_speeds)
        out = states + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[:, 2] = wrap_angle(out[:, 2])
        return out

    def rk4_step(self, state: np.ndarray, shaft_speeds: np.ndarray, dt: float) -> np.ndarray:
        return self.rk4_step_batch(state[None, :], shaft_speeds[None, :], dt)[0]


# Compatibility alias for older imports in notebooks or scripts.
CyberShipII = OtterUSV
