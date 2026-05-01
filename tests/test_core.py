from __future__ import annotations

import unittest

import numpy as np

from usv_sim.config import SimConfig
from usv_sim.dynamics import OtterUSV
from usv_sim.scenario import env_vector, make_demo_scenario
from usv_sim.simulator import simulate_scenario
from usv_sim.types import state6_to_state7


class CoreTest(unittest.TestCase):
    def test_mass_matrix_and_zero_step_are_finite(self) -> None:
        ship = OtterUSV()
        self.assertEqual(ship.mass_matrix.shape, (3, 3))
        state = np.array([0.0, 0.0, 0.0, 0.2, 0.0, 0.0])
        next_state = ship.rk4_step(state, np.zeros(2), 0.5)
        self.assertTrue(np.all(np.isfinite(next_state)))

    def test_twin_propeller_actuation(self) -> None:
        ship = OtterUSV()
        state = np.zeros(6)
        straight = ship.rk4_step(state, np.array([55.0, 55.0]), 0.5)
        turning = ship.rk4_step(state, np.array([65.0, 35.0]), 0.5)
        self.assertGreater(straight[3], 0.0)
        self.assertAlmostEqual(straight[5], 0.0, places=6)
        self.assertNotAlmostEqual(turning[5], 0.0, places=6)

    def test_env_and_state_shapes(self) -> None:
        scenario = make_demo_scenario()
        self.assertEqual(env_vector(scenario).shape, (57,))
        states6 = np.zeros((50, 6))
        self.assertEqual(state6_to_state7(states6).shape, (50, 7))

    def test_demo_simulation_shapes_and_constraints(self) -> None:
        sample = simulate_scenario(make_demo_scenario(), SimConfig(rng_seed=3), seed=3)
        self.assertEqual(sample.states.shape, (121, 7))
        self.assertEqual(sample.controls.shape, (120, 2))
        self.assertTrue(np.all(sample.controls >= -45.0 - 1e-9))
        self.assertTrue(np.all(sample.controls <= 76.0 + 1e-9))
        self.assertLessEqual(np.max(np.abs(np.diff(sample.controls[:, 0]))), 10.0 + 1e-9)
        self.assertLessEqual(np.max(np.abs(np.diff(sample.controls[:, 1]))), 10.0 + 1e-9)


if __name__ == "__main__":
    unittest.main()
