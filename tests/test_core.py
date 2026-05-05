from __future__ import annotations

import unittest
from pathlib import Path
from uuid import uuid4

import numpy as np

from usv_sim.config import SimConfig
from usv_sim.dynamics import OtterUSV
from usv_sim.dashboard import DashboardState, _balanced_targets, _balanced_worker_count
from usv_sim.io import save_dataset_npz
from usv_sim.scenario import env_vector, make_demo_scenario, sample_scenario
from usv_sim.simulator import simulate_scenario
from usv_sim.types import Obstacle, Scenario, TrajectorySample, state6_to_state7


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
        self.assertEqual(sample.states.shape, (181, 7))
        self.assertEqual(sample.controls.shape, (180, 2))
        self.assertTrue(np.all(sample.controls >= -45.0 - 1e-9))
        self.assertTrue(np.all(sample.controls <= 76.0 + 1e-9))
        self.assertLessEqual(np.max(np.abs(np.diff(sample.controls[:, 0]))), 10.0 + 1e-9)
        self.assertLessEqual(np.max(np.abs(np.diff(sample.controls[:, 1]))), 10.0 + 1e-9)

    def test_sampled_scenarios_are_at_least_80m(self) -> None:
        rng = np.random.default_rng(7)
        for difficulty in ("easy", "medium", "hard"):
            scenario = sample_scenario(rng, difficulty)
            distance = float(np.linalg.norm(scenario.goal - scenario.start[:2]))
            self.assertGreaterEqual(distance, 80.0)

    def test_dashboard_loads_saved_paths_and_difficulty(self) -> None:
        sample = simulate_scenario(make_demo_scenario(), SimConfig(rng_seed=5), seed=5)
        sample.metadata["difficulty"] = "hard"
        path = Path("tests") / f".dashboard_roundtrip_{uuid4().hex}.npz"
        generated_path = path.with_name(f"{path.stem}_generated.npz")
        try:
            save_dataset_npz([sample], path, {"seed": 5})
            state = DashboardState(path.parent, Path("tests/.no_runs"), generated_path)
            summaries = state.summaries()
            record = next(item for item in summaries["samples"] if item["source"].endswith(path.name))
            self.assertEqual(record["difficulty"], "hard")
            detail = state.sample_detail(record["id"])
            assert detail is not None
            self.assertEqual(len(detail["path"]), len(sample.path))
        finally:
            path.unlink(missing_ok=True)
            generated_path.unlink(missing_ok=True)

    def test_dashboard_filters_dynamic_obstacle_overlap(self) -> None:
        states6 = np.zeros((181, 6), dtype=float)
        states6[:, 0] = np.linspace(10.0, 90.0, len(states6))
        controls = np.zeros((180, 2), dtype=float)
        scenario = Scenario(
            start=states6[0],
            goal=np.array([90.0, 10.0]),
            static_obstacles=[],
            dynamic_obstacles=[
                Obstacle(
                    center=np.array([50.0, 22.5]),
                    radius=2.0,
                    velocity=np.array([0.0, -0.5]),
                    kind="dynamic",
                )
            ],
            name="dynamic_overlap",
        )
        sample = TrajectorySample(
            states=state6_to_state7(states6),
            controls=controls,
            env57=np.zeros(57),
            path=states6[:, :2],
            metadata={"success": True, "difficulty": "easy", "dt": 0.5},
            scenario=scenario,
        )
        path = Path("tests") / f".dashboard_collision_{uuid4().hex}.npz"
        generated_path = path.with_name(f"{path.stem}_generated.npz")
        try:
            save_dataset_npz([sample], path, {"seed": 11})
            state = DashboardState(path.parent, Path("tests/.no_runs"), generated_path)
            summaries = state.summaries()
            self.assertFalse(any(item["source"].endswith(path.name) for item in summaries["samples"]))
        finally:
            path.unlink(missing_ok=True)
            generated_path.unlink(missing_ok=True)

    def test_dashboard_balanced_generation_target_split(self) -> None:
        with self.assertRaises(ValueError):
            _balanced_targets(8999)
        self.assertEqual(_balanced_targets(9000), {"easy": 3000, "medium": 3000, "hard": 3000})
        self.assertEqual(_balanced_worker_count(3), 1)
        self.assertGreaterEqual(_balanced_worker_count(9000), 1)

    def test_dashboard_detail_rebuilds_trace_when_metadata_is_compact(self) -> None:
        sample = simulate_scenario(make_demo_scenario(), SimConfig(rng_seed=13), seed=13)
        sample.metadata["difficulty"] = "easy"
        sample.metadata.pop("trace", None)
        path = Path("tests") / f".dashboard_compact_{uuid4().hex}.npz"
        generated_path = path.with_name(f"{path.stem}_generated.npz")
        try:
            save_dataset_npz([sample], path, {"seed": 13})
            state = DashboardState(path.parent, Path("tests/.no_runs"), generated_path)
            summaries = state.summaries()
            record = next(item for item in summaries["samples"] if item["source"].endswith(path.name))
            detail = state.sample_detail(record["id"])
            assert detail is not None
            self.assertEqual(len(detail["trace"]), len(sample.states))
            self.assertIn("clearance", detail["trace"][0])
        finally:
            path.unlink(missing_ok=True)
            generated_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
