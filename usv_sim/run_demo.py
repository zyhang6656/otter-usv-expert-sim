from __future__ import annotations

import argparse
from pathlib import Path

from .config import SimConfig
from .io import save_sample_json
from .scenario import make_demo_scenario_with_seed
from .simulator import simulate_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a fixed CyberShip II expert-data demo.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--view", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("runs/latest_demo.json"))
    parser.add_argument("--steps", type=int, default=181)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--min-distance", type=float, default=80.0)
    parser.add_argument("--max-distance", type=float, default=100.0)
    parser.add_argument("--static-obstacles", type=int, default=6)
    parser.add_argument("--dynamic-obstacles", type=int, default=3)
    parser.add_argument("--max-demo-attempts", type=int, default=20)
    args = parser.parse_args()

    config = SimConfig(rng_seed=args.seed, n_steps=args.steps, dt=args.dt)
    sample = None
    for attempt in range(args.max_demo_attempts):
        scenario = make_demo_scenario_with_seed(
            args.seed + attempt,
            config=config,
            min_distance=args.min_distance,
            max_distance=args.max_distance,
            static_count=args.static_obstacles,
            dynamic_count=args.dynamic_obstacles,
        )
        candidate = simulate_scenario(scenario, config, seed=args.seed + attempt)
        candidate.metadata["difficulty"] = "medium"
        if candidate.metadata.get("success", False) or sample is None:
            sample = candidate
        if candidate.metadata.get("success", False):
            sample.metadata["demo_attempt"] = attempt + 1
            break
    assert sample is not None
    save_sample_json(sample, args.out)
    meta = sample.metadata
    print(
        "otter_demo",
        f"success={meta['success']}",
        f"reached={meta['reached']}",
        f"collided={meta['collided']}",
        f"initial_distance={meta['initial_distance']:.3f}",
        f"final_distance={meta['final_distance']:.3f}",
        f"min_clearance={meta['min_clearance']:.3f}",
        f"static={meta['static_obstacle_count']}",
        f"dynamic={meta['dynamic_obstacle_count']}",
        f"saved={args.out}",
    )
    if args.view:
        from .viewer import play_sample

        play_sample(sample)


if __name__ == "__main__":
    main()
