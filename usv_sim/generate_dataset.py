from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .config import SimConfig
from .io import save_dataset_npz
from .scenario import sample_scenario
from .simulator import simulate_scenario


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CyberShip II expert trajectory samples.")
    parser.add_argument("--num", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("data/expert_samples.npz"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-attempts", type=int, default=600)
    parser.add_argument("--max-saturation", type=float, default=0.9)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    config = SimConfig(rng_seed=args.seed)
    difficulties = ["easy", "medium", "hard"]
    probs = [0.3, 0.4, 0.3]
    samples = []
    failures: dict[str, int] = {"sample": 0, "not_reached": 0, "collided": 0, "saturated": 0}

    attempt = 0
    while len(samples) < args.num and attempt < args.max_attempts:
        attempt += 1
        difficulty = str(rng.choice(difficulties, p=probs))
        try:
            scenario = sample_scenario(rng, difficulty, config)
            sample = simulate_scenario(scenario, config, seed=args.seed + attempt)
        except Exception:
            failures["sample"] += 1
            continue
        if sample.metadata.get("success", False) and sample.metadata.get("control_saturation_ratio", 1.0) <= args.max_saturation:
            samples.append(sample)
            print(f"[{len(samples):04d}/{args.num:04d}] success attempt={attempt} difficulty={difficulty}")
        else:
            if sample.metadata.get("collided", False):
                failures["collided"] += 1
            elif not sample.metadata.get("reached", False):
                failures["not_reached"] += 1
            else:
                failures["saturated"] += 1

    save_dataset_npz(
        samples,
        args.out,
        {
            "requested": args.num,
            "attempts": attempt,
            "seed": args.seed,
            "failures": failures,
        },
    )
    print(f"saved={args.out} successes={len(samples)} attempts={attempt} failures={failures}")


if __name__ == "__main__":
    main()
