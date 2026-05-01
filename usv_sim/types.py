from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass
class Obstacle:
    center: Array
    radius: float
    velocity: Array = field(default_factory=lambda: np.zeros(2, dtype=float))
    active: bool = True
    kind: str = "static"

    def position_at(self, t: float) -> Array:
        if not self.active:
            return np.asarray(self.center, dtype=float)
        return np.asarray(self.center, dtype=float) + np.asarray(self.velocity, dtype=float) * t

    def to_json(self) -> dict[str, Any]:
        return {
            "center": np.asarray(self.center, dtype=float).tolist(),
            "radius": float(self.radius),
            "velocity": np.asarray(self.velocity, dtype=float).tolist(),
            "active": bool(self.active),
            "kind": self.kind,
        }

    @staticmethod
    def from_json(data: dict[str, Any]) -> "Obstacle":
        return Obstacle(
            center=np.asarray(data["center"], dtype=float),
            radius=float(data["radius"]),
            velocity=np.asarray(data.get("velocity", [0.0, 0.0]), dtype=float),
            active=bool(data.get("active", True)),
            kind=str(data.get("kind", "static")),
        )


@dataclass
class Scenario:
    start: Array
    goal: Array
    static_obstacles: list[Obstacle]
    dynamic_obstacles: list[Obstacle]
    workspace: tuple[float, float, float, float] = (0.0, 100.0, 0.0, 100.0)
    name: str = "scenario"

    def all_obstacles(self) -> list[Obstacle]:
        return [*self.static_obstacles, *self.dynamic_obstacles]

    def to_json(self) -> dict[str, Any]:
        return {
            "start": np.asarray(self.start, dtype=float).tolist(),
            "goal": np.asarray(self.goal, dtype=float).tolist(),
            "workspace": list(self.workspace),
            "name": self.name,
            "static_obstacles": [obs.to_json() for obs in self.static_obstacles],
            "dynamic_obstacles": [obs.to_json() for obs in self.dynamic_obstacles],
        }

    @staticmethod
    def from_json(data: dict[str, Any]) -> "Scenario":
        return Scenario(
            start=np.asarray(data["start"], dtype=float),
            goal=np.asarray(data["goal"], dtype=float),
            workspace=tuple(float(x) for x in data.get("workspace", [0.0, 100.0, 0.0, 100.0])),
            name=str(data.get("name", "scenario")),
            static_obstacles=[Obstacle.from_json(x) for x in data.get("static_obstacles", [])],
            dynamic_obstacles=[Obstacle.from_json(x) for x in data.get("dynamic_obstacles", [])],
        )


@dataclass
class TrajectorySample:
    states: Array
    controls: Array
    env57: Array
    path: Array
    metadata: dict[str, Any]
    scenario: Scenario

    def to_json(self) -> dict[str, Any]:
        return {
            "states": np.asarray(self.states, dtype=float).tolist(),
            "controls": np.asarray(self.controls, dtype=float).tolist(),
            "env57": np.asarray(self.env57, dtype=float).tolist(),
            "path": np.asarray(self.path, dtype=float).tolist(),
            "metadata": self.metadata,
            "scenario": self.scenario.to_json(),
        }

    @staticmethod
    def from_json(data: dict[str, Any]) -> "TrajectorySample":
        return TrajectorySample(
            states=np.asarray(data["states"], dtype=float),
            controls=np.asarray(data["controls"], dtype=float),
            env57=np.asarray(data["env57"], dtype=float),
            path=np.asarray(data.get("path", []), dtype=float),
            metadata=dict(data.get("metadata", {})),
            scenario=Scenario.from_json(data["scenario"]),
        )


def wrap_angle(angle: Array | float) -> Array | float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def state6_to_state7(states: Array) -> Array:
    states = np.asarray(states, dtype=float)
    out = np.empty((*states.shape[:-1], 7), dtype=float)
    out[..., 0:2] = states[..., 0:2]
    out[..., 2] = np.sin(states[..., 2])
    out[..., 3] = np.cos(states[..., 2])
    out[..., 4:7] = states[..., 3:6]
    return out
