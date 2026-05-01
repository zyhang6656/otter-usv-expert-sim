from __future__ import annotations

import heapq
import math

import numpy as np

from .config import SimConfig
from .types import Obstacle, Scenario


GridNode = tuple[int, int]


def _world_to_grid(point: np.ndarray, resolution: float, n: int) -> GridNode:
    gx = int(round(float(point[0]) / resolution))
    gy = int(round(float(point[1]) / resolution))
    return (int(np.clip(gx, 0, n - 1)), int(np.clip(gy, 0, n - 1)))


def _grid_to_world(node: GridNode, resolution: float) -> np.ndarray:
    return np.array([node[0] * resolution, node[1] * resolution], dtype=float)


def _build_occupancy(scenario: Scenario, config: SimConfig) -> np.ndarray:
    n = int(round(config.workspace_size / config.grid_resolution)) + 1
    occ = np.zeros((n, n), dtype=bool)
    for ix in range(n):
        for iy in range(n):
            p = _grid_to_world((ix, iy), config.grid_resolution)
            for obs in scenario.static_obstacles:
                if np.linalg.norm(p - obs.center) <= obs.radius + config.grid_clearance:
                    occ[ix, iy] = True
                    break
    return occ


def _line_of_sight(a: GridNode, b: GridNode, occ: np.ndarray) -> bool:
    x0, y0 = a
    x1, y1 = b
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    n = max(dx, dy)
    if n == 0:
        return not occ[x0, y0]
    for i in range(n + 1):
        t = i / n
        x = int(round(x0 + (x1 - x0) * t))
        y = int(round(y0 + (y1 - y0) * t))
        if x < 0 or y < 0 or x >= occ.shape[0] or y >= occ.shape[1] or occ[x, y]:
            return False
    return True


def _neighbors(node: GridNode, n: int) -> list[GridNode]:
    out = []
    x, y = node
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx = x + dx
            ny = y + dy
            if 0 <= nx < n and 0 <= ny < n:
                out.append((nx, ny))
    return out


def theta_star_path(scenario: Scenario, config: SimConfig | None = None) -> np.ndarray:
    config = config or SimConfig()
    occ = _build_occupancy(scenario, config)
    n = occ.shape[0]
    start = _world_to_grid(scenario.start[:2], config.grid_resolution, n)
    goal = _world_to_grid(scenario.goal, config.grid_resolution, n)
    occ[start] = False
    occ[goal] = False

    def dist(a: GridNode, b: GridNode) -> float:
        return math.hypot(a[0] - b[0], a[1] - b[1])

    open_heap: list[tuple[float, GridNode]] = []
    heapq.heappush(open_heap, (dist(start, goal), start))
    g_score: dict[GridNode, float] = {start: 0.0}
    parent: dict[GridNode, GridNode] = {start: start}
    closed: set[GridNode] = set()

    while open_heap:
        _f, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            break
        closed.add(current)

        for nb in _neighbors(current, n):
            if occ[nb] or nb in closed:
                continue
            pcur = parent[current]
            if _line_of_sight(pcur, nb, occ):
                tentative = g_score[pcur] + dist(pcur, nb)
                tentative_parent = pcur
            else:
                tentative = g_score[current] + dist(current, nb)
                tentative_parent = current
            if tentative < g_score.get(nb, float("inf")):
                g_score[nb] = tentative
                parent[nb] = tentative_parent
                heapq.heappush(open_heap, (tentative + dist(nb, goal), nb))

    if goal not in parent:
        return np.vstack([scenario.start[:2], scenario.goal])

    nodes = [goal]
    while nodes[-1] != start:
        nodes.append(parent[nodes[-1]])
    nodes.reverse()
    path = np.vstack([_grid_to_world(node, config.grid_resolution) for node in nodes])
    path[0] = scenario.start[:2]
    path[-1] = scenario.goal
    return _shortcut_path(path, scenario.static_obstacles, config)


def _shortcut_path(path: np.ndarray, obstacles: list[Obstacle], config: SimConfig) -> np.ndarray:
    if len(path) <= 2:
        return path
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1:
            if _segment_clear(path[i], path[j], obstacles, config.grid_clearance):
                break
            j -= 1
        out.append(path[j])
        i = j
    return np.vstack(out)


def _segment_clear(a: np.ndarray, b: np.ndarray, obstacles: list[Obstacle], clearance: float) -> bool:
    seg = b - a
    seg_norm_sq = float(np.dot(seg, seg))
    for obs in obstacles:
        if seg_norm_sq <= 1e-9:
            closest = a
        else:
            alpha = float(np.clip(np.dot(obs.center - a, seg) / seg_norm_sq, 0.0, 1.0))
            closest = a + alpha * seg
        if np.linalg.norm(closest - obs.center) <= obs.radius + clearance:
            return False
    return True
