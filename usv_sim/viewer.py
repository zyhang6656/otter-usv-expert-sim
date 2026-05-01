from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from .config import OtterConfig, SimConfig
from .io import load_sample_json
from .types import TrajectorySample


def _state7_to_state6(states7: np.ndarray) -> np.ndarray:
    out = np.empty((states7.shape[0], 6), dtype=float)
    out[:, 0:2] = states7[:, 0:2]
    out[:, 2] = np.arctan2(states7[:, 2], states7[:, 3])
    out[:, 3:6] = states7[:, 4:7]
    return out


class Viewer:
    def __init__(self, sample: TrajectorySample, width: int = 920, height: int = 920):
        try:
            import pygame
        except ModuleNotFoundError as exc:
            raise SystemExit("pygame is not installed. Run: python -m pip install -r requirements.txt") from exc

        self.pygame = pygame
        self.sample = sample
        self.states6 = _state7_to_state6(sample.states)
        self.controls = sample.controls
        self.config = SimConfig()
        self.ship = OtterConfig()
        self.width = width
        self.height = height
        self.padding = 56
        self.step = 0
        self.playing = True
        self.speed = 1

    def world_to_screen(self, p: np.ndarray | list[float] | tuple[float, float]) -> tuple[int, int]:
        x, y = float(p[0]), float(p[1])
        scale = (min(self.width, self.height) - 2 * self.padding) / self.config.workspace_size
        sx = int(self.padding + x * scale)
        sy = int(self.height - self.padding - y * scale)
        return sx, sy

    def radius_to_screen(self, r: float) -> int:
        scale = (min(self.width, self.height) - 2 * self.padding) / self.config.workspace_size
        return max(1, int(r * scale))

    def run(self) -> None:
        pygame = self.pygame
        pygame.init()
        screen = pygame.display.set_mode((self.width, self.height))
        clock = pygame.time.Clock()
        pygame.font.init()
        font = pygame.font.Font(None, 18)
        small_font = pygame.font.Font(None, 14)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.playing = not self.playing
                    elif event.key == pygame.K_r:
                        self.step = 0
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        self.speed = min(8, self.speed + 1)
                    elif event.key == pygame.K_MINUS:
                        self.speed = max(1, self.speed - 1)
                    elif event.key == pygame.K_RIGHT:
                        self.step = min(len(self.states6) - 1, self.step + 1)
                    elif event.key == pygame.K_LEFT:
                        self.step = max(0, self.step - 1)
                    elif event.key == pygame.K_ESCAPE:
                        running = False

            if self.playing:
                self.step = min(len(self.states6) - 1, self.step + self.speed)

            self.draw(screen, font, small_font)
            pygame.display.flip()
            clock.tick(12)
        pygame.quit()

    def draw(self, screen, font, small_font) -> None:
        pygame = self.pygame
        screen.fill((247, 249, 250))
        self._draw_grid(screen)
        self._draw_path(screen)
        self._draw_obstacles(screen)
        self._draw_start(screen)
        self._draw_goal(screen)
        self._draw_trace(screen)
        self._draw_vessel(screen)
        self._draw_hud(screen, font, small_font)

    def _draw_grid(self, screen) -> None:
        pygame = self.pygame
        rect = pygame.Rect(self.padding, self.padding, self.width - 2 * self.padding, self.height - 2 * self.padding)
        pygame.draw.rect(screen, (33, 47, 60), rect, 2)
        for v in range(0, 101, 10):
            a = self.world_to_screen((v, 0))
            b = self.world_to_screen((v, 100))
            c = self.world_to_screen((0, v))
            d = self.world_to_screen((100, v))
            pygame.draw.line(screen, (225, 231, 235), a, b, 1)
            pygame.draw.line(screen, (225, 231, 235), c, d, 1)

    def _draw_path(self, screen) -> None:
        pygame = self.pygame
        if len(self.sample.path) >= 2:
            pts = [self.world_to_screen(p) for p in self.sample.path]
            pygame.draw.lines(screen, (74, 111, 165), False, pts, 2)
            for p in pts:
                pygame.draw.circle(screen, (74, 111, 165), p, 4)

    def _draw_obstacles(self, screen) -> None:
        pygame = self.pygame
        t = self.step * self.config.dt
        for obs in self.sample.scenario.static_obstacles:
            center = self.world_to_screen(obs.position_at(t))
            pygame.draw.circle(screen, (98, 110, 122), center, self.radius_to_screen(obs.radius))
            pygame.draw.circle(
                screen,
                (169, 177, 184),
                center,
                self.radius_to_screen(obs.radius + self.config.safety_margin),
                1,
            )
        for obs in self.sample.scenario.dynamic_obstacles:
            center = self.world_to_screen(obs.position_at(t))
            pygame.draw.circle(screen, (218, 117, 44), center, self.radius_to_screen(obs.radius))
            pygame.draw.circle(
                screen,
                (234, 164, 102),
                center,
                self.radius_to_screen(obs.radius + self.config.safety_margin),
                1,
            )
            future = self.world_to_screen(obs.position_at(t + 6.0))
            pygame.draw.line(screen, (218, 117, 44), center, future, 2)

    def _draw_goal(self, screen) -> None:
        pygame = self.pygame
        g = self.world_to_screen(self.sample.scenario.goal)
        pygame.draw.circle(screen, (45, 156, 104), g, 10, 3)
        pygame.draw.line(screen, (45, 156, 104), (g[0] - 12, g[1]), (g[0] + 12, g[1]), 2)
        pygame.draw.line(screen, (45, 156, 104), (g[0], g[1] - 12), (g[0], g[1] + 12), 2)

    def _draw_start(self, screen) -> None:
        pygame = self.pygame
        s = self.world_to_screen(self.sample.scenario.start[:2])
        pygame.draw.circle(screen, (54, 95, 145), s, 8, 2)
        pygame.draw.line(screen, (54, 95, 145), (s[0] - 8, s[1] + 8), (s[0] + 8, s[1] - 8), 2)
        pygame.draw.line(screen, (54, 95, 145), (s[0] - 8, s[1] - 8), (s[0] + 8, s[1] + 8), 2)

    def _draw_trace(self, screen) -> None:
        pygame = self.pygame
        pts = [self.world_to_screen(p) for p in self.states6[: self.step + 1, :2]]
        if len(pts) >= 2:
            pygame.draw.lines(screen, (28, 132, 198), False, pts, 3)

    def _draw_vessel(self, screen) -> None:
        pygame = self.pygame
        state = self.states6[self.step]
        x, y, psi = state[:3]
        visual_length = 5.2
        visual_width = visual_length * self.ship.breadth / self.ship.length
        rot = np.array([[math.cos(psi), -math.sin(psi)], [math.sin(psi), math.cos(psi)]])
        origin = np.array([x, y])

        def transform(points: np.ndarray) -> list[tuple[int, int]]:
            world = points @ rot.T + origin
            return [self.world_to_screen(p) for p in world]

        pontoon_len = visual_length
        pontoon_width = 0.28 * visual_width
        y_offset = 0.36 * visual_width
        for side in (-1.0, 1.0):
            local = np.array(
                [
                    [0.50 * pontoon_len, side * y_offset],
                    [0.34 * pontoon_len, side * (y_offset + 0.5 * pontoon_width)],
                    [-0.44 * pontoon_len, side * (y_offset + 0.5 * pontoon_width)],
                    [-0.52 * pontoon_len, side * y_offset],
                    [-0.44 * pontoon_len, side * (y_offset - 0.5 * pontoon_width)],
                    [0.34 * pontoon_len, side * (y_offset - 0.5 * pontoon_width)],
                ]
            )
            pts = transform(local)
            pygame.draw.polygon(screen, (23, 100, 138), pts)
            pygame.draw.polygon(screen, (8, 43, 61), pts, 2)

        deck = np.array(
            [
                [0.24 * visual_length, 0.18 * visual_width],
                [-0.24 * visual_length, 0.18 * visual_width],
                [-0.24 * visual_length, -0.18 * visual_width],
                [0.24 * visual_length, -0.18 * visual_width],
            ]
        )
        cabin = np.array(
            [
                [0.16 * visual_length, 0.12 * visual_width],
                [-0.03 * visual_length, 0.12 * visual_width],
                [-0.03 * visual_length, -0.12 * visual_width],
                [0.16 * visual_length, -0.12 * visual_width],
            ]
        )
        bow = np.array([[0.62 * visual_length, 0.0], [0.38 * visual_length, 0.16 * visual_width], [0.38 * visual_length, -0.16 * visual_width]])
        pygame.draw.polygon(screen, (184, 199, 207), transform(deck))
        pygame.draw.polygon(screen, (8, 43, 61), transform(deck), 2)
        pygame.draw.polygon(screen, (235, 241, 243), transform(cabin))
        pygame.draw.polygon(screen, (8, 43, 61), transform(cabin), 2)
        pygame.draw.polygon(screen, (31, 122, 163), transform(bow))

    def _draw_hud(self, screen, font, small_font) -> None:
        pygame = self.pygame
        state = self.states6[self.step]
        control = self.controls[min(self.step, len(self.controls) - 1)]
        trace = self.sample.metadata.get("trace", [])
        clearance = trace[min(self.step, len(trace) - 1)].get("clearance", float("nan")) if trace else float("nan")
        dist_goal = trace[min(self.step, len(trace) - 1)].get("dist_goal", float("nan")) if trace else float("nan")
        speed = float(np.hypot(state[3], state[4]))
        title = (
            f"t={self.step * self.config.dt:4.1f}s  step={self.step:02d}/{len(self.states6)-1}  "
            f"U={speed:.2f} u={state[3]:.2f} v={state[4]:.2f} r={state[5]:.2f}  "
            f"n=[{control[0]:.1f},{control[1]:.1f}]  clearance={clearance:.2f}m"
        )
        pygame.display.set_caption(title)
        lines = [
            title,
            "Space pause/play | R reset | +/- speed | arrow keys step | Esc quit",
            (
                f"success={self.sample.metadata.get('success')}  "
                f"initial={self.sample.metadata.get('initial_distance', 0):.1f}m  "
                f"goal_now={dist_goal:.1f}m  final={self.sample.metadata.get('final_distance', 0):.2f}m  "
                f"obstacles={self.sample.metadata.get('static_obstacle_count', 0)}+"
                f"{self.sample.metadata.get('dynamic_obstacle_count', 0)}"
            ),
        ]
        for i, line in enumerate(lines):
            surface = (font if i == 0 else small_font).render(line, True, (33, 47, 60))
            screen.blit(surface, (self.padding, 16 + i * 21))


def play_sample(sample: TrajectorySample) -> None:
    Viewer(sample).run()


def main() -> None:
    parser = argparse.ArgumentParser(description="View a saved USV simulation JSON.")
    parser.add_argument("--input", required=True, type=Path)
    args = parser.parse_args()
    play_sample(load_sample_json(args.input))


if __name__ == "__main__":
    main()
