import math
from collections import deque

import numpy as np
import pygame

from src.env.config import *


class SumoRenderer:
    def __init__(
        self,
        screen,
        font_main,
        font_header,
        show_trails=True,
        show_sensors=True,
        show_ui=True,
    ):
        self.screen = screen
        self.font_main = font_main
        self.font_header = font_header
        self.offset_x = WIDTH // 2
        self.offset_y = HEIGHT // 2
        self.show_trails = show_trails
        self.show_sensors = show_sensors
        self.show_ui = show_ui
        self.trails = [deque(maxlen=500), deque(maxlen=500)]

    def _to_screen(self, x, y):
        return int(self.offset_x + x), int(self.offset_y - y)

    def draw_arena(self, radius):
        self.screen.fill((255, 255, 255))
        center = (self.offset_x, self.offset_y)
        pygame.draw.circle(self.screen, (255, 255, 255), center, radius)
        self._draw_grid(radius)
        pygame.draw.circle(self.screen, ARENA_COLOR, center, radius, 3)

    def _draw_grid(self, radius):
        grid_color = (235, 235, 235)
        step_px = int(ROBOT_SIDE / 2 * M_TO_PX)
        for x_val in range(-radius, radius, step_px):
            dx = abs(x_val)
            dy = math.sqrt(max(0, radius**2 - dx**2))
            p1 = self._to_screen(x_val, -dy)
            p2 = self._to_screen(x_val, dy)
            pygame.draw.line(self.screen, grid_color, p1, p2, 1)
        for y_val in range(-radius, radius, step_px):
            dy = abs(y_val)
            dx = math.sqrt(max(0, radius**2 - dy**2))
            p1 = self._to_screen(-dx, y_val)
            p2 = self._to_screen(dx, y_val)
            pygame.draw.line(self.screen, grid_color, p1, p2, 1)

    def draw_robot(self, robot, color, robot_idx):
        screen_pos = self._to_screen(robot.x, robot.y)
        self.trails[robot_idx].append(screen_pos)

        if self.show_trails and len(self.trails[robot_idx]) > 1:
            pygame.draw.lines(
                self.screen, color, False, list(self.trails[robot_idx]), 2
            )

        world_corners = robot.get_corners()
        screen_corners = [self._to_screen(c[0], c[1]) for c in world_corners]

        pygame.draw.polygon(self.screen, color, screen_corners)
        pygame.draw.polygon(self.screen, (0, 0, 0), screen_corners, 2)

        rad = math.radians(robot.angle)
        forward = np.array([math.cos(rad), math.sin(rad)])
        side = np.array([-math.sin(rad), math.cos(rad)])

        for m in [-1, 1]:
            wheel_center = np.array([robot.x, robot.y]) + side * (robot.width / 2 * m)
            s_world = wheel_center - forward * (robot.width * 0.2)
            e_world = wheel_center + forward * (robot.width * 0.2)
            pygame.draw.line(
                self.screen,
                (0, 0, 0),
                self._to_screen(s_world[0], s_world[1]),
                self._to_screen(e_world[0], e_world[1]),
                6,
            )

        front_mid_world = (world_corners[1] + world_corners[2]) / 2
        pygame.draw.line(
            self.screen,
            (220, 0, 0),
            screen_pos,
            self._to_screen(front_mid_world[0], front_mid_world[1]),
            3,
        )

    def draw_observations_visual(self, robots, observations):
        if not self.show_sensors or observations is None:
            return

        for i, (robot, obs) in enumerate(zip(robots, observations)):
            global_angle_rad = math.atan2(obs[3], obs[4])
            start_pos = self._to_screen(robot.x, robot.y)

            d_opp_px = obs[5] * (ARENA_RADIUS * 2)
            rel_angle_opp = math.atan2(obs[6], obs[7])
            total_angle_opp = global_angle_rad + rel_angle_opp

            target_opp_x = robot.x + math.cos(total_angle_opp) * d_opp_px
            target_opp_y = robot.y + math.sin(total_angle_opp) * d_opp_px
            pygame.draw.line(
                self.screen,
                (0, 200, 0),
                start_pos,
                self._to_screen(target_opp_x, target_opp_y),
                1,
            )

            d_edge_px = obs[8] * ARENA_RADIUS
            rel_angle_cntr = math.atan2(obs[9], obs[10])
            total_angle_cntr = global_angle_rad + rel_angle_cntr

            edge_x = robot.x - math.cos(total_angle_cntr) * d_edge_px
            edge_y = robot.y - math.sin(total_angle_cntr) * d_edge_px
            pygame.draw.line(
                self.screen, (255, 0, 0), start_pos, self._to_screen(edge_x, edge_y), 2
            )

    def draw_ui(self, robots, observations=None):
        if not self.show_ui:
            return

        ui_configs = [
            {"name": "ROBOT 1 (GREEN)", "color": (0, 150, 0), "x": 20},
            {"name": "ROBOT 2 (BLUE)", "color": (0, 0, 200), "x": WIDTH - 205},
        ]

        labels_info = [
            ("V", "fwd"),
            ("V", "side"),
            ("ω", ""),
            ("sin(θ)", ""),
            ("cos(θ)", ""),
            ("L", "opp"),
            ("sin(opp)", ""),
            ("cos(opp)", ""),
            ("L", "edge"),
            ("sin(cntr)", ""),
            ("cos(cntr)", ""),
        ]

        font_sub = pygame.font.SysFont("Consolas", 10, bold=True)

        for i, data in enumerate(ui_configs):
            header = self.font_header.render(data["name"], True, data["color"])
            self.screen.blit(header, (data["x"], 20))

            curr_y = 45
            if observations is not None:
                header_obs = self.font_main.render(
                    "OBSERVED STATE:", True, (100, 100, 100)
                )
                self.screen.blit(header_obs, (data["x"], curr_y))
                curr_y += 25

                for j, (main_txt, sub_txt) in enumerate(labels_info):
                    val = observations[i][j]
                    val_color = (
                        (0, 150, 0)
                        if val > 0.1
                        else (150, 0, 0) if val < -0.1 else (60, 60, 60)
                    )

                    main_surf = self.font_main.render(main_txt, True, (0, 0, 0))
                    self.screen.blit(main_surf, (data["x"], curr_y))
                    current_width = main_surf.get_width()

                    if sub_txt:
                        sub_surf = font_sub.render(sub_txt, True, (70, 70, 70))
                        self.screen.blit(
                            sub_surf, (data["x"] + current_width, curr_y + 5)
                        )
                        current_width += sub_surf.get_width()

                    colon_surf = self.font_main.render(":", True, (0, 0, 0))
                    self.screen.blit(colon_surf, (data["x"] + current_width, curr_y))

                    val_str = f"{val:6.2f}"
                    val_surf = self.font_main.render(val_str, True, val_color)
                    self.screen.blit(val_surf, (data["x"] + 80, curr_y))

                    curr_y += 18

    def clear_trails(self):
        self.trails[0].clear()
        self.trails[1].clear()
