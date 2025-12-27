import pygame
import math
import numpy as np
from env.config import *
from collections import deque

class SumoRenderer:
    def __init__(self, screen, font_main, font_header, 
                 show_trails=True, 
                 show_sensors=True, 
                 show_ui=True):
        self.screen = screen
        self.font_main = font_main
        self.font_header = font_header
        self.center_x, self.center_y = WIDTH // 2, HEIGHT // 2
        
        self.show_trails = show_trails
        self.show_sensors = show_sensors
        self.show_ui = show_ui
        
        self.trails = [deque(maxlen=500), deque(maxlen=500)]

    def draw_arena(self, radius):
        self.screen.fill((255, 255, 255)) 
        pygame.draw.circle(self.screen, (255, 255, 255), (self.center_x, self.center_y), radius)
        self._draw_grid(radius)
        pygame.draw.circle(self.screen, ARENA_COLOR, (self.center_x, self.center_y), radius, 3)

    def _draw_grid(self, radius):
        grid_color = (235, 235, 235) 
        step_px = int(ROBOT_SIDE / 2 * M_TO_PX)
        for x in range(int(self.center_x - radius), int(self.center_x + radius), step_px):
            dx = abs(x - self.center_x)
            if dx < radius:
                dy = math.sqrt(radius**2 - dx**2)
                pygame.draw.line(self.screen, grid_color, (x, self.center_y - dy), (x, self.center_y + dy), 1)
        for y in range(int(self.center_y - radius), int(self.center_y + radius), step_px):
            dy = abs(y - self.center_y)
            if dy < radius:
                dx = math.sqrt(radius**2 - dy**2)
                pygame.draw.line(self.screen, grid_color, (self.center_x - dx, y), (self.center_x + dx, y), 1)

    def draw_robot(self, robot, color, robot_idx):
        self.trails[robot_idx].append((int(robot.x), int(robot.y)))
        if self.show_trails and len(self.trails[robot_idx]) > 1:
            pygame.draw.lines(self.screen, color, False, list(self.trails[robot_idx]), 2)

        corners = robot.get_corners()
        pygame.draw.polygon(self.screen, color, corners)
        pygame.draw.polygon(self.screen, (0, 0, 0), corners, 2)
        
        rad = math.radians(-robot.angle)
        forward = np.array([math.cos(rad), math.sin(rad)])
        side = np.array([math.cos(rad + math.pi/2), math.sin(rad + math.pi/2)])
        
        for m in [-1, 1]:
            wheel_center = np.array([robot.x, robot.y]) + side * (robot.width/2 * m)
            s = wheel_center - forward * (robot.width * 0.2)
            e = wheel_center + forward * (robot.width * 0.2)
            pygame.draw.line(self.screen, (0, 0, 0), s, e, 6)
        
        front_mid = (corners[1] + corners[2]) / 2
        pygame.draw.line(self.screen, (220, 0, 0), (robot.x, robot.y), front_mid, 3)

    def draw_observations_visual(self, robots, observations):
        if not self.show_sensors or observations is None:
            return

        for i, (robot, obs) in enumerate(zip(robots, observations)):
            global_angle_rad = math.atan2(obs[3], obs[4])
            
            # 1. Line to the opponent (obs[5, 6, 7])
            d_opp_px = obs[5] * (ARENA_RADIUS * 2)
            rel_angle_opp = math.atan2(obs[6], obs[7])
            total_angle_opp = global_angle_rad + rel_angle_opp
            
            target_opp_x = robot.x + math.cos(total_angle_opp) * d_opp_px
            target_opp_y = robot.y + math.sin(total_angle_opp) * d_opp_px
            pygame.draw.line(self.screen, (0, 200, 0), (robot.x, robot.y), (target_opp_x, target_opp_y), 1)

            # 2. Line to the edge (obs[8, 9, 10])
            d_edge_px = obs[8] * ARENA_RADIUS
            rel_angle_cntr = math.atan2(obs[9], obs[10])
            total_angle_cntr = global_angle_rad + rel_angle_cntr
            
            # Draw a vector to the edge (red)
            edge_x = robot.x - math.cos(total_angle_cntr) * d_edge_px
            edge_y = robot.y - math.sin(total_angle_cntr) * d_edge_px
            pygame.draw.line(self.screen, (255, 0, 0), (robot.x, robot.y), (edge_x, edge_y), 2)

    def draw_ui(self, robots, observations=None):
        if not self.show_ui:
            return

        ui_configs = [
            {"name": "ROBOT 1", "color": ROBOT_COLOR_1, "x": 20},
            {"name": "ROBOT 2", "color": ROBOT_COLOR_2, "x": WIDTH - 240}
        ]
        obs_labels = ["v_fwd", "v_side", "omega", "sin_ang", "cos_ang", "dist_opp", "sin_opp", "cos_opp", "dist_edge", "sin_cntr", "cos_cntr"]

        for i, data in enumerate(ui_configs):
            header = self.font_header.render(data["name"], True, data["color"])
            self.screen.blit(header, (data["x"], 20))
            
            curr_y = 45
            if observations is not None:
                header_obs = self.font_main.render("OBSERVED STATE:", True, (100, 100, 100))
                self.screen.blit(header_obs, (data["x"], curr_y))
                curr_y += 20
                for label, val in zip(obs_labels, observations[i]):
                    val_color = (0, 150, 0) if val > 0.1 else (150, 0, 0) if val < -0.1 else (60, 60, 60)
                    surf = self.font_main.render(f"{label:>9}: {val:6.2f}", True, val_color)
                    self.screen.blit(surf, (data["x"], curr_y))
                    curr_y += 15

    def clear_trails(self):
        self.trails[0].clear()
        self.trails[1].clear()