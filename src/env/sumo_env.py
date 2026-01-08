import math
import os

import numpy as np
import pygame

from src.env.collisions import check_sat_collision, get_robot_global_velocity
from src.env.config import *
from src.env.renderer import SumoRenderer
from src.env.robot import SumoRobot


class SumoEnv:
    def __init__(self, render_mode=False, render_vectors=False):
        self.render_mode = render_mode
        self.render_vectors = render_vectors
        self.ARENA_RADIUS = ARENA_RADIUS
        self.center_x, self.center_y = WIDTH // 2, HEIGHT // 2
        pygame.display.set_caption("RoboSumo-RL: Cross Play")

        self.screen = None
        self.renderer = None

        self.has_collision_occurred = False

        if self.render_mode:
            if "SDL_VIDEODRIVER" not in os.environ and os.name == "posix":
                os.environ["SDL_AUDIODRIVER"] = "dummy"

            pygame.display.init()
            pygame.font.init()

            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))

            self.renderer = SumoRenderer(
                self.screen,
                pygame.font.SysFont("Consolas", 14, bold=True),
                pygame.font.SysFont("Consolas", 16, bold=True),
                show_trails=self.render_vectors,
                show_sensors=self.render_vectors,
                show_ui=True,
            )
            self.clock = pygame.time.Clock()

        self.reset()

    def reset(self, randPositions=False):
        if self.renderer:
            self.renderer.clear_trails()

        r1_cfg, r2_cfg = self._generate_start_positions(randPositions)

        self.robot1 = SumoRobot(x=r1_cfg["x"], y=r1_cfg["y"], angle=r1_cfg["angle"])
        self.robot2 = SumoRobot(x=r2_cfg["x"], y=r2_cfg["y"], angle=r2_cfg["angle"])

        self.robots = [self.robot1, self.robot2]
        self.done = False

        self.has_collision_occurred = False
        self.last_action1 = np.zeros(2)
        self.last_action2 = np.zeros(2)

        return self._get_all_obs()

    def step(self, action1, action2):
        self.last_action1 = action1
        self.last_action2 = action2

        for r, action in zip(self.robots, [action1, action2]):
            r.compute_dynamics(action[0], action[1])
            r.compute_kinematics()

        is_collision = self._handle_collisions()

        if is_collision:
            self.has_collision_occurred = True

        obs, rewards, done, info = self._calculate_env_logic()

        info["is_collision"] = is_collision
        info["has_collision"] = self.has_collision_occurred

        return obs, rewards, done, info

    def _handle_collisions(self):
        overlap_info = check_sat_collision(
            self.robot1.get_corners(), self.robot2.get_corners()
        )
        if overlap_info:
            overlap, axis = overlap_info
            dir_vec = np.array(
                [self.robot2.x - self.robot1.x, self.robot2.y - self.robot1.y]
            )
            if np.dot(dir_vec, axis) < 0:
                axis = -axis

            m1, m2 = self.robot1.mass, self.robot2.mass
            total_mass = m1 + m2

            push = axis * (overlap + 0.1)
            self.robot1.x -= push[0] * (m2 / total_mass)
            self.robot1.y -= push[1] * (m2 / total_mass)
            self.robot2.x += push[0] * (m1 / total_mass)
            self.robot2.y += push[1] * (m1 / total_mass)

            gv1 = get_robot_global_velocity(self.robot1)
            gv2 = get_robot_global_velocity(self.robot2)
            v_rel_normal = np.dot(gv1 - gv2, axis)

            if v_rel_normal > 0:
                restitution = 0.05
                impulse_mag = (1 + restitution) * v_rel_normal / (1 / m1 + 1 / m2)
                impulse_vec = impulse_mag * axis

                self.robot1.apply_impulse(-impulse_vec)
                self.robot2.apply_impulse(impulse_vec)

                self.robot1.v_side *= 0.2
                self.robot2.v_side *= 0.2
                self.robot1.omega *= 0.3
                self.robot2.omega *= 0.3

            return True
        return False

    def _generate_start_positions(self, randPositions):
        dist = self.ARENA_RADIUS * 0.7
        line_angle_deg = np.random.uniform(0, 360) if randPositions else 0.0
        rad = np.radians(line_angle_deg)

        off_x = dist * np.cos(rad)
        off_y = dist * np.sin(rad)

        r1_x = -off_x
        r1_y = -off_y
        r2_x = off_x
        r2_y = off_y

        r1_angle = line_angle_deg
        r2_angle = (line_angle_deg + 180) % 360

        return {"x": r1_x, "y": r1_y, "angle": r1_angle}, {
            "x": r2_x,
            "y": r2_y,
            "angle": r2_angle,
        }

    def _get_obs(self, viewer, target):
        v_fwd = viewer.v
        v_side = viewer.v_side
        omega = viewer.omega
        global_angle_rad = math.radians(viewer.angle % 360)

        dx_opp = target.x - viewer.x
        dy_opp = target.y - viewer.y
        dist_opp = math.hypot(dx_opp, dy_opp)

        angle_to_opp_raw = math.atan2(dy_opp, dx_opp) - global_angle_rad
        angle_to_opp = (angle_to_opp_raw + math.pi) % (2 * math.pi) - math.pi

        dist_to_center = math.hypot(viewer.x, viewer.y)
        dist_to_edge = self.ARENA_RADIUS - dist_to_center

        angle_to_center_raw = math.atan2(-viewer.y, -viewer.x) - global_angle_rad
        angle_to_center = (angle_to_center_raw + math.pi) % (2 * math.pi) - math.pi

        return np.array(
            [
                v_fwd / MAX_SPEED,
                v_side / MAX_SPEED,
                omega / ROTATE_SPEED,
                math.sin(global_angle_rad),
                math.cos(global_angle_rad),
                dist_opp / (ARENA_RADIUS * 2),
                math.sin(angle_to_opp),
                math.cos(angle_to_opp),
                dist_to_edge / ARENA_RADIUS,
                math.sin(angle_to_center),
                math.cos(angle_to_center),
            ],
            dtype=np.float32,
        )

    def _get_all_obs(self):
        return [
            self._get_obs(self.robot1, self.robot2),
            self._get_obs(self.robot2, self.robot1),
        ]

    def _calculate_env_logic(self):
        winner = 0
        for idx, r in enumerate(self.robots):
            corners = r.get_corners()
            for corner in corners:
                dist = math.hypot(corner[0], corner[1])
                if dist > self.ARENA_RADIUS:
                    self.done = True
                    winner = 2 if idx == 0 else 1
                    break
            if self.done:
                break
        return self._get_all_obs(), [0.0, 0.0], self.done, {"winner": winner}

    def render(self, names=None, archs=None):
        if not self.render_mode or self.renderer is None:
            return
        self.clock.tick(FPS)
        self.renderer.draw_arena(self.ARENA_RADIUS)
        obs = self._get_all_obs()
        if self.render_vectors:
            self.renderer.draw_observations_visual(self.robots, obs)
        self.renderer.draw_robot(self.robot1, ROBOT_COLOR_1, 0)
        self.renderer.draw_robot(self.robot2, ROBOT_COLOR_2, 1)
        self.renderer.draw_ui(self.robots, observations=obs, names=names, archs=archs)
        actions = [self.last_action1, self.last_action2]
        self.renderer.draw_actions(actions)

        pygame.display.flip()
