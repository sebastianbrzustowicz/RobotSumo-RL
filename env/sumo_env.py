import pygame
import numpy as np
from env.config import *
from env.robot import SumoRobot
from env.collisions import check_sat_collision, get_robot_global_velocity
from env.renderer import SumoRenderer
import os

class SumoEnv:
    def __init__(self, render_mode=False, render_vectors=False):
        self.render_mode = render_mode
        self.render_vectors = render_vectors
        self.ARENA_RADIUS = ARENA_RADIUS
        self.center_x, self.center_y = WIDTH // 2, HEIGHT // 2
        
        self.screen = None
        self.renderer = None

        if self.render_mode:
            if 'SDL_VIDEODRIVER' not in os.environ and os.name == 'posix':
                os.environ['SDL_AUDIODRIVER'] = 'dummy'
            
            pygame.display.init()
            pygame.font.init()
            
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            
            # Przekazujemy flagi do renderera
            self.renderer = SumoRenderer(
                self.screen, 
                pygame.font.SysFont("Consolas", 14, bold=True),
                pygame.font.SysFont("Consolas", 16, bold=True),
                show_trails=self.render_vectors,
                show_sensors=self.render_vectors,
                show_ui=True                
            )
            self.clock = pygame.time.Clock()
        
        self.reset()

    def reset(self, randPositions=False):
        if self.renderer:
            self.renderer.clear_trails()
            
        r1_cfg, r2_cfg = self._generate_start_positions(randPositions)
        
        self.robot1 = SumoRobot(x=r1_cfg['x'], y=r1_cfg['y'], angle=r1_cfg['angle'])
        self.robot2 = SumoRobot(x=r2_cfg['x'], y=r2_cfg['y'], angle=r2_cfg['angle'])
        
        self.robots = [self.robot1, self.robot2]
        self.done = False
        return self._get_all_obs()

    def _generate_start_positions(self, randPositions):
        dist = self.ARENA_RADIUS * 0.7
        
        line_angle_deg = np.random.uniform(0, 360) if randPositions else 0.0
        rad = np.radians(line_angle_deg)
        off_x = dist * np.cos(rad)
        off_y = dist * np.sin(rad)
        
        r1_x = self.center_x - off_x
        r1_y = self.center_y - off_y
        
        r2_x = self.center_x + off_x
        r2_y = self.center_y + off_y
        
        r1_angle = (360 - line_angle_deg) % 360
        r2_angle = (360 - line_angle_deg + 180) % 360
        
        r1_cfg = {'x': r1_x, 'y': r1_y, 'angle': r1_angle}
        r2_cfg = {'x': r2_x, 'y': r2_y, 'angle': r2_angle}
        
        return r1_cfg, r2_cfg

    def step(self, action1, action2):
        for r, action in zip(self.robots, [action1, action2]):
            r.compute_dynamics(action[0], action[1])
            r.compute_kinematics()
        
        collision_occurred = self._handle_collisions()
        
        obs, rewards, done, info = self._calculate_env_logic()
        
        info['collision'] = collision_occurred
        return obs, rewards, done, info

    def _handle_collisions(self):
        overlap_info = check_sat_collision(self.robot1.get_corners(), self.robot2.get_corners())
        if overlap_info:
            overlap, axis = overlap_info
            dir_vec = np.array([self.robot2.x - self.robot1.x, self.robot2.y - self.robot1.y])
            if np.dot(dir_vec, axis) < 0: axis = -axis
            
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
                impulse_mag = (1 + restitution) * v_rel_normal / (1/m1 + 1/m2)
                
                impulse_vec = impulse_mag * axis
                
                self.robot1.apply_impulse(-impulse_vec)
                self.robot2.apply_impulse(impulse_vec)
                
                self.robot1.v_side *= 0.2
                self.robot2.v_side *= 0.2
                
                self.robot1.omega *= 0.3
                self.robot2.omega *= 0.3

            return True
        return False

    def _get_obs(self, viewer, target):
        # 1. Self motion and orientation (SI units)
        v_fwd = viewer.v
        v_side = viewer.v_side
        omega = viewer.omega
        global_angle_rad = math.radians(viewer.angle % 360)

        # 2. Relative position to opponent
        dx_opp = target.x - viewer.x
        dy_opp = target.y - viewer.y
        dist_opp = math.hypot(dx_opp, dy_opp)
        angle_to_opp = math.atan2(dy_opp, dx_opp) - global_angle_rad
        
        # 3. Relation to arena edge (closest safe direction)
        dx_center = self.center_x - viewer.x
        dy_center = self.center_y - viewer.y
        dist_to_center = math.hypot(dx_center, dy_center)
        
        # Distance to arena edge
        dist_to_edge = self.ARENA_RADIUS - dist_to_center
        angle_to_center = math.atan2(dy_center, dx_center) - global_angle_rad

        return np.array([
            v_fwd / MAX_SPEED,           # 1. Forward/backward velocity (local frame)
            v_side / MAX_SPEED,          # 2. Lateral velocity
            omega / ROTATE_SPEED,        # 3. Angular velocity
            math.sin(global_angle_rad),  # 4. Orientation (sin)
            math.cos(global_angle_rad),  # 5. Orientation (cos)
            dist_opp / (ARENA_RADIUS*2), # 6. Distance to opponent (normalized)
            math.sin(angle_to_opp),      # 7. Relative angle to opponent (sin)
            math.cos(angle_to_opp),      # 8. Relative angle to opponent (cos)
            dist_to_edge / ARENA_RADIUS, # 9. Distance to arena edge
            math.sin(angle_to_center),   # 10. Direction to arena center (sin)
            math.cos(angle_to_center)    # 11. Direction to arena center (cos)
        ], dtype=np.float32)

    def _get_all_obs(self):
        return [self._get_obs(self.robot1, self.robot2), 
                self._get_obs(self.robot2, self.robot1)]

    def _calculate_env_logic(self):
        winner = 0
        for idx, r in enumerate(self.robots):
            corners = r.get_corners()
            for corner in corners:
                dist = math.hypot(corner[0] - self.center_x, corner[1] - self.center_y)
                if dist > self.ARENA_RADIUS:
                    self.done = True
                    winner = 2 if idx == 0 else 1
                    break
            if self.done: break
                
        return self._get_all_obs(), [0.0, 0.0], self.done, {'winner': winner}

    def render(self):
        if not self.render_mode or self.renderer is None:
            return
            
        self.clock.tick(FPS)
        self.renderer.draw_arena(self.ARENA_RADIUS)
        
        obs = self._get_all_obs()
        
        if self.render_vectors:
            self.renderer.draw_observations_visual(self.robots, obs)
        
        self.renderer.draw_robot(self.robot1, ROBOT_COLOR_1, 0)
        self.renderer.draw_robot(self.robot2, ROBOT_COLOR_2, 1)
        
        self.renderer.draw_ui(self.robots, observations=obs)
        
        pygame.display.flip()