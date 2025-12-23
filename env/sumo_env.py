import pygame
import math
import random
import numpy as np
from env.config import *

class SumoEnv:
    def __init__(self, render_mode=False):
        # Local refs to config values
        self.ROBOT_RADIUS = ROBOT_RADIUS
        self.MAX_SPEED = MAX_SPEED
        self.ARENA_RADIUS = ARENA_RADIUS
        self.center_x, self.center_y = WIDTH // 2, HEIGHT // 2
        self.render_mode = render_mode

        if self.render_mode:
            pygame.init()
            self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("2D Robot Sumo - RL Env")
            self.clock = pygame.time.Clock()

        self.reset()

    def create_robot(self, start_side):
        # Spawn robots at a distance facing the center
        dist = self.ARENA_RADIUS * 0.7
        x = self.center_x + start_side * dist + random.uniform(-20, 20)
        y = self.center_y + random.uniform(-20, 20)
        
        # -1 starts facing right, 1 starts facing left
        angle = 0 if start_side == -1 else 180
        return {'x': x, 'y': y, 'angle': angle, 'vx': 0, 'vy': 0}

    def reset(self):
        self.robot1 = self.create_robot(-1)
        self.robot2 = self.create_robot(1)
        self.robots = [self.robot1, self.robot2]
        self.done = False
        return self._get_obs()

    def _get_obs(self):
        r1, r2 = self.robot1, self.robot2
        
        # Relative position to opponent
        dx = r2['x'] - r1['x']
        dy = r2['y'] - r1['y']
        
        # Edge detection logic: how far to the rim and where is the center
        dist_center = math.hypot(r1['x'] - self.center_x, r1['y'] - self.center_y)
        edge_dist = self.ARENA_RADIUS - (dist_center + self.ROBOT_RADIUS)
        
        # Normalized vector pointing back home (to center)
        vec_to_center_x = (self.center_x - r1['x']) / (dist_center + 1e-5)
        vec_to_center_y = (self.center_y - r1['y']) / (dist_center + 1e-5)

        # Build the observation vector (normalized for the RL model)
        obs = [
            r1['vx'] / MAX_SPEED,
            r1['vy'] / MAX_SPEED,
            math.sin(math.radians(r1['angle'])),
            math.cos(math.radians(r1['angle'])),
            dx / (self.ARENA_RADIUS * 2),
            dy / (self.ARENA_RADIUS * 2),
            r2['vx'] / MAX_SPEED,
            r2['vy'] / MAX_SPEED,
            edge_dist / self.ARENA_RADIUS, # 1.0 at center, 0.0 at edge
            vec_to_center_x,
            vec_to_center_y,
            dist_center / self.ARENA_RADIUS
        ]
        return np.array(obs, dtype=np.float32)

    def step(self, action1, action2):
        actions = [action1, action2]
        for r, a in zip(self.robots, actions):
            move, turn = a

            # Forward/backward thrust handling
            if move == 1:
                r['vx'] += ACCELERATION * math.cos(math.radians(r['angle']))
                r['vy'] -= ACCELERATION * math.sin(math.radians(r['angle']))
            elif move == 2:
                r['vx'] -= (ACCELERATION / 2) * math.cos(math.radians(r['angle']))
                r['vy'] += (ACCELERATION / 2) * math.sin(math.radians(r['angle']))

            # Steering
            if turn == 1: r['angle'] += ROTATE_SPEED
            elif turn == 2: r['angle'] -= ROTATE_SPEED

        # Run physics update
        for r in self.robots:
            # Apply some random jitter (slippage)
            r['vx'] += random.uniform(-SLIP, SLIP)
            r['vy'] += random.uniform(-SLIP, SLIP)
            
            # Speed cap
            speed = math.hypot(r['vx'], r['vy'])
            if speed > MAX_SPEED:
                r['vx'] *= (MAX_SPEED / speed)
                r['vy'] *= (MAX_SPEED / speed)
                
            r['x'] += r['vx']
            r['y'] += r['vy']
            
            # Simple linear friction
            r['vx'] *= (1 - FRICTION)
            r['vy'] *= (1 - FRICTION)

        self._handle_collisions()
        return self._calculate_logic()

    def _handle_collisions(self):
        # Calculate distance between bots
        dx = self.robot2['x'] - self.robot1['x']
        dy = self.robot2['y'] - self.robot1['y']
        dist = math.hypot(dx, dy)
        min_dist = 2 * ROBOT_RADIUS
        
        # Check if they overlap
        if dist < min_dist and dist != 0:
            nx, ny = dx / dist, dy / dist
            overlap = min_dist - dist
            
            # Push bots apart and transfer momentum
            for i, r in enumerate(self.robots):
                sign = -1 if i == 0 else 1
                r['x'] += sign * nx * overlap / 2
                r['y'] += sign * ny * overlap / 2
                
                # Impact force
                r['vx'] += sign * nx * 0.8
                r['vy'] += sign * ny * 0.8

    def _calculate_logic(self):
        winner = 0
        # Boundary check - did anyone fall off?
        for idx, r in enumerate(self.robots):
            dist_to_center = math.hypot(r['x'] - self.center_x, r['y'] - self.center_y)
            if dist_to_center + ROBOT_RADIUS > self.ARENA_RADIUS:
                self.done = True
                winner = 2 if idx == 0 else 1
        
        # Note: reward calculation is handled externally in training loop
        return self._get_obs(), [0.0, 0.0], self.done, {'winner': winner}

    def render(self):
        if not self.render_mode: return
        
        self.clock.tick(FPS)
        self.screen.fill(BACKGROUND_COLOR)
        
        # Draw the ring (Dohyo)
        pygame.draw.circle(self.screen, ARENA_COLOR, (self.center_x, self.center_y), self.ARENA_RADIUS, 5)
        
        # Draw both bots
        for r, color in zip(self.robots, [ROBOT_COLOR_1, ROBOT_COLOR_2]):
            # Body
            pygame.draw.circle(self.screen, color, (int(r['x']), int(r['y'])), ROBOT_RADIUS)
            
            # Indicator line showing the front
            end_x = r['x'] + (ROBOT_RADIUS + 5) * math.cos(math.radians(r['angle']))
            end_y = r['y'] - (ROBOT_RADIUS + 5) * math.sin(math.radians(r['angle']))
            pygame.draw.line(self.screen, (255, 0, 0), (r['x'], r['y']), (end_x, end_y), 3)
            
        pygame.display.flip()