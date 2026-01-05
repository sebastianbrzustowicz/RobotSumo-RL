import math
import numpy as np
from src.env.config import *

class SumoRobot:
    def __init__(self, x, y, angle, mass=1.0):
        self.x = x
        self.y = y
        self.angle = angle 
        
        self.v = 0.0          
        self.v_side = 0.0     
        self.omega = 0.0      
        
        self.mass = mass
        self.width = ROBOT_SIZE_PX
        
        self.accel = ACCELERATION / mass
        self.accel_angular = ACCEL_ANGULAR / mass
        self.friction = FRICTION
        self.lateral_friction = LATERAL_FRICTION

    def compute_dynamics(self, v_target_raw, omega_target_raw):
        """Raw control inputs are expected in range [-1, 1]."""
        v_target = v_target_raw * MAX_SPEED
        omega_target = omega_target_raw * ROTATE_SPEED

        if self.v < v_target:
            self.v = min(self.v + self.accel, v_target)
        elif self.v > v_target:
            self.v = max(self.v - self.accel, v_target)
            
        if self.omega < omega_target:
            self.omega = min(self.omega + self.accel_angular, omega_target)
        elif self.omega > omega_target:
            self.omega = max(self.omega - self.accel_angular, omega_target)

        if abs(v_target_raw) < 0.05:
            self.v *= (1.0 - self.friction)
            
        self.v_side *= (1.0 - LATERAL_FRICTION)

    def compute_kinematics(self, dt=1.0):
        self.angle += math.degrees(self.omega * dt)
        
        rad = math.radians(self.angle)
        
        forward_vec = np.array([math.cos(rad), math.sin(rad)])
        side_vec = np.array([-math.sin(rad), math.cos(rad)])
        
        self.x += (self.v * forward_vec[0] + self.v_side * side_vec[0]) * dt
        self.y += (self.v * forward_vec[1] + self.v_side * side_vec[1]) * dt

    def apply_impulse(self, impulse_vec):
        """Convert collision impulse into local velocity changes."""
        dv_x = impulse_vec[0] / self.mass
        dv_y = impulse_vec[1] / self.mass
        
        rad = math.radians(self.angle)
        forward_vec = np.array([math.cos(rad), math.sin(rad)])
        side_vec = np.array([-math.sin(rad), math.cos(rad)])
        
        self.v += np.dot(np.array([dv_x, dv_y]), forward_vec)
        self.v_side += np.dot(np.array([dv_x, dv_y]), side_vec)

    def get_corners(self):
        """Return square vertices for SAT collision detection."""
        corners = []
        half = self.width / 2
        rad = math.radians(self.angle)
        for dx, dy in [(-half, -half), (half, -half), (half, half), (-half, half)]:
            rx = self.x + dx * math.cos(rad) - dy * math.sin(rad)
            ry = self.y + dx * math.sin(rad) + dy * math.cos(rad)
            corners.append(np.array([rx, ry]))
        return corners