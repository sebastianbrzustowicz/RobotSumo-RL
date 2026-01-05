import math

import numpy as np


def check_sat_collision(corners1, corners2):
    def get_axes(corners):
        axes = []
        for i in range(len(corners)):
            edge = corners[(i + 1) % len(corners)] - corners[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 0:
                axes.append(normal / norm)
        return axes

    def project(corners, axis):
        dots = [np.dot(c, axis) for c in corners]
        return min(dots), max(dots)

    axes = get_axes(corners1) + get_axes(corners2)
    min_overlap = float("inf")
    best_axis = None

    for axis in axes:
        minA, maxA = project(corners1, axis)
        minB, maxB = project(corners2, axis)
        overlap = min(maxA, maxB) - max(minA, minB)

        if overlap <= 0:
            return None

        if overlap < min_overlap:
            min_overlap, best_axis = overlap, axis

    return min_overlap, best_axis


def get_robot_global_velocity(robot):
    rad = math.radians(-robot.angle)
    forward_vec = np.array([math.cos(rad), math.sin(rad)])
    side_vec = np.array([math.cos(rad + math.pi / 2), math.sin(rad + math.pi / 2)])
    return robot.v * forward_vec + robot.v_side * side_vec
