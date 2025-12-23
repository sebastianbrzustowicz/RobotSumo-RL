# General display and timing settings
WIDTH, HEIGHT = 600, 600
FPS = 60

# Physical arena scale (conversion ratio and dimensions)
CM_TO_PX = 3.87
ARENA_DIAMETER_CM = 154
# Calculated radius for the circle
ARENA_RADIUS = int((ARENA_DIAMETER_CM / 2) * CM_TO_PX)

# Visuals for the bots and play area
ROBOT_RADIUS = 15
ROBOT_COLOR_1 = (0, 255, 0)
ROBOT_COLOR_2 = (0, 0, 255)
BACKGROUND_COLOR = (50, 50, 50)
ARENA_COLOR = (200, 200, 200)

# Movement and physics constants
ROTATE_SPEED = 4
MAX_SPEED = 6
ACCELERATION = 0.3
FRICTION = 0.05
SLIP = 0.01