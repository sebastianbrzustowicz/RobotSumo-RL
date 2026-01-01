import math

# Display
WIDTH, HEIGHT = 900, 700
FPS = 60
M_TO_PX = 387

# Arena
ARENA_DIAMETER_M = 1.54
ARENA_RADIUS = int((ARENA_DIAMETER_M / 2) * M_TO_PX)
BACKGROUND_COLOR = (255, 255, 255)
ARENA_COLOR = (0, 0, 0)

# Robot
ROBOT_SIDE = 0.2
ROBOT_SIZE_PX = int(ROBOT_SIDE * M_TO_PX)
ROBOT_COLOR_1 = (0, 255, 0)
ROBOT_COLOR_2 = (0, 0, 255)

WHEEL_DIAMETER = 0.025    
MOTOR_MAX_RPM = 1500.0      
WHEEL_V_MAX = 2 * math.pi * WHEEL_DIAMETER / 2 * MOTOR_MAX_RPM / 60 # ~1.9634 m/s

V_MAX_MPS = WHEEL_V_MAX
OMEGA_MAX_RADS = WHEEL_V_MAX /  (ROBOT_SIDE / 2) # 1.9634 / 0.1 = 19.634 rad/s

# Dynamics
A_MPS2 = 25.0
ALPHA_RADS2 = 250.0 # Angular acceleration (rad/s^2)

# Conversion to training units (per frame)
MAX_SPEED = (V_MAX_MPS * M_TO_PX) / FPS      # px/frame
ROTATE_SPEED = OMEGA_MAX_RADS / FPS          # rad/frame
ACCELERATION = (A_MPS2 * M_TO_PX) / (FPS**2) # px/frame^2
ACCEL_ANGULAR = ALPHA_RADS2 / (FPS**2)       # rad/frame^2

# Friction and other
FRICTION = 0.02
LATERAL_FRICTION = 0.10