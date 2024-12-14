import numpy as np
import torch

HEIGHT = 3
WIDTH = 3

# Load conditions are the following:
# B_LEFT; B_MIDDLE; B_RIGHT; M_LEFT; M_MIDDLE; M RIGHT; T_LEFT; T_MIDDLE; T_RIGHT

LOAD_VAR = "B_RIGHT"

# Bounded conditions are the following:
# ALL_LEFT
# ALL_RIGHT
# ALL_TOP
# ALL_BOTTOM
# ALL_LEFT_RIGHT
# ALL_TOP_BOTTOM
# LT_LB (top and bottom left side)
# RT_RB (top and bottom right side)
# LT_RT (left and right side top)
# LB_RB (left and right side bottom)
# LT_RB (left top and right bottom)
# LB_RT (left bottom and right top)

BOUND_VAR = "ALL_LEFT"

# physical parameters
YOUNG_MODULUS = 68
"""The Value of the Young modulus of the used material"""

POISSON_RATIO = 0,32
"""The value of the Poisson ratio of the used material"""

YIELD_STRENGTH = 275
"""The specific yield strength of the used material in MPa"""

STRAIN_LIMIT = 0.002
"""The specific strain limit of the used material"""

NUMBER_SUBPROCESSES = 1
"""The number of subprocesses to be used in the multiprocessing 
environment."""

LOG_DIR = "log/"
"""The directory of where to save the best model."""

TS_BOARD_DIR = "ts_board/"
"""The directory of where to save the tensorboard logs."""

TIMESTEPS = 5e5
"""The number of timesteps to be used during the training of the model."""
#5e6

DESIGN = 0
"""The Constant Used to access the design dimension of the design space"""
BOUND = 1
"""The Constant used to access the Boundary layout of the design space"""
LOADED_X = 2
"""The Constant used to access the nodes which are loaded in the x direction"""
LOADED_Y = 3
"""The Constant used to access the a which are loaded in the y direction"""

BOUNDED_CHAR = "B"
E = 1.0
v = 0.33
C_real = E / (1.0 + v) / (1.0 - 2.0 * v) * np.array([[1.0 - v, v, 0.0], [v, 1.0 - v, 0.0], [0.0, 0.0, 0.5 - v]])

DUMMY_MATERIAL_E = 1e-6
DUMMY_MATERIAL_V = 1

C_dummy = DUMMY_MATERIAL_E / (1.0 + DUMMY_MATERIAL_V) / (1.0 - 2.0 * DUMMY_MATERIAL_V) * np.array([[1.0 - DUMMY_MATERIAL_V, DUMMY_MATERIAL_V, 0.0], [DUMMY_MATERIAL_V, 1.0 - DUMMY_MATERIAL_V, 0.0], [0.0, 0.0, 0.5 - DUMMY_MATERIAL_V]])

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#DEVICE = "cpu"

FIG_DIR ="figures/"