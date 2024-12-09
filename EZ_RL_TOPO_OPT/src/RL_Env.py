import design_space_funcs as dsf
from numpy.linalg import LinAlgError
import numpy as np
import gymnasium as gym
import constants as const
import random

import FEM as fem

def reward_function(design, initial_max_stress, current_max_stress, initial_max_strain, current_max_strain, initial_avg_stress, current_avg_stress, initial_avg_strain, current_avg_strain):
    # Calculate the ratio of initial to current number of elements
    initial_num_elements = np.size(design)
    current_num_elements = np.size(np.where(design != 0))

    element_ratio = (initial_num_elements / current_num_elements) ** 2
    w_max_stress = 4
    w_max_strain = 4
    # Calculate the ratios of initial to current stress and strain values
    stress_ratio = (initial_max_stress / current_max_stress) * w_max_stress + (initial_avg_stress / current_avg_stress)
    strain_ratio = (initial_max_strain / current_max_strain) * w_max_strain + (initial_avg_strain / current_avg_strain)

    # Combine the ratios and square the result
    reward = element_ratio * (stress_ratio + strain_ratio) ** 2

    return reward

def get_reward(grid, init_stress, init_strain, init_avg_stress, init_avg_strain):
    a,b,c,d = dsf.extract_fem_data(grid)
    max_stress, max_strain, avg_u1, avg_u2, element_count, average_stress, average_strain, max_displacement_1, max_displacement_2, avg_strain_over_nodes = fem.FEM(a, b, c, d, plot_flag = False)
    reward = reward_function(grid, init_stress, max_stress, init_strain, max_strain, init_avg_stress, average_stress, init_avg_strain, average_strain)
    return reward, max_stress, max_strain, average_stress, average_strain

def get_needed_fem_values(grid):
    a,b,c,d = dsf.extract_fem_data(grid)
    max_stress, max_strain, avg_u1, avg_u2, element_count, average_stress, average_strain, max_displacement_1, max_displacement_2, avg_strain_over_nodes = fem.FEM(a, b, c, d, plot_flag = False)
    return max_stress, max_strain, average_stress, average_strain

def get_observation_space_range(height,width):
        length = (4*height * width)+4
        dim = height * width
        design_space_low = np.zeros(length)
        design_space_high = np.zeros(length)
        for i in range(length):
            if i / dim < 1:
                design_space_low[i] = 0
                design_space_high[i] = 1
            elif i / dim < 2:
                design_space_low[i] = 0
                design_space_high[i] = 1
            elif i / dim < 3:
                design_space_low[i] = -100
                design_space_high[i] = 100
            elif i / dim < 4:
                design_space_low[i] = -100
                design_space_high[i] = 100
            else:
                design_space_low[i] = -1000
                design_space_high[i] = 1000
        shape = (length,)
        return design_space_low, design_space_high, shape

class TopOptEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, height, width, bounded=[(0, 0), (-1, 0)], loaded =[(-1, -1, "LY20")], mode="train", threshold=0.3, ):
        self.mode = mode
        self.threshold = threshold
        self.height = height
        self.width = width
        
        if self.mode=="train":
            self.loaded = self.get_random_loaded(self.height, self.width)
            self.bounded = self.get_random_bounded(self.height, self.width,self.loaded)
        else:
            self.bounded = bounded
            self.loaded = loaded

        self.grid = dsf.create_grid(self.height, self.width, self.bounded, self.loaded)
        self.init_max_stress, self.init_max_strain, self.init_avg_stress, self.init_avg_strain = get_needed_fem_values(self.grid)
        
        self.action_space = gym.spaces.Discrete(height * width)
        low, high, shape = get_observation_space_range(self.height, self.width)
        self.observation_space = gym.spaces.Box(low=low, high=high, shape=shape, dtype=np.float64)
        self.step_count = 0
        self.reward = 0

    def step(self, action):
        self.step_count += 1
        terminated = False
        truncated = False
        if self.is_illegal_action(action, verbose=self.is_eval()):
            self.reward = -10
            terminated = True
            
        else:
            self.grid = self.take_action(action)
            try:
                self.reward, max_stress, max_strain, avg_stress, avg_strain = get_reward(self.grid, self.init_max_stress, self.init_max_strain, self.init_avg_stress, self.init_avg_strain)
            except LinAlgError:
                self.reward = -10
                terminated = True
            if self.get_constraint(self.grid) < self.threshold:
                terminated = True
                truncated = True
                if self.mode == "train":
                    self.reward = self.reward*2
                self.obs = self.create_observation(self.grid, max_stress, max_strain, avg_stress, avg_strain)
        #print("Reward: ", self.reward)
        return self.obs, self.reward, terminated, truncated, self.get_info()
    
    def create_observation(self, grid, m_stress,m_strain,a_stress,a_strain):
        obs = np.zeros((4*self.height*self.width)+4)
        obs[0:self.height*self.width] = grid[0,:,:].flatten()
        obs[self.height*self.width:2*self.height*self.width] = grid[1,:,:].flatten()
        obs[2*self.height*self.width:3*self.height*self.width] = grid[2,:,:].flatten()
        obs[3*self.height*self.width:4*self.height*self.width] = grid[3,:,:].flatten()
        obs[4*self.height*self.width] = m_stress
        obs[(4*self.height*self.width)+1] = m_strain
        obs[(4*self.height*self.width)+2] = a_stress
        obs[(4*self.height*self.width)+3] = a_strain
        return obs

    def reset(self, seed=None):
        if self.mode == "train":
            self.loaded = self.get_random_loaded(self.height, self.width)
            self.bounded = self.get_random_bounded(self.height, self.width, self.loaded)
            self.grid = dsf.create_grid(self.height, self.width, self.bounded, self.loaded)
            self.init_max_stress, self.init_max_strain, self.init_avg_stress, self.init_avg_strain = get_needed_fem_values(self.grid)
        self.grid = dsf.create_grid(self.height, self.width, self.bounded, self.loaded)
        self.step_count = 0
        self.reward = 0
        self.obs = self.create_observation(self.grid, self.init_max_stress, self.init_max_strain, self.init_avg_stress, self.init_avg_strain)

        return self.obs, self.get_info()


    def is_train(self):
        return self.mode == "train"
    def is_eval(self):
        return self.mode == "eval"
    
    def are_all_bounded_nodes_connected_to_all_loaded_nodes(self, row, col):
        # Temporarily remove the node
        original_value = self.grid[0, row, col]
        self.grid[0, row, col] = 0

        # Perform DFS or BFS to check connectivity
        visited = np.zeros((self.height, self.width), dtype=bool)
        stack = []

        # Find all bounded nodes
        bounded_nodes = [(r, c) for r in range(self.height) for c in range(self.width) if self.grid[1, r, c] == 1]

        # Find all loaded nodes
        loaded_nodes = [(r, c) for r in range(self.height) for c in range(self.width) if self.grid[2, r, c] != 0 or self.grid[3, r, c] != 0]

        # Perform DFS from each bounded node
        for start_node in bounded_nodes:
            stack.append(start_node)
            while stack:
                r, c = stack.pop()
                if visited[r, c]:
                    continue
                visited[r, c] = True
                # Check if we reached all loaded nodes
                if all(visited[lr, lc] for lr, lc in loaded_nodes):
                    self.grid[0, row, col] = original_value  # Restore the node
                    return True
                # Add neighbors to the stack
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.height and 0 <= nc < self.width and not visited[nr, nc] and self.grid[0, nr, nc] != 0:
                        stack.append((nr, nc))

        self.grid[0, row, col] = original_value  # Restore the node
        return False

    def is_illegal_action(self, action, verbose=False):
        row, col = self.action_to_coordinates(action)
        if self.grid[0, row, col] == 0:
            if verbose:
                print("Tried removing material from an empty cell.")
            return True
        if self.grid[1, row, col] == 1:
            if verbose:
                print("Tried removing material from a bounded cell.")
            return True
        if self.grid[2, row, col] != 0 or self.grid[3, row, col] != 0:
            if verbose:
                print("Tried removing material from a loaded cell.")
            return True
        if not self.are_all_bounded_nodes_connected_to_all_loaded_nodes(row, col):
            if verbose:
                print("Action would disconnect bounded nodes from loaded nodes.")
            return True
        return False

    def get_constraint(self, grid):
        return grid[0, :, :].sum() / (self.height * self.width)
    
    def take_action(self, action):
        row, col = self.action_to_coordinates(action)
        return dsf.remove_material(self.grid, row, col)

    def action_to_coordinates(self, action):
        row, col = divmod(action, self.width)
        return row, col

    def get_random_loaded(self, height, width):
        loaded = []
        row = self.get_random_number(0, height - 1)
        col = self.get_random_number(0, width - 1)
        negative_bool = self.get_random_number(0, 1)
        x_or_y = self.get_random_number(0, 1)
        if negative_bool == 0:
            load_value = self.get_random_number(10, 100)
        else:
            load_value = -self.get_random_number(10, 100)
        if x_or_y == 0:
            loaded.append((row, col, "LX" + str(load_value)))
        else:
            loaded.append((row, col, "LY" + str(load_value)))
        return loaded
    
    def get_random_number(self, low, high):
        return random.randint(low, high)

    def get_random_edge_coordinate(self, height, width):
        edge = self.get_random_number(0, 3)
        if edge == 0:  # Top edge
            row = 0
            col = self.get_random_number(0, width - 1)
        elif edge == 1:  # Bottom edge
            row = height - 1
            col = self.get_random_number(0, width - 1)
        elif edge == 2:  # Left edge
            row = self.get_random_number(0, height - 1)
            col = 0
        else:  # Right edge
            row = self.get_random_number(0, height - 1)
            col = width - 1
        return row, col

    def get_random_bounded(self, height, width, loaded):
        bounded = []
        num_bounded = self.get_random_number(1, max(height, width))
        for i in range(num_bounded):
            while True:
                row, col = self.get_random_edge_coordinate(height, width)
                if (row, col) in bounded:
                    continue
                if any((row == lr and col == lc) or
                    (row == lr + dr and col == lc + dc)
                    for lr, lc, _ in loaded
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]):
                    continue
                bounded.append((row, col))
                break
        return bounded
    
    def get_info(self):
        return {"grid" : self.grid, 
                "reward" : self.reward, 
                "step" : self.step_count}

def make_env(height, width, bounded, loaded, mode="train", threshold=0.3):
    def _init():
        env = TopOptEnv(height, width, bounded, loaded, mode, threshold)
        return env
    return _init
