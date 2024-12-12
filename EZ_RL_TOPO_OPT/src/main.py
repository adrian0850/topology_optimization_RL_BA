"""
main.py

This module serves as the entry point for the reinforcement learning environment 
setup and testing. It imports necessary modules and libraries, sets up the 
environment, and defines functions for testingthe Finite Element Method (FEM) within the design space.

Modules and Libraries:
- design_space_funcs as dsf: Functions related to the design space.
- FEM as fem: Functions and classes for Finite Element Method simulations.
- RL_Env as rl: Custom reinforcement learning environment.
- feature_extractor as fe: Functions for feature extraction.
- stable_baselines3.common.env_checker: Utility to check the custom environment.
- stable_baselines3.common.vec_env: Utilities for vectorized environments.
- stable_baselines3: Reinforcement learning algorithms.
- torch: PyTorch library for deep learning.

Functions:
- fem_test(): Prompts the user for grid dimensions and sets up a basic FEM test 
with predefined boundary and load conditions.

Usage:
Run this module to set up and test the reinforcement learning environment and 
FEM simulations.
"""
import design_space_funcs as dsf
import FEM as fem
import RL_Env as rl
import feature_extractor as fe
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3 import PPO
import torch

num_envs = 10

def FEM_test():
    width = int(input("Enter grid width: "))
    height = int(input("Enter grid height: "))

    bounded = [(0, 0), (-1, 0)]
    loaded = [(-1, -1, "LY20")]

    # bounded_input = input("Enter bounded coordinates as (row,col) separated by spaces: ")
    # bounded = [tuple(map(int, coord.split(','))) for coord in bounded_input.split()]

    # loaded_input = input("Enter loaded coordinates as (row,col,val) separated by spaces: ")
    # loaded = [tuple(map(int, coord.split(','))) for coord in loaded_input.split()]

    grid = dsf.create_grid(height, width, bounded, loaded)
    print("Grid created:")
    print(grid)

    a,b,c,d = dsf.extract_fem_data(grid)
    print("nodes", a)
    print("element", b)
    print("bounded:", c)
    print("loaded:", d)
    fem.plot_mesh(a, b)

    # Call the FEM function
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    (init_max_stress, init_max_strain, _, _, _,
    init_average_stress, init_average_strain, _, _, _) = fem.FEM(
        a, b, c, d, plot_flag=True, grid=grid, device=device
    )

    bad = False
    good = False

    #bad = True
    good = True
    # Modify the grid
    if bad:
        dsf.remove_material(grid, -2, -1)
        print("Reward,\t", rl.get_reward(grid, init_max_stress, init_max_strain,
                                         init_average_stress, init_average_strain))
        dsf.remove_material(grid, -2, -2)
        print("Reward,\t", rl.get_reward(grid, init_max_stress, init_max_strain,
                                         init_average_stress, init_average_strain))
        dsf.remove_material(grid, -2, -3)
        print("Reward,\t", rl.get_reward(grid, init_max_stress, init_max_strain,
                                         init_average_stress, init_average_strain))
    elif good:
        dsf.remove_material(grid, 0, -1)
        print("Reward,\t", rl.get_reward(grid, init_max_stress, init_max_strain,
                                         init_average_stress, init_average_strain))
        dsf.remove_material(grid, 0, -2)
        print("Reward,\t", rl.get_reward(grid, init_max_stress, init_max_strain,
                                         init_average_stress, init_average_strain))
        dsf.remove_material(grid, 1, -1)
        print("Reward,\t", rl.get_reward(grid, init_max_stress, init_max_strain,
                                         init_average_stress, init_average_strain))
    else:
        grid[-0.5*height][-0.5*width] = 0

def Env_test():
    """
    Prompts the user to input grid dimensions, initializes a TopOptEnv environment,
    and checks the environment for compatibility with RL algorithms.
    The function performs the following steps:
    1. Prompts the user to enter the grid width and height.
    2. Initializes the environment with predefined boundary and load conditions.
    3. Checks the environment using the `check_env` function from the RL library.
    Parameters:
    None
    Returns:
    None
    """
    width = int(input("Enter grid width: "))
    height = int(input("Enter grid height: "))

    bounded = [(0, 0), (-1, 0)]
    loaded = [(-1, -0, "LY50")]

    env = rl.TopOptEnv(height, width, bounded, loaded)
    check_env(env, warn=True)

def reinforcement_learning_test():
    """
    Runs a reinforcement learning test for topology optimization using PPO (Proximal Policy Optimization).
    This function performs the following steps:
    1. Prompts the user to input the grid width and height.
    2. Defines boundary and load conditions for the environment.
    3. Creates a vectorized environment using SubprocVecEnv.
    4. Sets up a TensorBoard logger for monitoring training progress.
    5. Defines policy keyword arguments for the PPO model.
    6. Creates and trains the PPO model using the specified environment and policy.
    7. Saves the trained model.
    8. Loads the trained model and evaluates it in a new environment.
    9. Runs a loop to predict actions and step through the environment, printing the environment state.
    Note:
        - The environment and model are specific to topology optimization tasks.
        - The model is trained on a CUDA-enabled GPU if available, otherwise on the CPU.
        - The training process logs progress to TensorBoard.
    Inputs:
        - Grid width and height are provided by the user via input prompts.
    Outputs:
        - The trained PPO model is saved to a file named "ppo_topopt".
        - The environment state is printed during evaluation.
    Raises:
        - Any exceptions raised by the underlying libraries (e.g., gym, stable_baselines3, torch) during environment creation, model training, or evaluation.
    """
    width = int(input("Enter grid width: "))
    height = int(input("Enter grid height: "))

    bounded = [(0, 0), (-1, 0)]
    loaded = [(-1, -1, "LY50")]

    # Create the vectorized environment
    env = SubprocVecEnv([rl.make_env(height, width, bounded, loaded) for _ in range(NUM_ENVS)])
    env = VecMonitor(env)
    #env = rl.TopOptEnv(height, width, bounded, loaded)
    # Set up TensorBoard logger
    log_dir = "./tensorboard_logs/"

    policy_kwargs = dict(
        features_extractor_class=fe.CustomCombinedExtractor,
        features_extractor_kwargs={}
    )
    # Create the PPO model with the logger
    model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log=log_dir, policy_kwargs=policy_kwargs,device='cuda' if torch.cuda.is_available() else 'cpu')

    # Train the model
    model.learn(total_timesteps=5e5, progress_bar=True)

    model.save("ppo_topopt")

    env.close()
    env = rl.TopOptEnv(height, width, bounded, loaded, mode="eval")
    model = PPO.load("ppo_topopt", env = env)
    obs, _ = env.reset()
    for i in range(10):
        action, _states = model.predict(obs)
        action = action.item()

        obs, rewards, dones, _, info = env.step(action)
        env.print()

def load_test():
    width = int(input("Enter grid width: "))
    height = int(input("Enter grid height: "))

    bounded = [(0, 0), (-1, 0)]
    loaded = [(-1, -1, "LY50")]

    env = SubprocVecEnv([rl.make_env(height, width, bounded, loaded) for _ in range(num_envs)])

    env = VecMonitor(env)
    #env = rl.TopOptEnv(height, width, bounded, loaded)
    # Set up TensorBoard logger
    log_dir = "./tensorboard_logs/"

    policy_kwargs = dict(
        features_extractor_class=fe.CustomCombinedExtractor,
        features_extractor_kwargs={}
    )
    # Create the PPO model with the logger
    model = PPO.load("ppo_topopt", env = env)

    # Train the model
    model.learn(total_timesteps=1e6, progress_bar=True, tb_log_name="PPO_33", reset_num_timesteps=False)

    obs, _ = env.reset()
    for i in range(20):
        action, _states = model.predict(obs)
        action = action.item()

        obs, rewards, dones, _, info = env.step(action)
        env.print()


def main():
    #Env_test()
    #fem_test()
    reinforcement_learning_test()
    #load_test()

if __name__ == "__main__":
    main()
