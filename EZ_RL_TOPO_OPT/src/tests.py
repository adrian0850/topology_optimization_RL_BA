import random

from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import torch
from torchsummary import summary
import numpy as np
from torch.profiler import profile, record_function, ProfilerActivity

import design_space_funcs as dsf
import FEM as fem
import RL_Env as rl
import feature_extractor as fe
import constants as const


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def fem_analysis_func(strat):
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
    # print("bounded:", c)
    # print("loaded:", d)
    fem.plot_mesh(a, b)

    # Call the FEM function
    init_vm_stresses, init_avg_strain = fem.FEM(a, b, c, d, plot_flag=False,
                                                grid=grid, device=const.DEVICE)

    grid =dsf.convert_grid_with_von_mises(grid, init_vm_stresses)

    print(np.round(grid, 3))
    print("np.size(grid)", np.size(grid))
    bad = False
    good = False

    #bad = True
    good = True
    # Modify the grid

    if strat == "bad":
        dsf.remove_material(grid, -2, -1)
        print("Reward,\t", rl.get_reward(grid, init_avg_strain)[0])
        dsf.remove_material(grid, -2, -2)
        print("Reward,\t", rl.get_reward(grid, init_avg_strain)[0])
        dsf.remove_material(grid, -2, -3)
        print("Reward,\t", rl.get_reward(grid, init_avg_strain)[0])
    elif strat == "good":
        dsf.remove_material(grid, 0, -1)
        print("Reward,\t", rl.get_reward(grid, init_avg_strain)[0])
        dsf.remove_material(grid, 0, -2)
        print("Reward,\t", rl.get_reward(grid, init_avg_strain)[0])
        dsf.remove_material(grid, 1, -1)
        print("Reward,\t", rl.get_reward(grid, init_avg_strain)[0])
    else:
        grid[-0.5*height][-0.5*width] = 0

def env_compatability():
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

def learn(num_envs, num_timesteps=5e5):
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
    env = SubprocVecEnv([rl.make_env(height, width, bounded, loaded) for _ in range(num_envs)])
    env = VecMonitor(env)
    #env = rl.TopOptEnv(height, width, bounded, loaded)
    # Set up TensorBoard logger
    log_dir = "./tensorboard_logs/"

    policy_kwargs=dict(
        features_extractor_class=CommonCNN,
        net_arch=[dict(pi=[64, 64], vf=[64, 64])]
    )
    # Create the PPO model with the logger
    model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir, policy_kwargs=policy_kwargs,device=const.DEVICE)

    # Train the model
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
    #     with record_function("model_training"):
    #         # Train the model
    #         model.learn(total_timesteps=num_timesteps, progress_bar=True)
    model.learn(total_timesteps=num_timesteps, progress_bar=True)
    model.save("ppo_topopt")
    # prof.step()

    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))


    # Save the profiling results to TensorBoard
    prof.export_stacks("profiler_stacks.txt", "self_cuda_time_total")
    prof.export_chrome_trace(log_dir + "trace.json")



    env.close()
    env = rl.TopOptEnv(height, width, bounded, loaded, mode="eval")
    model = PPO.load("ppo_topopt", env = env)
    obs, _ = env.reset()
    for i in range(10):
        action, _states = model.predict(obs)
        action = action.item()

        obs, rewards, dones, _, info = env.step(action)
        env.print()

def loading(num_envs):
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

def cnn():
    set_seed(42)
    # Get grid dimensions from user input
    width = int(input("Enter grid width: "))
    height = int(input("Enter grid height: "))

    # Define bounded and loaded points
    bounded = [(0, 0), (-1, 0)]
    loaded = [(-1, -1, "LY50")]

    # Instantiate your actual topology optimization environment
    env = rl.make_env(height, width, bounded, loaded, mode="eval")()

    # Instantiate the feature extractor
    feature_extractor = fe.CustomCombinedExtractor(env.observation_space)

    device = const.DEVICE


    print("CNN Structure:")
    for key, extractor in feature_extractor.extractors.items():
        print(f"Extractor for key: {key}")
        extractor.to(device)
        summary(extractor, input_size=env.observation_space[key].shape, device=device)

    # Get a sample observation
    sample_observation, _ = env.reset()
    print(env.grid)
    #Convert the sample observation to a PyTorch tensor
    sample_observation_tensor = {
        key: torch.tensor(value).unsqueeze(0) for key, value in sample_observation.items()
    }
    # print(sample_observation_tensor)

    # Pass the sample observation through the feature extractor
    features = feature_extractor(sample_observation_tensor)
    feature_extractor.to(device)

    # Print the extracted features
    print("Extracted Features Shape:", features.shape)
    print("Extracted Features:", features)

    feature_extractor.visualize_feature_maps()

    # Check value ranges
    min_val = features.min().item()
    max_val = features.max().item()
    print("Min value in features:", min_val)
    print("Max value in features:", max_val)

    # Plot histogram of feature values
    features_np = features.detach().cpu().numpy().flatten()
    plt.hist(features_np, bins=50, color='blue', alpha=0.7)
    plt.title("Distribution of Extracted Features")
    plt.xlabel("Feature Value")
    plt.ylabel("Frequency")
    plt.show()

    # Visualize features as an image
    # Reshape the features to a 2D array for visualization
    features_np = features.detach().cpu().numpy().reshape(1, -1)
    plt.imshow(features_np, aspect='auto', cmap='viridis')
    plt.colorbar()
    plt.title("Extracted Features Heatmap")
    plt.xlabel("Feature Index")
    plt.ylabel("Batch Index")
    plt.show()

