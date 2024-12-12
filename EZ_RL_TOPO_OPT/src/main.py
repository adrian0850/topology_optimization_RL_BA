import design_space_funcs as dsf
import FEM as fem
import RL_Env as rl
import feature_extractor as fe
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback
import torch

num_envs = 8

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
    init_max_stress, init_max_strain, avg_u1, avg_u2, element_count, init_average_stress, init_average_strain, max_displacement_1, max_displacement_2, avg_strain_over_nodes = fem.FEM(a, b, c, d, plot_flag=True, grid=grid, device=device)

    bad = False
    good = False

    #bad = True
    good = True
    # Modify the grid
    if bad:
        dsf.remove_material(grid, -2, -1)
        print("Reward,\t", rl.get_reward(grid, init_max_stress, init_max_strain, init_average_stress, init_average_strain))
        dsf.remove_material(grid, -2, -2)
        print("Reward,\t", rl.get_reward(grid, init_max_stress, init_max_strain, init_average_stress, init_average_strain))
        dsf.remove_material(grid, -2, -3)
        print("Reward,\t", rl.get_reward(grid, init_max_stress, init_max_strain, init_average_stress, init_average_strain))
    elif good:
        dsf.remove_material(grid, 0, -1)
        print("Reward,\t", rl.get_reward(grid, init_max_stress, init_max_strain, init_average_stress, init_average_strain))
        dsf.remove_material(grid, 0, -2)
        print("Reward,\t", rl.get_reward(grid, init_max_stress, init_max_strain, init_average_stress, init_average_strain))
        dsf.remove_material(grid, 1, -1)
        print("Reward,\t", rl.get_reward(grid, init_max_stress, init_max_strain, init_average_stress, init_average_strain))
    else:
        grid[-0.5*height][-0.5*width] = 0

def Env_test():
    width = int(input("Enter grid width: "))
    height = int(input("Enter grid height: "))

    bounded = [(0, 0), (-1, 0)]
    loaded = [(-1, -0, "LY50")]

    env = rl.TopOptEnv(height, width, bounded, loaded)
    check_env(env, warn=True)

def reinforcement_learning_test():
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
    env = rl.TopOptEnv(height, width, bounded, loaded, mode="eval")
    model = PPO.load("ppo_topopt", env = env)
    obs, _ = env.reset()
    for i in range(20):
        action, _states = model.predict(obs)
        action = action.item()

        obs, rewards, dones, _, info = env.step(action)
        env.print()


def main():
    #Env_test()
    #FEM_test()
    reinforcement_learning_test()
    #load_test()

if __name__ == "__main__":
    main()

