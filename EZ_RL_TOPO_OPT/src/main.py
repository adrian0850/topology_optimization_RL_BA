import design_space_funcs as dsf
import FEM as fem
import RL_Env as rl
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import ProgressBarCallback

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
    init_max_stress, init_max_strain, avg_u1, avg_u2, element_count, init_average_stress, init_average_strain, max_displacement_1, max_displacement_2, avg_strain_over_nodes = fem.FEM(a, b, c, d, plot_flag = True)

    bad = False
    good = True

    bad = False
    #good = True
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
    loaded = [(-1, -0, "LY50")]

    # Create the vectorized environment
    env = SubprocVecEnv([rl.make_env(height, width, bounded, loaded) for _ in range(num_envs)])
    #env = rl.TopOptEnv(height, width, bounded, loaded)
    # Set up TensorBoard logger
    log_dir = "./tensorboard_logs/"


    # Create the PPO model with the logger
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

    # Train the model
    model.learn(total_timesteps=100000, progress_bar=True)


    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()


def main():
    #Env_test()
    #FEM_test()
    reinforcement_learning_test()




    

if __name__ == "__main__":
    main()

