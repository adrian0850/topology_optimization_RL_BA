import tests as test

NUM_ENVS = 10
NUM_TIMESTEPS = 1e6


def main():
    #test.env_compatability()
    test.fem_analysis_func(strat="good")
    #test.learn(NUM_ENVS)
    #test.loading(NUM_ENVS)
    #test.cnn()

if __name__ == "__main__":
    main()
