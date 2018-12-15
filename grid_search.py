"""
Defining Grid Search functionality
"""

from train import*

# STD
import itertools
import operator
import functools

# CONSTANTS
EPS = float(np.finfo(np.float32).eps)


def grid_search(ENVIRONMENTS, hyperparameter_options, num_episodes):

    # Let's run it!
    memory_size = 10000

    # init envs
    envs = {name: gym.envs.make(name) for name in ENVIRONMENTS}

    for _,env in enumerate(envs.values()):

        # Perform grid search over 5 random seeds
        for i in range(5):
            print("Seed:",i)
            best_sum_rewards = -np.inf
            # Perform Grid Search
            n_combinations = functools.reduce(operator.mul, [len(options) for options in hyperparameter_options.values()])
            for i, hyperparams in enumerate(itertools.product(*hyperparameter_options.values())):
                current_model_params = dict(zip(hyperparameter_options.keys(), hyperparams))

                #print(
                 #   "\rTrying out combination {}/{}: {}".format(
                  #      i + 1, n_combinations, str(current_model_params)
                 #   ), flush=True, end=""
                #)

                num_hidden = current_model_params['num_hidden']

                #  Single DQN
                memory = ReplayMemory(memory_size)
                n_out = env.action_space.n
                n_in = len(env.observation_space.low)
                model = QNetwork(n_in, n_out, num_hidden)
                episode_durations_single, cum_reward_single  = run_episodes(train, model, memory, env, num_episodes, **current_model_params)

                # Double DQN
                memory = ReplayMemory(memory_size)
                n_out = env.action_space.n
                n_in = len(env.observation_space.low)
                model = QNetwork(n_in, n_out, num_hidden)
                model_2 = QNetwork(n_in, n_out, num_hidden)

                episode_durations_double, cum_reward_double = run_episodes(train, model, memory, env, num_episodes, model_2 =model_2,**current_model_params)

                # Calculation best score to select best hyperparameters
                # best score = sum of cumulative rewards over all episodes and over Double DQN and DQN
                sum_rewards = cum_reward_double + cum_reward_single
                if sum_rewards > best_sum_rewards:
                    print("\n New highest score found ({:.4f})".format(sum_rewards))
                    best_sum_rewards = sum_rewards
                    best_parameters = current_model_params

            print("\n Found best parameters")
            print(str(best_parameters))
            print("Score:", best_sum_rewards )


if __name__ == "__main__":
    ENVIRONMENTS = ['Acrobot-v1']#[['MountainCar-v0'] 'MountainCarContinuous-v0'] # ['MountainCar-v0']  # 'CartPole-v1']]

    hyperparameter_opt_mountain_car = {
        "batch_size": [128],
        "discount_factor": [0.9, 0.99],
        "learn_rate": [0.01, 0.001],
        "num_hidden": [128],
        "update_target": [5, 10,20]
    }

    grid_search(ENVIRONMENTS,hyperparameter_opt_mountain_car, num_episodes=200)

