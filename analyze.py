"""
Implement functions to analyze the data gathered.
"""

# EXT
import numpy as np
from scipy.stats import mannwhitneyu, shapiro
import torch


def test_difference(q_data: np.array, dq_data: np.array, p_threshold=0.05):
    """
    Test whether the difference in performance between the DQN and the Double DQN is significant
    using a Welch's t-test (testing whether the mean of samples for a certain time step is significantly different
    for the two models, NOT assuming equal variance).
    Input is expected to by a K x D matrix of K trials with D data points each.
    """
    timesteps = q_data.shape[1]
    p_values = np.zeros(timesteps)
    num_non_rankable = 0

    for t in range(timesteps):
        try:
            _, p_values[t] = mannwhitneyu(q_data[:, t], dq_data[:, t], alternative="two-sided")
        except ValueError:
            # Sometimes it can happen that all samples from a distribution are identical, which produces
            # an error because elements have to be ranked -> Set p-value high and ignore this timestep
            p_values[t] = 1
            num_non_rankable += 1

    significant_timesteps = p_values <= p_threshold
    sig = np.sum(significant_timesteps)  # Number of time steps with significant differences
    percentage_sig = sig / len(p_values) * 100  # Percentage of those significant instances
    print(f"There is a significant difference for {sig}/{len(p_values)} ({percentage_sig:.2f} %) data points.")
    if num_non_rankable > 0:
        print(f"{num_non_rankable} timestep(s) couldn't be tested for significance.")

    significant_timesteps = np.where(significant_timesteps)[0]  # Remember the timesteps for plotting

    return p_values, significant_timesteps


def test_gaussian(data, p_threshold=0.05):
    """
    Test whether the distributions over multiple timesteps are normal distributions using a Shapiro-Wilk Test.
    """
    timesteps = data.shape[1]
    p_values = np.zeros(timesteps)

    for t in range(timesteps):
        _, p_values[t] = shapiro(data[:, t])

    significant_timesteps = p_values <= p_threshold
    sig = np.sum(significant_timesteps)  # Number of time steps with significant differences
    percentage_sig = sig / len(p_values) * 100  # Percentage of those significant instances
    print(f"{sig}/{len(p_values)} ({percentage_sig:.2f} %) distributions can be assumed to be gaussian.")

    return p_values


def get_actual_returns(env, models: list, discount_factor):
    """
    Calculate the actual average cumulative discounted returns for all visited states by using trained models
    and recording the actual returns.
    """
    def simulate_episode(model, env):
        returns = []
        state = env.reset()
        done = False

        while not done:
            # Select action
            actions = model(torch.Tensor(state))
            action = torch.argmax(actions).item()
            next_state, reward, done, _ = env.step(action)
            returns.append(reward)  # Remember encountered rewards

            # Prepare for next iter
            state = next_state

        return returns

    all_returns = []

    for model in models:
        returns = simulate_episode(model, env)

        G = 0  # Cumulative rewards
        for return_ in returns[::-1]:
            G = return_ + discount_factor * G
            all_returns.append(G)

    return sum(all_returns) / len(all_returns)
