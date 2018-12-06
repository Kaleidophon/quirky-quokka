"""
Implement functions to analyze the data gathered.
"""

# EXT
import numpy as np
from scipy.stats import ttest_ind
import torch

# PROJECT
from models import QNetwork


def test_difference(q_data: np.array, dq_data: np.array, p_threshold=0.05):
    """
    Test whether the difference in performance between the DQN and the Double DQN is significant
    using a Welch's t-test (testing whether the mean of samples for a certain time step is significantly different
    for the two models, NOT assuming equal variance).
    Input is expected to by a K x D matrix of K trials with D data points each.
    """
    _, p_values = ttest_ind(q_data, dq_data, axis=0, equal_var=False)

    sig = np.sum(p_values <= p_threshold)  # Number of time steps with significant differences
    percentage_sig = sig / len(p_values) * 100  # Percentage of those significant instances
    print(f"There is a significant difference for {sig}/{len(p_values)} ({percentage_sig:.2f} %) data points.")

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
