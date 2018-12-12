"""
Defining training functions and experiments for the project.
"""

# STD
import random

# EXT
import numpy as np
import torch
import torch.optim as optim
import gym
import torch.nn.functional as F
from gym.spaces import Box
import time
import sys

# PROJECT
from models import ReplayMemory, QNetwork
from train import create_data_for_env
from hyperparameters import HYPERPARAMETERS

# CONSTANTS
EPS = float(np.finfo(np.float32).eps)
ENVIRONMENTS = ["Pendulum-v0", "Acrobot-v1", "MountainCar-v0", "CartPole-v1"]
SPLITS = 9  # TODO: Pass this as argument


def discrete_to_continuous(index, env):
    dims = env.action_space.shape[0]
    idx = index
    low = env.action_space.low
    high = env.action_space.high
    interval = high - low
    continuous_actions = []

    for i in range(dims):
        rem = idx % SPLITS
        idx = int(idx / SPLITS)
        continuous_actions.append(rem)

    continuous_actions = (np.array(continuous_actions) / (SPLITS - 1)) * interval + low

    return continuous_actions[0]


def select_action(model, state, epsilon):
    with torch.no_grad():
        action = model(torch.Tensor(state))
        return torch.argmax(action).item() if random.random() > epsilon else random.choice([0,1])


def test_reinforce_model(model, env, num_episodes):
    for i in range(num_episodes):
        state = env.reset()
        time.sleep(1)
        env.render()
        done = False

        while not done:
            action = select_action(model, state, 0)  # Greedy action
            if isinstance(env.action_space, Box):
                action = [discrete_to_continuous(action, env)]
            next_state, reward, done, _ = env.step(action)
            env.render()
            state = next_state
            time.sleep(0.05)
        env.close()


if __name__ == "__main__":
    filename = sys.argv[1]
    print(filename)
    # Init envs
    envs = {name: gym.envs.make(name) for name in ENVIRONMENTS}
    env_name = filename.split('_')[0]
    model = torch.load(filename)
    env = envs[env_name]
    test_reinforce_model(model, env, num_episodes=10)
