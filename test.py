"""
Easy script to test models and render the corresponding environments.
"""

# STD
import random
import time
import sys

# EXT
import torch
import gym
from gym.spaces import Box

# PROJECT
from analyze import discrete_to_continuous
from train import ENVIRONMENTS


def select_greedy_action(model, state):
    with torch.no_grad():
        action = model(torch.Tensor(state))
        return torch.argmax(action).item()


def test_reinforce_model(model, env, num_episodes):
    for i in range(num_episodes):
        state = env.reset()
        time.sleep(1)
        env.render()
        done = False

        while not done:
            action = select_greedy_action(model, state)  # Greedy action
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
