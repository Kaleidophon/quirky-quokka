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
import copy

# PROJECT
from models import ReplayMemory, QNetwork
from plotting import create_plots_for_env
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


def get_epsilon(it):
    return 0.05 if it >= 1000 else - 0.00095 * it + 1


def select_action(model, state, epsilon):
    with torch.no_grad():
        action = model(torch.Tensor(state))
        return torch.argmax(action).item() if random.random() > epsilon else random.choice([0,1])


def compute_q_val(model, state, action):
    action_index = torch.stack(action.chunk(state.size(0)))
    return model(state).gather(1, action_index)


def compute_target(model, reward, next_state, done, discount_factor):
    targets = reward + (model(next_state).max(1)[0] * discount_factor) * (1-done.float())
    return targets.unsqueeze(1)


def train(model, memory, optimizer, batch_size, discount_factor, model_2):
    # don't learn without some decent experience
    if len(memory) < batch_size:
        return None

    # random transition batch is taken from experience replay memory
    transitions = memory.sample(batch_size)

    # transition is a list of 4-tuples, instead we want 4 vectors (as torch.Tensor's)
    state, action, reward, next_state, done = zip(*transitions)

    # convert to PyTorch and define types
    state = torch.tensor(state, dtype=torch.float)
    action = torch.tensor(action, dtype=torch.int64)  # Need 64 bit to use them as index
    next_state = torch.tensor(next_state, dtype=torch.float)
    reward = torch.tensor(reward, dtype=torch.float)
    done = torch.tensor(done, dtype=torch.uint8)  # Boolean
    action = action.squeeze()

    # compute the q value
    q_val = compute_q_val(model, state, action)

    with torch.no_grad():  # Don't compute gradient info for the target (semi-gradient)
        model_ = model_2 if model_2 is not None else model
        target = compute_target(model_, reward, next_state, done, discount_factor)

    # loss is measured from error between current and newly expected Q values
    loss = F.smooth_l1_loss(q_val, target)

    # backpropagation of loss to Neural Network (PyTorch magic)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()  # Returns a Python scalar, and releases history (similar to .detach())


def run_episodes(train, model, memory, env, num_episodes, copy_mode, batch_size, discount_factor, learn_rate, model_2=None, update_target_q=10, max_steps=1000):
    optimizer1 = optim.Adam(model.parameters(), learn_rate)
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []
    q_vals = []
    episode_rewards = []

    if model_2 is not None:
        model_2.load_state_dict(model.state_dict())
        optimizer2 = optim.Adam(model_2.parameters(), learn_rate)

    for i in range(num_episodes):
        steps = 0
        state = env.reset()
        episode_q_vals = []
        cum_reward = 0
        done = False
        while not done:
            steps += 1
            flip = random.random()

            eps = get_epsilon(global_steps)

            if copy_mode or (flip > 0.5 or model_2 is None):
                action = select_action(model, state, eps)

                if model_2 is None and steps % 10 == 0:
                   model_2 = copy.deepcopy(model)

                q_val = compute_q_val(model, torch.tensor([state], dtype=torch.float), torch.tensor([action], dtype=torch.int64))
                train(model, memory, optimizer1, batch_size, discount_factor, model_2)

            else:
                action = select_action(model_2, state, eps)

                q_val = compute_q_val(model_2, torch.tensor([state], dtype=torch.float),
                                      torch.tensor([action], dtype=torch.int64))
                train(model_2, memory, optimizer2, batch_size, discount_factor, model)

            episode_q_vals.append(q_val.detach().numpy().squeeze().tolist())

            # only convert to continuous action when actually performing an action in the envs
            action_env = action

            if isinstance(env.action_space, Box):
                action_env = [discrete_to_continuous(action, env)]

            next_state, reward, done, _ = env.step(action_env)

            cum_reward += reward

            if "MountainCar" in type(env.unwrapped).__name__:
                # If environment is MountainCar, adjust rewards
                # Adjust reward based on car position
                reward = state[0] + 0.5

                # Adjust reward for task completion
                if state[0] >= 0.5:
                    reward += 1

            memory.push((state, action, reward, next_state, done))
            state = next_state

            if steps >= max_steps:
                done = True

        q_vals.append(np.mean(episode_q_vals))
        episode_durations.append(steps)
        global_steps += steps
        episode_rewards.append(cum_reward)

    return episode_durations, q_vals, episode_rewards


def run_single_dqn(env, num_episodes, copy_mode,memory_size, num_hidden, batch_size, discount_factor, learn_rate):
    memory = ReplayMemory(memory_size)

    # continuous action space
    if isinstance(env.action_space, Box):
        dims = env.action_space.shape[0]
        n_out = SPLITS ** dims
    # discrete action space
    else:
        n_out = env.action_space.n

    n_in = len(env.observation_space.low)
    model = QNetwork(n_in, n_out, num_hidden)
    episode_durations, q_vals, cum_reward = run_episodes(train, model, memory, env, num_episodes, copy_mode, batch_size, discount_factor, learn_rate)

    return model, episode_durations, q_vals, cum_reward


def run_double_dqn(env, num_episodes, copy_mode, memory_size, num_hidden, batch_size, discount_factor, learn_rate):
    memory = ReplayMemory(memory_size)

    # continuous action space
    if isinstance(env.action_space, Box):
        dims = env.action_space.shape[0]
        n_out = SPLITS ** dims
    # discrete action space
    else:
        n_out = env.action_space.n

    n_in = len(env.observation_space.low)
    model = QNetwork(n_in, n_out, num_hidden)
    model_2 = QNetwork(n_in, n_out, num_hidden)

    episode_durations, q_vals, cum_reward = run_episodes(train, model, memory, env, num_episodes, copy_mode, batch_size, discount_factor, learn_rate, model_2)
    return model, episode_durations, q_vals, cum_reward


if __name__ == "__main__":

    # Init envs
    envs = {name: gym.envs.make(name) for name in ENVIRONMENTS}

    # Collect experiments
    exps = [('Single DQN', run_single_dqn), ('Double DQN', run_double_dqn)]

    for env_name, env in envs.items():
        create_plots_for_env(
            env_name, env, HYPERPARAMETERS[env_name], image_path="./img/", k=2, copy_mode=True,
            dqn_experiment=run_single_dqn, ddqn_experiment=run_double_dqn
        )
