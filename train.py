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

# PROJECT
from models import ReplayMemory, QNetwork
from plotting import plot_exp_performance, plot_exps_with_intervals
from analyze import test_difference, get_actual_returns
from hyperparameters import HYPERPARAMETERS

# CONSTANTS
EPS = float(np.finfo(np.float32).eps)
ENVIRONMENTS = ['CartPole-v1', 'Acrobot-v1']


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


def run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate, model_2=None, update_target_q=10):
    optimizer = optim.Adam(model.parameters(), learn_rate)
    global_steps = 0  # Count the steps (do not reset at episode start, to compute epsilon)
    episode_durations = []

    for i in range(num_episodes):
        steps = 0
        state = env.reset()
        done = False
        while not done:
            steps += 1

            if model_2 is not None and steps % update_target_q == 0:
                model_2.load_state_dict(model.state_dict())

            eps = get_epsilon(global_steps)
            action = select_action(model, state, eps)
            next_state, reward, done, _ = env.step(action)
            memory.push((state, action, reward, next_state, done))
            state = next_state
            loss = train(model, memory, optimizer, batch_size, discount_factor, model_2)

        episode_durations.append(steps)
        global_steps += steps
    return episode_durations


def run_single_dqn(env, memory_size, num_hidden, batch_size, discount_factor, learn_rate, **hyper):
    memory = ReplayMemory(memory_size)
    n_out = env.action_space.n
    n_in = len(env.observation_space.low)
    model = QNetwork(n_in, n_out, num_hidden)
    episode_durations = run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate)
    return model, episode_durations


def run_double_dqn(env, memory_size, num_hidden, batch_size, discount_factor, learn_rate, update_target_q):
    memory = ReplayMemory(memory_size)
    n_out = env.action_space.n
    n_in = len(env.observation_space.low)
    model = QNetwork(n_in, n_out, num_hidden)
    model_2 = QNetwork(n_in, n_out, num_hidden)

    episode_durations = run_episodes(train, model, memory, env, num_episodes, batch_size, discount_factor, learn_rate,
                                     model_2, update_target_q)
    return model, episode_durations


if __name__ == "__main__":
    # Let's run it!
    num_episodes = 100

    # init envs
    envs = {name: gym.envs.make(name) for name in ENVIRONMENTS}
    # collect experiments
    exps = [('Single DQN', run_single_dqn), ('Double DQN', run_double_dqn)]

    # train
    #exp_results = [([(exp(env), exp_name) for exp_name, exp in exps], env_name) for env_name, env in envs.items()]
    #plot_exp_performance(exp_results, path="./img")

    env = envs["CartPole-v1"]
    hyperparams = HYPERPARAMETERS["CartPole-v1"]
    q_data = []
    q_models = []
    dq_data = []
    dq_models = []

    for run in range(5):
        print(f"Run #{run+1}...")
        q_model, q_score = run_single_dqn(env, **hyperparams)
        dq_model, dq_score = run_single_dqn(env, **hyperparams)

        q_models.append(q_model)
        q_data.append(q_score)
        dq_models.append(dq_model)
        dq_data.append(dq_score)

    q_data = np.stack(q_data)
    dq_data = np.stack(dq_data)

    true_q = get_actual_returns(env, q_models, hyperparams["discount_factor"])
    true_dq = get_actual_returns(env, dq_models, hyperparams["discount_factor"])
    print(true_q, true_dq)

    plot_exps_with_intervals(
        q_data, dq_data, title="CartPole Episode Durations", file_name="./img/test.png",
        smooth_curves=False, true_q=true_q, true_dq=true_dq  # Dummy values
    )
    test_difference(q_data, dq_data)


